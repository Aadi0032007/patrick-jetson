"""One active turn, ordered provider attempts, coherent responses and bounded waits."""
from dataclasses import replace
import logging
import math
import time
from .ports import Response


class FailoverProvider:
    def __init__(self, providers, timeout=20, local_timeout=60, clock=time.monotonic):
        if not providers:
            raise ValueError("Configure at least one provider")
        if not all(math.isfinite(value) and value > 0 for value in (timeout, local_timeout)):
            raise ValueError("Provider timeouts must be positive")
        self.providers = providers
        self.timeout, self.local_timeout, self.clock = timeout, local_timeout, clock
        self.turn_id = None
        self.attempt_id = 0
        self.index = 0
        self.buffer = []
        self.pending_events = []
        self.history = []
        self.text = ""
        self.started = 0
        self.last_provider = None
        self.plan = None

    def submit(self, turn_id, text):
        if self.turn_id is not None:
            self.cancel(self.turn_id)
        self.turn_id, self.text, self.index = turn_id, text, 0
        self.pending_events = []
        self._start()

    def _start(self):
        self.buffer = []
        self.attempt_id += 1
        self.started = self.clock()
        name, provider = self.providers[self.index]
        logging.getLogger("patrick.providers").info("Attempting provider=%s model=%s", name, getattr(provider, "model", "unknown"))
        try:
            # Share the successful session context when switching backends.
            if hasattr(provider, "history"):
                with provider.lock:
                    provider.history = [dict(message) for message in self.history]
                    provider.history_turns = [None] * (len(self.history) // 2)
            submit_planned = getattr(provider, "submit_planned", None)
            if self.plan is not None and submit_planned:
                submit_planned(self.attempt_id, self.text, self.plan)
            else:
                provider.submit(self.attempt_id, self.text)
        except Exception:
            self._next("submit failed")

    def _next(self, reason, network_unavailable=False):
        name, provider = self.providers[self.index]
        if network_unavailable and self.plan is not None:
            self.plan = replace(self.plan, web_available=False)
        self._failure(name, reason, network_unavailable)
        logging.getLogger("patrick.providers").warning("Provider=%s failed (%s); trying next", name, reason)
        try:
            provider.cancel(self.attempt_id)
        except Exception:
            pass
        self.buffer = []
        self.index += 1
        if self.index == len(self.providers):
            self.pending_events.append(Response(self.turn_id, error="All configured providers failed. Check credentials, models and local server."))
            self.turn_id = None
        else:
            self._start()

    def _failure(self, name, reason, network_unavailable):
        """Optional per-policy failure notification."""

    def poll(self):
        if self.turn_id is None:
            events, self.pending_events = self.pending_events, []
            return events
        name, provider = self.providers[self.index]
        try:
            events = provider.poll()
        except Exception:
            self._next("poll failed")
            return self._drain_pending()
        for event in events:
            if event.turn_id != self.attempt_id:
                continue
            if event.error:
                self._next("request failed", event.network_unavailable)
                return self._drain_pending()
            if event.notice:
                self.pending_events.append(replace(event, turn_id=self.turn_id))
                if event.network_unavailable:
                    if self.plan is not None:
                        self.plan = replace(self.plan, web_available=False)
                    self._failure(name, "search unavailable", True)
                continue
            if event.text or event.pcm:
                self.buffer.append(event)
                if sum(len(r.text) + len(r.pcm) for r in self.buffer) > 65536:
                    self._next("response too large")
                    return self._drain_pending()
            if event.final:
                answer = "".join(r.text for r in self.buffer)
                if not answer.strip() and not any(r.pcm for r in self.buffer):
                    self._next("empty response")
                    return self._drain_pending()
                # Commit only a complete successful answer, avoiding mixed spoken replies
                # if an upstream stream fails after producing some tokens.
                if any(r.pcm for r in self.buffer):
                    result = [replace(r, turn_id=self.turn_id, final=False) for r in self.buffer]
                    result.append(Response(self.turn_id, final=True))
                else:
                    result = [Response(self.turn_id, text=answer, final=True)]
                self.history = [dict(message) for message in getattr(provider, "history", [])]
                self.last_provider = name
                logging.getLogger("patrick.providers").info("Completed provider=%s model=%s", name, getattr(provider, "model", "unknown"))
                self.turn_id = None
                self.buffer = []
                return self._drain_pending() + result
        limit = self.local_timeout if name == "local" else self.timeout
        if self.clock() - self.started >= limit:
            self._next("deadline exceeded", network_unavailable=name != "local")
        return self._drain_pending()

    def _drain_pending(self):
        events, self.pending_events = self.pending_events, []
        return events

    def reset_session(self):
        self.history.clear()
        for _, provider in self.providers:
            reset = getattr(provider, "reset_session", None)
            if reset:
                reset()

    def cancel(self, turn_id):
        if self.turn_id != turn_id:
            return
        # Invalidate before invoking the active transport; no fallback on user stop.
        self.turn_id = None
        self.buffer = []
        self.pending_events = []
        try:
            self.providers[self.index][1].cancel(self.attempt_id)
        except Exception:
            logging.getLogger("patrick.providers").warning("Transport cancellation failed")
