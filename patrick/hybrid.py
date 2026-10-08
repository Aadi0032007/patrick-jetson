"""Per-turn language routing, asynchronous local planning and offline fallback."""
from dataclasses import replace
import logging
import math
import queue
import threading
from .failover import FailoverProvider
from .text_normalization import correct_terms
from .turn_planning import TurnPlan


class HybridProvider(FailoverProvider):
    def __init__(self, providers, planner, router_timeout=10, offline_retry=30, **kwargs):
        super().__init__(providers, **kwargs)
        if "local" not in dict(providers) or any(name not in ("local", "openai") for name, _ in providers):
            raise ValueError("Hybrid requires a local backend and optionally OpenAI")
        if not all(math.isfinite(x) and x > 0 for x in (router_timeout, offline_retry)):
            raise ValueError("Hybrid routing timeout and offline retry must be positive and finite")
        self.configured_providers = list(providers)
        self.planner = planner
        self.router_timeout, self.offline_retry = router_timeout, offline_retry
        self.offline_until = 0
        self.routing = None
        # One worker plus submit-time draining bounds retained decisions without
        # allowing a stale result to occupy the only slot and hide a fresh result.
        self.decisions = queue.Queue()
        self.routing_slot = threading.BoundedSemaphore(1)

    def submit(self, turn_id, text):
        if self.turn_id is not None:
            self.cancel(self.turn_id)
        self.turn_id, self.text, self.index = turn_id, correct_terms(text), 0
        self.pending_events, self.buffer = [], []
        self.plan = None
        while not self.decisions.empty():
            try:
                self.decisions.get_nowait()
            except queue.Empty:
                break
        token = threading.Event()
        self.routing = token
        self.started = self.clock()
        history = [dict(m) for m in self.history]
        question = self.text
        if not self.routing_slot.acquire(blocking=False):
            self._route(TurnPlan())
            return

        def run():
            try:
                plan = self.planner.plan(question, history, token)
            except Exception:
                logging.getLogger("patrick.providers").warning(
                    "Hybrid decision unavailable; using conservative cloud routing with local fallback")
                plan = TurnPlan()
            try:
                if not token.is_set():
                    self.decisions.put_nowait((token, plan))
            finally:
                self.routing_slot.release()
        threading.Thread(target=run, daemon=True).start()

    def _route(self, plan):
        self.routing.set()
        self.routing = None
        offline = self.clock() < self.offline_until
        self.plan = replace(plan, web_available=not offline)
        order = ("local",) if offline else ("local", "openai") if plan.english else ("openai", "local")
        by_name = dict(self.configured_providers)
        self.providers = [(name, by_name[name]) for name in order if name in by_name]
        logging.getLogger("patrick.providers").info(
            "Hybrid input=%s answer=%s needs_web=%s offline=%s primary=%s",
            plan.input_language, plan.answer_language, plan.needs_web, offline, self.providers[0][0])
        self._start()

    def poll(self):
        if self.routing is not None:
            while True:
                try:
                    token, plan = self.decisions.get_nowait()
                except queue.Empty:
                    break
                if token is self.routing:
                    self._route(plan)
                    break
            if self.routing is not None:
                if self.clock() - self.started >= self.router_timeout:
                    logging.getLogger("patrick.providers").warning("Hybrid routing deadline exceeded")
                    self._route(TurnPlan())
                else:
                    return []
        return super().poll()

    def _failure(self, name, reason, network_unavailable):
        if network_unavailable:
            self.offline_until = self.clock() + self.offline_retry
            logging.getLogger("patrick.providers").warning(
                "Cloud/search unreachable; using local generation and disabling search for %.0fs", self.offline_retry)

    def cancel(self, turn_id):
        if self.turn_id == turn_id and self.routing is not None:
            self.routing.set()
            self.routing = None
            self.turn_id = None
            self.pending_events, self.buffer = [], []
            return
        super().cancel(turn_id)

    def reset_session(self):
        self.history.clear()
        for _, provider in self.configured_providers:
            reset = getattr(provider, "reset_session", None)
            if reset:
                reset()
