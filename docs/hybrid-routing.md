# Hybrid language routing and offline fallback

Start Ollama and configure `.env` in the project root:

```dotenv
DEFAULT_PROVIDER=hybrid
LOCAL_MODEL=qwen3.5:2b
LOCAL_API=ollama
OPENAI_MODEL=gpt-6-luna
OPENAI_API_KEY=your-key
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=openai
WEB_SEARCH_MODEL=gpt-6-luna
HYBRID_ROUTER_TIMEOUT_SECONDS=10
HYBRID_OFFLINE_RETRY_SECONDS=30
```

Keep existing keys and other settings. Restart the assistant after changes.

```powershell
python -m patrick text --provider hybrid
python -m patrick voice --provider hybrid --input-device 2 --output-device 5 --headphones --no-barge-in
```

Device indices are machine-specific. Omitting `--output-device` uses the default output. Wake phrase: **Hey Patrick**. Spoken identity: **Patrick**.

## One turn

1. Shared normalization corrects company sound-alikes to Revobots before retrieval and generation in direct local and hybrid text/voice paths. Covered families include riverbots, rivabots, rivobots, revabots, reevobots, rewo bots, riverboards, riva boards, revo boards, revo bords, riverbods and revo borts. Case, spaces, hyphens, doubled consonants and singular endings are handled. Ordinary words such as robots, boards and riverboats remain unchanged. Speech recognition can produce unpredictable spellings, so this is a bounded family of variants rather than a claim to recognize every possible transcription.
2. Ollama makes a short structured decision: question language, expected answer language, whether public search is needed, and a public search query. It reads the question and up to two recent exchanges. It uses no cloud and receives no RAG excerpts. JSON output, `think=false`, temperature 0 and a 160-token limit keep the decision small. The same context size as generation avoids changing Ollama's context allocation. The answer still uses the configured temperature (0.4).
3. English question + English expected answer selects Ollama first. English requesting Hindi/Japanese/another language, non-English, mixed, or uncertain input selects OpenAI first. Explicit output-language requests override the input language for the reply. Missing cloud credentials leave only local available. The classifier is a small model and can make mistakes; inspect the `Hybrid input=... answer=... needs_web=... primary=...` log.
4. A planned public search runs in the answer worker before generation. General stable explanations need not search; explicit lookups and current/latest facts should. For example, a general physical-AI explanation can stay local, while the latest NVIDIA physical-AI news needs search. Both routes retain their ordinary bounded search/knowledge tool loop. Public search uses the configured search service; an English answer still comes from Ollama when search uses OpenAI.
5. `Ummm, let me check.` plays before search and is excluded from answer history. Search results are untrusted reference data. Private RAG excerpts remain local and must not be put into public search queries.
6. A cloud transport error falls back to Ollama. Search transport failure during local generation keeps Ollama running; during cloud generation it switches to Ollama. Current information that cannot be verified must be disclosed as unverified. No offline web search is simulated.

## Timeouts, cancellation and memory

`HYBRID_ROUTER_TIMEOUT_SECONDS` bounds the classification phase (default 10). Failure, malformed JSON, or timeout selects the conservative OpenAI-then-Ollama route. A routing worker still finishing a cancelled question is bounded; a new question need not wait for it and may take this conservative route. All classification and generation run away from the microphone/control loop. Late routing results and stale generation events are ignored.

`PROVIDER_TIMEOUT_SECONDS` controls each cloud answer attempt and `LOCAL_TIMEOUT_SECONDS` each local attempt. A cloud deadline temporarily treats the service as unreachable, which does not prove the entire internet is down. Network errors and cloud deadlines disable cloud/search for `HYBRID_OFFLINE_RETRY_SECONDS` (default 30), then a later question retries according to its language. HTTP authentication/quota failures fall back without marking the internet offline. A local server connection failure does not mark the cloud offline. No internet probe is performed before every question.

Successful session history is shared when switching backends; RAG stays available. Existing `SESSION_MEMORY_EXCHANGES` and idle-session reset rules still apply. Final answers are buffered to completion to avoid speaking conflicting partial answers from different backends. Progress speech passes through immediately. Stop/cancellation invalidates both routing and generation, discards pending output and never triggers fallback.

Explicit `--provider local` retains normalization, RAG and model-selected search without language classification. `--provider openai` directly selects OpenAI. `auto` retains its legacy Gemini/OpenAI/local ordering.

## Code reference

| Module / entry point | Responsibility |
|---|---|
| `text_normalization.correct_terms(text)` | Shared company-name correction, also used by `transcription.TranscriptionProvider` |
| `turn_planning.TurnPlan` | Validated language/search decision and per-turn web availability |
| `turn_planning.OllamaTurnPlanner.plan(text, history, cancelled)` | Short structured local classification; supports native Ollama and compatible local JSON responses |
| `hybrid.HybridProvider.submit()` / `poll()` | Asynchronous planning, stale-decision rejection and provider order |
| `hybrid.HybridProvider._route()` / `_failure()` | Language policy and temporary offline cooldown |
| `hybrid.HybridProvider.cancel()` / `reset_session()` | Cancel routing or generation; clear all configured backend sessions |
| `failover.FailoverProvider._start()` / `_next()` / `poll()` | Bounded attempts, shared history, coherent final answers, immediate progress notices |
| `providers.ChatProvider.submit_planned()` | Normalize before retrieval, apply requested language and offline instructions |
| `providers.ChatProvider._run()` / `_search_notice()` | Planned search, bounded ordinary tool loop, streaming, progress and answer history |
| `tools.SearchTools.execute()` | Safe search results/errors; transport failures carry `network_unavailable` |
| `ports.Response.notice` / `network_unavailable` | Progress separate from answer text; transport state without request secrets |
| `tests/test_hybrid.py` | Routing, correction, search, offline policy, timeouts, cancellation and history regressions |

Tests use fake cloud/search transports. Live local classification checks validate example routing; they do not guarantee perfect language detection, web-tool choice or factual accuracy for every question.
