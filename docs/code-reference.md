# Code reference

Current Python module/function index. Read the architecture in the project README first; inspect linked source for implementation details. Nested callbacks appear under their containing callable.

## `__init__.py`

Patrick's local-first conversation runtime.

| Function | Purpose |
|---|---|

## `__main__.py`

Application module.

| Function | Purpose |
|---|---|

## `audio.py`

Optional microphone adapter. Audio dependencies are loaded only in voice mode.

| Function | Purpose |
|---|---|
| [`SystemProcessedAudio.capture()`](../patrick/audio.py#L18) | Returns PCM unchanged. Actual AEC/noise suppression must already be applied by the OS/hardware. |
| [`SystemProcessedAudio.reference()`](../patrick/audio.py#L21) | No-op reference hook; OS routing supplies the playback reference externally. |
| [`devices()`](../patrick/audio.py#L25) | Prints PortAudio input/output devices and defaults. |
| [`run_microphone()`](../patrick/audio.py#L30) | Loads speech adapters, opens the microphone, consumes queued frames, detects wake/stop/endpoints and ticks the controller. |
| [`run_microphone.callback()`](../patrick/audio.py#L61) | PortAudio callback: timestamps/copies frames and flags status/queue overflow without network or synthesis work. |

## `chatterbox_speech.py`

Chatterbox TTS in a persistent worker with independent speech dependencies.

| Function | Purpose |
|---|---|
| [`prepare_text_assets()`](../patrick/chatterbox_speech.py#L26) | Prefetch the upstream tokenizer's mapping cache and Chinese segmenter. |
| [`check_v3_support()`](../patrick/chatterbox_speech.py#L41) | Internal implementation; inspect the linked source in its module context. |
| [`load_model()`](../patrick/chatterbox_speech.py#L49) | Load the official multilingual V3 checkpoint with the real Perth watermarker. |
| [`ChatterboxSynthesizer.__init__()`](../patrick/chatterbox_speech.py#L91) | Internal implementation; inspect the linked source in its module context. |
| [`ChatterboxSynthesizer._read()`](../patrick/chatterbox_speech.py#L108) | Internal implementation; inspect the linked source in its module context. |
| [`ChatterboxSynthesizer._read.read_line()`](../patrick/chatterbox_speech.py#L110) | Internal implementation; inspect the linked source in its module context. |
| [`ChatterboxSynthesizer.synthesize()`](../patrick/chatterbox_speech.py#L128) | Internal implementation; inspect the linked source in its module context. |
| [`ChatterboxSynthesizer.close()`](../patrick/chatterbox_speech.py#L143) | Internal implementation; inspect the linked source in its module context. |
| [`worker()`](../patrick/chatterbox_speech.py#L156) | Internal implementation; inspect the linked source in its module context. |

## `cli.py`

Application module.

| Function | Purpose |
|---|---|
| [`main()`](../patrick/cli.py#L13) | Parses CLI, loads config/environment, constructs adapters, runs the selected mode and cleans up. |
| [`run_text()`](../patrick/cli.py#L77) | Runs a responsive text simulation while processing controller ticks and console commands. |
| [`run_text.read()`](../patrick/cli.py#L81) | Background console reader; places lines/EOF in a queue. |
| [`demo()`](../patrick/cli.py#L117) | Runs deterministic synthetic wake/pause/barge-in/stop/session assertions. |
| [`demo.DemoProvider.__init__()`](../patrick/cli.py#L121) | Initializes demo.DemoProvider state and its dependencies. |
| [`demo.DemoProvider.submit()`](../patrick/cli.py#L123) | Records a submitted test request without generating an answer. |
| [`demo.DemoProvider.cancel()`](../patrick/cli.py#L125) | Records a cancelled test turn. |
| [`demo.DemoProvider.poll()`](../patrick/cli.py#L127) | Drains injected synthetic response events. |
| [`demo.DemoPlayback.enqueue()`](../patrick/cli.py#L132) | Prints a demo response and marks playback busy to exercise state transitions. |

## `config.py`

Application module.

| Function | Purpose |
|---|---|
| [`Config.__post_init__()`](../patrick/config.py#L28) | Validates durations, silence ordering, sample rate/frame size, VAD range, booleans and required phrases. |
| [`Config.load()`](../patrick/config.py#L52) | Reads TOML, rejects unknown keys, converts emergency phrase lists to tuples and constructs Config. |

## `controller.py`

Application module.

| Function | Purpose |
|---|---|
| [`Metrics.__init__()`](../patrick/controller.py#L17) | Initializes Metrics state and its dependencies. |
| [`Metrics.emit()`](../patrick/controller.py#L20) | Clamps/rounds timing, records details, retains the latest 1,000 records and logs them. |
| [`Controller.__init__()`](../patrick/controller.py#L29) | Initializes Controller state and its dependencies. |
| [`Controller._state()`](../patrick/controller.py#L48) | Changes state only when necessary; records/logs bounded transition history. |
| [`Controller._listen()`](../patrick/controller.py#L59) | Resets endpoint/speech confirmation and starts a fresh listening window. |
| [`Controller._cancel()`](../patrick/controller.py#L66) | Invalidates the active turn, stops playback, cancels provider work and clears generation completion. |
| [`Controller.emergency_stop()`](../patrick/controller.py#L79) | Stops simulated movement/actions first, cancels speech/generation, resumes listening and optionally beeps. |
| [`Controller.input()`](../patrick/controller.py#L90) | Handles stop before other behavior, wake activation, ordinary barge-in confirmation, transcript updates and ticks. |
| [`Controller.provider_endpoint()`](../patrick/controller.py#L159) | Optional cloud semantic endpoint; local fallback continues independently. |
| [`Controller._submit()`](../patrick/controller.py#L164) | Creates a new active turn, records endpoint latency and starts provider work; handles submit failure. |
| [`Controller.tick()`](../patrick/controller.py#L186) | Drains provider/audio-start events, rejects stale turns, queues speech, completes turns and checks endpoint/session timers. |
| [`Controller.close()`](../patrick/controller.py#L220) | Stops robot/actions, cancels active work, resets endpoint and returns to idle. |

## `endpoint.py`

Application module.

| Function | Purpose |
|---|---|
| [`normalize()`](../patrick/endpoint.py#L5) | Lowercases and extracts alphanumeric/apostrophe words for phrase matching. |
| [`PhraseWakeDetector.__init__()`](../patrick/endpoint.py#L12) | Initializes PhraseWakeDetector state and its dependencies. |
| [`PhraseWakeDetector.detect()`](../patrick/endpoint.py#L15) | Matches the configured normalized phrase with word boundaries. |
| [`Endpoint.__init__()`](../patrick/endpoint.py#L23) | Initializes Endpoint state and its dependencies. |
| [`Endpoint.reset()`](../patrick/endpoint.py#L27) | Clears utterance timestamps, text and final flag. |
| [`Endpoint.update()`](../patrick/endpoint.py#L33) | Records confirmed speech timing and nonempty ASR text/finality. |
| [`Endpoint.threshold()`](../patrick/endpoint.py#L42) | Chooses command, complete, incomplete or unknown silence using conservative word heuristics. |
| [`Endpoint.ready()`](../patrick/endpoint.py#L53) | Checks maximum utterance duration or elapsed end-of-speech silence. |

## `environment.py`

Load a simple project .env without overriding existing shell variables.

| Function | Purpose |
|---|---|
| [`load_env()`](../patrick/environment.py#L7) | Parses a simple optional UTF-8 `.env` and fills only environment variables absent from the shell. |

## `failover.py`

One active turn, ordered provider attempts, coherent responses and bounded waits.

| Function | Purpose |
|---|---|
| [`FailoverProvider.__init__()`](../patrick/failover.py#L10) | Initializes FailoverProvider state and its dependencies. |
| [`FailoverProvider.submit()`](../patrick/failover.py#L28) | Cancels a previous active turn, stores the current question and starts the first candidate. |
| [`FailoverProvider._start()`](../patrick/failover.py#L35) | Creates a fresh attempt ID/deadline and submits to the selected provider. |
| [`FailoverProvider._next()`](../patrick/failover.py#L55) | Cancels the failed attempt, clears partial buffered output and advances or emits a final error. |
| [`FailoverProvider._failure()`](../patrick/failover.py#L73) | Optional per-policy failure notification. |
| [`FailoverProvider.poll()`](../patrick/failover.py#L76) | Processes attempt events, buffers/commits one complete answer, enforces deadlines and drives fallback. |
| [`FailoverProvider._drain_pending()`](../patrick/failover.py#L127) | Drains wrapper-generated pending events. |
| [`FailoverProvider.reset_session()`](../patrick/failover.py#L131) | Internal implementation; inspect the linked source in its module context. |
| [`FailoverProvider.cancel()`](../patrick/failover.py#L138) | Cancels the matching active attempt and invalidates buffered/pending output without fallback. |

## `hybrid.py`

Per-turn language routing, asynchronous local planning and offline fallback.

| Function | Purpose |
|---|---|
| [`HybridProvider.__init__()`](../patrick/hybrid.py#L13) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider.submit()`](../patrick/hybrid.py#L29) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider.submit.run()`](../patrick/hybrid.py#L49) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider._route()`](../patrick/hybrid.py#L63) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider.poll()`](../patrick/hybrid.py#L76) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider._failure()`](../patrick/hybrid.py#L94) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider.cancel()`](../patrick/hybrid.py#L100) | Internal implementation; inspect the linked source in its module context. |
| [`HybridProvider.reset_session()`](../patrick/hybrid.py#L109) | Internal implementation; inspect the linked source in its module context. |

## `knowledge.py`

Small local BM25 reference retriever; no model downloads or external requests.

| Function | Purpose |
|---|---|
| [`tokens()`](../patrick/knowledge.py#L11) | Extracts lowercase word/number terms, dropping common English stopwords; no domain aliases. |
| [`KnowledgeBase.__init__()`](../patrick/knowledge.py#L16) | Reads page JSON, removes page furniture, chunks words and builds term/document-frequency statistics. |
| [`KnowledgeBase.retrieve()`](../patrick/knowledge.py#L31) | Scores chunks with BM25-style ranking and returns top positive matches with source/page metadata. |
| [`KnowledgeTools.__init__()`](../patrick/knowledge.py#L51) | Internal implementation; inspect the linked source in its module context. |
| [`KnowledgeTools.execute()`](../patrick/knowledge.py#L65) | Internal implementation; inspect the linked source in its module context. |

## `languages.py`

Language metadata and local output-language selection for multilingual TTS.

| Function | Purpose |
|---|---|
| [`language_code()`](../patrick/languages.py#L12) | Internal implementation; inspect the linked source in its module context. |
| [`SpeechLanguageRouter.__init__()`](../patrick/languages.py#L24) | Internal implementation; inspect the linked source in its module context. |
| [`SpeechLanguageRouter.choose()`](../patrick/languages.py#L30) | Internal implementation; inspect the linked source in its module context. |

## `local.py`

Application module.

| Function | Purpose |
|---|---|
| [`LocalProvider.__init__()`](../patrick/local.py#L6) | Initializes LocalProvider state and its dependencies. |
| [`LocalProvider.submit()`](../patrick/local.py#L9) | Produces a deterministic offline stub response, explicitly avoiding invented battery telemetry. |
| [`LocalProvider.cancel()`](../patrick/local.py#L16) | Drops queued stub responses for the specified turn. |
| [`LocalProvider.poll()`](../patrick/local.py#L19) | Drains queued stub responses. |
| [`SimRobot.__init__()`](../patrick/local.py#L25) | Initializes SimRobot state and its dependencies. |
| [`SimRobot.stop_movement()`](../patrick/local.py#L30) | Sets simulated moving state false and increments a test counter. |
| [`SimRobot.cancel_actions()`](../patrick/local.py#L34) | Increments the simulated action-cancellation counter. |
| [`ConsolePlayback.__init__()`](../patrick/local.py#L39) | Initializes ConsolePlayback state and its dependencies. |
| [`ConsolePlayback.enqueue()`](../patrick/local.py#L42) | Prints text as Patrick without physical audio playback. |
| [`ConsolePlayback.stop()`](../patrick/local.py#L46) | Clears simulated busy state. |
| [`ConsolePlayback.beep()`](../patrick/local.py#L49) | Prints the acknowledgement marker. |

## `ollama_settings.py`

Validated, configurable Ollama sampling and model context discovery.

| Function | Purpose |
|---|---|
| [`options()`](../patrick/ollama_settings.py#L8) | Validates native sampling/output/context environment values; queries model metadata if context is full. |

## `playback.py`

Interruptible local TTS and PCM output, independent of model generation.

| Function | Purpose |
|---|---|
| [`VoicePlayback.__init__()`](../patrick/playback.py#L23) | Loads selected TTS, allocates queues/epoch/condition and starts synthesis and playback workers. |
| [`VoicePlayback.busy()`](../patrick/playback.py#L53) | Reports active synthesis/playback or queued work under the condition lock. |
| [`VoicePlayback.enqueue()`](../patrick/playback.py#L57) | Logs original text, makes speech phrases, bounds the pending queue and wakes workers. |
| [`VoicePlayback.beep()`](../patrick/playback.py#L71) | Generates a short local sine-wave PCM acknowledgement and queues it with turn -1. |
| [`VoicePlayback.stop()`](../patrick/playback.py#L76) | Increments epoch, clears all queued audio/events, terminates system-TTS subprocess and aborts speaker stream. |
| [`VoicePlayback.drain_started()`](../patrick/playback.py#L88) | Drains actual audio-start notifications for controller metrics/state. |
| [`VoicePlayback._synthesize()`](../patrick/playback.py#L94) | Calls Chatterbox V3 or runs the fixed Windows/Linux system-TTS command, respecting epoch cancellation/timeouts. |
| [`VoicePlayback._prepare_worker()`](../patrick/playback.py#L136) | Waits for pending work/prepared-queue capacity, synthesizes WAV or accepts PCM, converts/checks format and queues current-epoch audio. |
| [`VoicePlayback._worker()`](../patrick/playback.py#L170) | Opens output stream, writes PCM blocks, reports first audio and discards stopped/stale work. |
| [`VoicePlayback.close()`](../patrick/playback.py#L211) | Stops audio, signals shutdown and joins both workers with bounded waits. |

## `ports.py`

Hardware/provider contracts. Implementations must never block the audio loop.

| Function | Purpose |
|---|---|
| [`Provider.submit()`](../patrick/ports.py#L22) | Contract: start generation asynchronously for a turn. |
| [`Provider.cancel()`](../patrick/ports.py#L23) | Contract: invalidate/cancel a turn. |
| [`Provider.poll()`](../patrick/ports.py#L24) | Contract: return available Response events without blocking. |
| [`Playback.busy()`](../patrick/ports.py#L29) | Contract: whether output work remains. |
| [`Playback.enqueue()`](../patrick/ports.py#L30) | Contract: queue current-turn text/audio. |
| [`Playback.stop()`](../patrick/ports.py#L31) | Contract: stop hardware and discard buffered output. |
| [`Playback.beep()`](../patrick/ports.py#L32) | Contract: play an acknowledgement. |
| [`Robot.moving()`](../patrick/ports.py#L37) | Contract: current movement status. |
| [`Robot.stop_movement()`](../patrick/ports.py#L38) | Contract: immediately request motion stop. |
| [`Robot.cancel_actions()`](../patrick/ports.py#L39) | Contract: invalidate pending robot actions. |
| [`WakeDetector.detect()`](../patrick/ports.py#L43) | Contract: decide whether text contains activation. |
| [`AudioProcessor.capture()`](../patrick/ports.py#L47) | Contract: return processed capture PCM. |
| [`AudioProcessor.reference()`](../patrick/ports.py#L48) | Contract: provide playback reference when supported. |

## `providers.py`

Streaming Chat Completions transport shared by OpenAI, Gemini and local servers.

| Function | Purpose |
|---|---|
| [`ChatProvider.__init__()`](../patrick/providers.py#L25) | Validates endpoint/model/timeout, allocates thread-safe transport state, loads prompt and optional knowledge base. |
| [`ChatProvider.submit()`](../patrick/providers.py#L71) | Builds a prompt/question/RAG request with bounded session history and starts a bounded generation worker. |
| [`ChatProvider.submit_planned()`](../patrick/providers.py#L74) | Internal implementation; inspect the linked source in its module context. |
| [`ChatProvider._emit()`](../patrick/providers.py#L114) | Enqueues response events with cancellation-aware backpressure. |
| [`ChatProvider._run()`](../patrick/providers.py#L122) | Resolves Ollama options, performs streaming requests, parses text/tool calls, runs bounded tool rounds and emits completion/errors. |
| [`ChatProvider._search_notice()`](../patrick/providers.py#L354) | Internal implementation; inspect the linked source in its module context. |
| [`ChatProvider._register_stream()`](../patrick/providers.py#L359) | Stores the currently active HTTP stream under a lock for cancellation. |
| [`ChatProvider.reset_session()`](../patrick/providers.py#L365) | Internal implementation; inspect the linked source in its module context. |
| [`ChatProvider.cancel()`](../patrick/providers.py#L371) | Sets cancellation and schedules stream closing outside the control thread. |
| [`ChatProvider._close_stream()`](../patrick/providers.py#L388) | Closes a transport best-effort without propagating close exceptions. |
| [`ChatProvider.poll()`](../patrick/providers.py#L394) | Drains currently queued provider events without waiting. |
| [`make_provider()`](../patrick/providers.py#L403) | Builds offline, native local, compatible cloud, or legacy auto-fallback providers from arguments/environment. |

## `qwen_asr.py`

Qwen ASR in a persistent, dependency-isolated worker.

| Function | Purpose |
|---|---|
| [`QwenASRTranscriber.__init__()`](../patrick/qwen_asr.py#L17) | Internal implementation; inspect the linked source in its module context. |
| [`QwenASRTranscriber._read()`](../patrick/qwen_asr.py#L37) | Internal implementation; inspect the linked source in its module context. |
| [`QwenASRTranscriber._read.read_line()`](../patrick/qwen_asr.py#L39) | Internal implementation; inspect the linked source in its module context. |
| [`QwenASRTranscriber.transcribe()`](../patrick/qwen_asr.py#L57) | Internal implementation; inspect the linked source in its module context. |
| [`QwenASRTranscriber.close()`](../patrick/qwen_asr.py#L67) | Internal implementation; inspect the linked source in its module context. |
| [`asr_dtype()`](../patrick/qwen_asr.py#L80) | Select supported floating precision; quantization requires another backend. |
| [`asr_worker()`](../patrick/qwen_asr.py#L95) | Internal implementation; inspect the linked source in its module context. |

## `speech_text.py`

Bounded phrase chunks for local neural TTS; source links stay in console text.

| Function | Purpose |
|---|---|
| [`spoken_text()`](../patrick/speech_text.py#L5) | Remove presentation markup and expand common symbols for speech only. |
| [`speech_chunks()`](../patrick/speech_text.py#L29) | Splits cleaned speech at sentence/word boundaries into bounded phrases, preserving common title abbreviations. |

## `text_normalization.py`

Shared corrections applied before routing, retrieval and language generation.

| Function | Purpose |
|---|---|
| [`correct_terms()`](../patrick/text_normalization.py#L16) | Correct distinctive Revobots ASR variants without rewriting ordinary words. |

## `tools.py`

Allowlisted read-only tools. Search results are data, never executable actions.

| Function | Purpose |
|---|---|
| [`SearchTools.__init__()`](../patrick/tools.py#L32) | Validates credentials/timeout and allocates search cache and lock. |
| [`SearchTools.execute()`](../patrick/tools.py#L42) | Enforces tool name/query schema, cancellation, cache/response bounds and normalizes search success or error. |
| [`SearchTools._request()`](../patrick/tools.py#L92) | Builds a Tavily search POST with bounded result options. |
| [`SearchTools._result()`](../patrick/tools.py#L97) | Converts Tavily JSON to bounded, URL-validated snippets with retrieval provenance. |
| [`OpenAISearchTools.__init__()`](../patrick/tools.py#L111) | Uses the shared search executor with OpenAI credentials and a configured search model. |
| [`OpenAISearchTools._request()`](../patrick/tools.py#L117) | Builds a Responses API request requiring hosted web_search, cited summary and no response storage. |
| [`OpenAISearchTools._result()`](../patrick/tools.py#L126) | Requires completed search plus citations, extracts bounded summary/sources and logs source URLs. |
| [`tools_from_env()`](../patrick/tools.py#L150) | Validates search enablement/provider and constructs the appropriate optional tool adapter. |

## `transcription.py`

Transcribe captured utterances off the capture/control thread, then call the LLM.

| Function | Purpose |
|---|---|
| [`TranscriptionProvider.__init__()`](../patrick/transcription.py#L15) | Wraps a downstream provider with transcription, audio snapshot, wake phrase and one-worker slot. |
| [`TranscriptionProvider.submit()`](../patrick/transcription.py#L24) | Snapshots PCM and schedules final transcription; rejects concurrent transcription work. |
| [`TranscriptionProvider.submit.run()`](../patrick/transcription.py#L32) | Worker body: transcribes, corrects terms, strips wake, ignores empty speech, checks cancellation and submits valid text. |
| [`TranscriptionProvider.reset_session()`](../patrick/transcription.py#L61) | Internal implementation; inspect the linked source in its module context. |
| [`TranscriptionProvider.cancel()`](../patrick/transcription.py#L68) | Marks transcription cancelled and forwards cancellation downstream. |
| [`TranscriptionProvider.poll()`](../patrick/transcription.py#L75) | Combines downstream response events with transcription completion/error events. |

## `turn_planning.py`

Small local structured decision, independent of the cloud and public search.

| Function | Purpose |
|---|---|
| [`TurnPlan.english()`](../patrick/turn_planning.py#L19) | Internal implementation; inspect the linked source in its module context. |
| [`OllamaTurnPlanner.__init__()`](../patrick/turn_planning.py#L37) | Internal implementation; inspect the linked source in its module context. |
| [`OllamaTurnPlanner.plan()`](../patrick/turn_planning.py#L40) | Internal implementation; inspect the linked source in its module context. |

