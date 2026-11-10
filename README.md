# FilmOpen helper 1.0.0

MIT helper for Wan2GP 18.21 (`9d25c601`), installed as `plugins/wan2gp-filmopen`.
No requirements, key, MCP server or configuration changes. Enable it in Wan2GP's
Plugins tab and restart. FilmOpen's Set up action uses Wan2GP's own installer.

Six named Gradio events take one JSON string and return one JSON string:
`filmopen_info`, `filmopen_models`, `filmopen_download`, `filmopen_generate`,
`filmopen_job`, `filmopen_cancel`. Models/download take `{tier, models}` where
models name `modelType`, `transformerQuantization`, `textEncoderQuantization`.
The helper holds no stack catalogue. Downloads include the separate encoder.
Generate takes `{clientId, settings}`; overrides in settings are Wan2GP text.
Jobs are in memory: after restart their ids are unknown. Cancel requests; poll
job to learn the actual end. Only one download worker runs at a time.
Generate answers `{busy: "text", message}` while a completion runs: one GPU job
at a time, text or render (`admission: "one-gpu-job"` in `filmopen_info`).

Requests, answers, file decisions, progress and errors print to Wan2GP's console.
Do not put credentials in these requests. Offline tests use stubbed Wan2GP APIs.

**Text endpoint.** The helper also serves `POST /filmopen/v1/chat/completions`
on WanGP's own server, through WanGP's plug-in web-app API (no WanGP file is
changed): `{model, messages, max_tokens, tools?, tool_choice?}` in, the local
engine the request names answering `{text, tool_calls,
usage: {inputTokens, outputTokens}, finish}`. Tools go through Qwen's own chat
template; a reply whose call does not parse is returned as words. WanGP's engine, the values to put back and the switch are read and decided
under WanGP's config lock. A reply stopped at its maximum is refused `cutOff`
before any tool call is decoded where tools were offered, where it holds tool
markup or where no words came. Where WanGP is set to another local LLM (its `enhancer_enabled`),
the helper sets it to the one asked for, with the quantization WanGP's
Configuration would give it, only while WanGP's loader loads it and under
WanGP's own config lock (a save WanGP makes meanwhile waits and writes WanGP's
own values), puts WanGP's own back before answering, and asks WanGP to load its
own again at its next use. Refusals are
`409 busy` (the helper's render submitted and not done, WanGP rendering or its
queue holding a task, or another answer runs; `gpu.busy` logs the holder and its
start), `409 unknown-engine`, `422
cutOff` and `500 failed` (an engine that cannot load is named), each
`{error, message}`. The
console logs the request, the messages as rewritten, the tools, the raw engine
output, every decision, the load time and a failure's stage and trace, and
`complete.measure {model, inputTokens, maxTokens, peakVram, peakReserved, ram,
ramUsed, ms, outcome}` once an admitted call ends (the peak counted from its
admission). A render started from WanGP's own window during an answer is
outside the helper.

`filmopen_info` also reports `gpus` (every GPU's name, VRAM and free VRAM),
`torch`, `cuda`, `ramBytes` and `llm` (`engine`, `entry`, `quantization`, `loaded`).
