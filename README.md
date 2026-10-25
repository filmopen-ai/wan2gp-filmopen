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

Requests, answers, file decisions, progress and errors print to Wan2GP's console.
Do not put credentials in these requests. Offline tests use stubbed Wan2GP APIs.
