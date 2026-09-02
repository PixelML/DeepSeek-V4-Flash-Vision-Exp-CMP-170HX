# API integration

The server exposes an OpenAI-compatible API. Keep it on a private network, or
publish it through an authenticated TLS reverse proxy. This page assumes the
vision-path server from `scripts/launch-vision-server.sh`; the text-path
server (`scripts/launch-text-server.sh`) accepts the same text requests but
rejects image content.

## List models

```bash
export DSV4_BASE_URL="http://127.0.0.1:18099/v1"

curl -fsS "${DSV4_BASE_URL}/models"
```

## Text request

```bash
curl -fsS "${DSV4_BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [{"role": "user", "content": "Reply with exactly one word: the color of a clear daytime sky."}],
    "temperature": 0,
    "max_tokens": 200
  }'
```

## Image request

The vision-path server accepts an image as a `data:` URL alongside text in
the same message. This is the request shape used by the measured golden
corpus (a 64x64 gradient PNG):

```bash
curl -fsS "${DSV4_BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Name the two colors this gradient blends between, left color first."},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}}
        ]
      }
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

Count prompt and completion tokens from the final `usage` object in the
response, not from an intermediate streaming event. Image tokens appear
inside `usage.prompt_tokens_details` for a multimodal request, separate from
plain text prompt tokens.

## Tool calls

The launch recipe enables auto tool choice with the `deepseek_v4` tool-call
parser (`--enable-auto-tool-choice --tool-call-parser deepseek_v4`). Pass
`tools` and, optionally, `tool_choice` the same way as any OpenAI-compatible
tool-calling client:

```bash
curl -fsS "${DSV4_BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [{"role": "user", "content": "What is the weather in Boston?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 200
  }'
```

A tool call in the response arrives on `choices[0].message.tool_calls`, each
entry with a `function.name` and a JSON-encoded `function.arguments` string,
same as the standard OpenAI tool-calling contract.

## Notes

- Context is 262,144 tokens at the current launch pin. The prefill ladder
  passes cleanly through 65,000 prompt tokens; the 131,000-token rung
  crashes the engine — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and the
  README "Limitations" section before sending a very long prompt.
- `--limit-mm-per-prompt '{"image": 2}'` caps images at 2 per request. The
  measured golden corpus uses one image per request.
- There is no video encoder in the checkpoint. Sending a GIF is read as a
  single still frame.
- Concurrency above c=2 on the vision-path server is outside the measured
  stable range — see the README "Limitations" section for the c=4 crash
  class before sending concurrent requests in production.
