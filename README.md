# 🔀 multi-llm-router

**Smart routing across LLM providers.**

Stop overpaying for simple tasks. Stop manually switching between APIs. Let the router pick the best provider for each request.

```
               ┌─────────────┐
               │  Your App    │
               └──────┬───────┘
                      │
               ┌──────▼───────┐
               │  llm-router  │  ← strategy: cost|speed|quality
               └──┬───┬───┬───┘
                  │   │   │
         ┌────────┘   │   └────────┐
         ▼            ▼            ▼
    ┌─────────┐ ┌──────────┐ ┌──────────┐
    │ OpenAI  │ │Anthropic │ │ DeepSeek │ ...
    └─────────┘ └──────────┘ └──────────┘
```

## Install

```bash
pip install multi-llm-router
```

Or download:
```bash
curl -O https://raw.githubusercontent.com/leiMizzou/multi-llm-router/main/llm_router.py
```

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Route a prompt (cheapest provider)
llm-router route "Translate hello to French" --strategy cost

# Route for quality
llm-router route "Explain quantum entanglement" --strategy quality

# Route for speed
llm-router route "Say hi" --strategy speed

# See available providers
llm-router providers

# Classify a task
llm-router classify "Write a Python function to sort a list"
```

## Proxy Mode

Drop-in replacement for OpenAI API:

```bash
# Start proxy
llm-router serve --port 8080 --strategy cost

# Use from any OpenAI client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Works with any OpenAI SDK:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
response = client.chat.completions.create(
    model="auto",  # router picks the best model
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Routing Strategies

| Strategy | Logic |
|----------|-------|
| **cost** | Cheapest model that meets quality threshold for the task type |
| **speed** | Lowest latency provider |
| **quality** | Highest quality model, with task-specific preferences |

## Task-Aware Routing

The router classifies prompts automatically:

| Task | Preferred Models | Min Quality |
|------|-----------------|-------------|
| Coding | GPT-4o, Claude Sonnet | 0.85 |
| Reasoning | Claude Opus, GPT-5 | 0.90 |
| Translation | GPT-4o-mini, DeepSeek | 0.75 |
| Creative | Claude Sonnet, GPT-4o | 0.85 |
| Summarize | GPT-4o-mini, Gemini Flash | 0.75 |
| Chat | Gemini Flash, Groq Llama | 0.70 |

## Supported Providers

OpenAI, Anthropic, DeepSeek, Google Gemini, Groq — any OpenAI-compatible API.

## Features

- **11 providers** pre-configured
- **Auto task classification** — coding, reasoning, translation, creative, summarize, chat
- **3 routing strategies** — cost, speed, quality
- **Fallback chains** — if primary fails, tries next best
- **Proxy server** — OpenAI-compatible drop-in replacement
- **Zero dependencies** — pure Python 3.8+

## License

MIT
