#!/usr/bin/env python3
"""
🔀 multi-llm-router — Smart routing across LLM providers.

Route LLM requests to the cheapest, fastest, or best provider automatically.
Includes an OpenAI-compatible proxy server for drop-in replacement.

Usage:
    python llm_router.py route "Explain quantum computing" --strategy cost
    python llm_router.py serve --port 8080
    python llm_router.py providers
    python llm_router.py config

Zero dependencies (stdlib only). Python 3.8+.
"""

import argparse
import http.server
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

__version__ = "0.1.0"

# ─── Provider Definitions ───────────────────────────────────────────────────

@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str        # Environment variable for API key
    models: List[str]
    input_cost: float       # per 1M tokens
    output_cost: float      # per 1M tokens
    avg_latency_ms: int     # typical latency
    quality_score: float    # 0-1 quality rating
    api_type: str = "openai"  # openai | anthropic
    headers_fn: Optional[str] = None

DEFAULT_PROVIDERS = [
    Provider("openai-gpt4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY",
             ["gpt-4o-mini"], 0.15, 0.60, 300, 0.82),
    Provider("openai-gpt4o", "https://api.openai.com/v1", "OPENAI_API_KEY",
             ["gpt-4o"], 2.50, 10.00, 400, 0.92),
    Provider("openai-gpt5", "https://api.openai.com/v1", "OPENAI_API_KEY",
             ["gpt-5"], 10.00, 30.00, 600, 0.97),
    Provider("anthropic-haiku", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
             ["claude-haiku-4"], 0.25, 1.25, 250, 0.80, "anthropic"),
    Provider("anthropic-sonnet", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
             ["claude-sonnet-4"], 3.00, 15.00, 500, 0.93, "anthropic"),
    Provider("anthropic-opus", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
             ["claude-opus-4"], 15.00, 75.00, 800, 0.98, "anthropic"),
    Provider("deepseek-v3", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY",
             ["deepseek-chat"], 0.27, 1.10, 350, 0.85),
    Provider("deepseek-r1", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY",
             ["deepseek-reasoner"], 0.55, 2.19, 500, 0.90),
    Provider("gemini-flash", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
             ["gemini-2.0-flash"], 0.10, 0.40, 200, 0.78),
    Provider("gemini-pro", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
             ["gemini-2.5-pro"], 1.25, 10.00, 450, 0.91),
    Provider("groq-llama", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
             ["llama-3.3-70b-versatile"], 0.59, 0.79, 100, 0.80),
]

# ─── Task Classification ────────────────────────────────────────────────────

TASK_PATTERNS = {
    "coding": [r'\b(code|function|class|def |import |bug|error|debug|refactor|test|api|endpoint)\b'],
    "reasoning": [r'\b(explain|why|analyze|compare|evaluate|think|reason|logic|proof|math)\b'],
    "translation": [r'\b(translat|翻译|übersetze|tradui|переведи)\b'],
    "creative": [r'\b(write|story|poem|creative|imagine|fiction|blog|essay)\b'],
    "summarize": [r'\b(summar|tldr|brief|condense|key points|overview)\b'],
    "chat": [r'\b(hello|hi|hey|thanks|how are)\b'],
}

def classify_task(prompt: str) -> str:
    """Classify a prompt into a task type."""
    prompt_lower = prompt.lower()
    scores = {}
    for task, patterns in TASK_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, prompt_lower, re.IGNORECASE))
        if score > 0:
            scores[task] = score
    return max(scores, key=scores.get) if scores else "general"

# ─── Routing Strategies ─────────────────────────────────────────────────────

TASK_PREFERENCES = {
    "coding": {"min_quality": 0.85, "prefer": ["openai-gpt4o", "anthropic-sonnet"]},
    "reasoning": {"min_quality": 0.90, "prefer": ["anthropic-opus", "openai-gpt5"]},
    "translation": {"min_quality": 0.75, "prefer": ["openai-gpt4o-mini", "deepseek-v3"]},
    "creative": {"min_quality": 0.85, "prefer": ["anthropic-sonnet", "openai-gpt4o"]},
    "summarize": {"min_quality": 0.75, "prefer": ["openai-gpt4o-mini", "gemini-flash"]},
    "chat": {"min_quality": 0.70, "prefer": ["gemini-flash", "groq-llama"]},
    "general": {"min_quality": 0.80, "prefer": ["openai-gpt4o-mini", "deepseek-v3"]},
}

def get_available_providers() -> List[Provider]:
    """Return providers that have API keys configured."""
    available = []
    for p in DEFAULT_PROVIDERS:
        key = os.environ.get(p.api_key_env, "")
        if key:
            available.append(p)
    return available

def route_by_cost(providers: List[Provider], task: str = "") -> List[Provider]:
    """Route to cheapest provider that meets quality threshold."""
    prefs = TASK_PREFERENCES.get(task, TASK_PREFERENCES["general"])
    min_q = prefs["min_quality"]
    eligible = [p for p in providers if p.quality_score >= min_q]
    if not eligible:
        eligible = providers
    return sorted(eligible, key=lambda p: p.output_cost)

def route_by_speed(providers: List[Provider], task: str = "") -> List[Provider]:
    """Route to fastest provider."""
    return sorted(providers, key=lambda p: p.avg_latency_ms)

def route_by_quality(providers: List[Provider], task: str = "") -> List[Provider]:
    """Route to highest quality provider."""
    prefs = TASK_PREFERENCES.get(task, TASK_PREFERENCES["general"])
    preferred = prefs.get("prefer", [])
    
    def sort_key(p):
        priority = preferred.index(p.name) if p.name in preferred else 999
        return (priority, -p.quality_score)
    
    return sorted(providers, key=sort_key)

STRATEGIES = {
    "cost": route_by_cost,
    "speed": route_by_speed,
    "quality": route_by_quality,
}

# ─── API Callers ─────────────────────────────────────────────────────────────

def call_openai(provider: Provider, messages: List[Dict], max_tokens: int = 1000) -> Dict:
    """Call OpenAI-compatible API."""
    api_key = os.environ.get(provider.api_key_env, "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    body = json.dumps({
        "model": provider.models[0],
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()
    
    req = urllib.request.Request(
        f"{provider.base_url}/chat/completions",
        data=body, headers=headers,
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        latency = (time.time() - start) * 1000
        data["_router_meta"] = {
            "provider": provider.name,
            "latency_ms": round(latency, 1),
            "strategy": "routed",
        }
        return data
    except Exception as e:
        return {"error": str(e), "provider": provider.name}

def call_anthropic(provider: Provider, messages: List[Dict], max_tokens: int = 1000) -> Dict:
    """Call Anthropic API."""
    api_key = os.environ.get(provider.api_key_env, "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    
    body = json.dumps({
        "model": provider.models[0],
        "max_tokens": max_tokens,
        "messages": messages,
    }).encode()
    
    req = urllib.request.Request(
        f"{provider.base_url}/messages",
        data=body, headers=headers,
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        latency = (time.time() - start) * 1000
        
        # Convert to OpenAI format
        content = result.get("content", [{}])[0].get("text", "")
        usage = result.get("usage", {})
        return {
            "id": result.get("id", ""),
            "object": "chat.completion",
            "model": provider.models[0],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            "_router_meta": {"provider": provider.name, "latency_ms": round(latency, 1)},
        }
    except Exception as e:
        return {"error": str(e), "provider": provider.name}

def call_provider(provider: Provider, messages: List[Dict], max_tokens: int = 1000) -> Dict:
    """Call any provider."""
    if provider.api_type == "anthropic":
        return call_anthropic(provider, messages, max_tokens)
    return call_openai(provider, messages, max_tokens)

def route_and_call(prompt: str, strategy: str = "cost", max_tokens: int = 1000) -> Dict:
    """Route a prompt and call the best provider with fallback."""
    available = get_available_providers()
    if not available:
        return {"error": "No providers configured. Set API key environment variables."}
    
    task = classify_task(prompt)
    route_fn = STRATEGIES.get(strategy, route_by_cost)
    ranked = route_fn(available, task)
    
    messages = [{"role": "user", "content": prompt}]
    
    for provider in ranked[:3]:  # Try top 3
        result = call_provider(provider, messages, max_tokens)
        if "error" not in result:
            result.setdefault("_router_meta", {})["task_type"] = task
            result["_router_meta"]["strategy"] = strategy
            return result
    
    return {"error": f"All providers failed. Last: {ranked[0].name if ranked else 'none'}"}

# ─── Proxy Server ────────────────────────────────────────────────────────────

class RouterProxy(BaseHTTPRequestHandler):
    """OpenAI-compatible proxy that routes to the best provider."""
    
    strategy = "cost"
    
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length).decode())
            
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 1000)
            
            # Extract prompt for classification
            prompt = messages[-1].get("content", "") if messages else ""
            
            available = get_available_providers()
            task = classify_task(prompt)
            route_fn = STRATEGIES.get(self.strategy, route_by_cost)
            ranked = route_fn(available, task)
            
            for provider in ranked[:3]:
                result = call_provider(provider, messages, max_tokens)
                if "error" not in result:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                    return
            
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "All providers failed"}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == "/v1/models":
            available = get_available_providers()
            models = []
            for p in available:
                for m in p.models:
                    models.append({"id": m, "object": "model", "owned_by": p.name})
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": models}).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "version": __version__}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        sys.stderr.write(f"[router] {args[0]}\n")

# ─── Output ──────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

def format_providers(providers: List[Provider], available: List[str]) -> str:
    lines = []
    lines.append(f"\n{BOLD}🔀 multi-llm-router v{__version__} — Providers{RESET}")
    lines.append(f"{DIM}{'─' * 75}{RESET}")
    lines.append(f"  {'Provider':<22} {'Model':<25} {'Out/1M':>8} {'Latency':>8} {'Quality':>8} {'Status':>8}")
    lines.append(f"  {'─' * 72}")
    
    for p in DEFAULT_PROVIDERS:
        status = f"{GREEN}✓{RESET}" if p.name in available else f"{RED}✗{RESET}"
        lines.append(f"  {p.name:<22} {p.models[0]:<25} ${p.output_cost:<7.2f} {p.avg_latency_ms:>6}ms {p.quality_score:>7.2f} {status:>8}")
    
    lines.append(f"\n{DIM}  Set API keys via environment variables to enable providers.{RESET}\n")
    return '\n'.join(lines)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="llm-router", description="🔀 Smart routing across LLM providers")
    sub = parser.add_subparsers(dest="command")
    
    # Route
    route_p = sub.add_parser("route", help="Route a prompt to the best provider")
    route_p.add_argument("prompt", help="The prompt text")
    route_p.add_argument("--strategy", "-s", choices=["cost", "speed", "quality"], default="cost")
    route_p.add_argument("--max-tokens", "-t", type=int, default=1000)
    route_p.add_argument("--json", action="store_true")
    
    # Serve
    serve_p = sub.add_parser("serve", help="Start OpenAI-compatible proxy server")
    serve_p.add_argument("--port", "-p", type=int, default=8080)
    serve_p.add_argument("--strategy", "-s", choices=["cost", "speed", "quality"], default="cost")
    
    # Providers
    sub.add_parser("providers", help="List configured providers")
    
    # Classify
    cls_p = sub.add_parser("classify", help="Classify a prompt's task type")
    cls_p.add_argument("prompt")
    
    parser.add_argument("--version", "-v", action="version", version=f"llm-router {__version__}")
    
    args = parser.parse_args()
    
    if args.command == "providers":
        available = [p.name for p in get_available_providers()]
        print(format_providers(DEFAULT_PROVIDERS, available))
    
    elif args.command == "classify":
        task = classify_task(args.prompt)
        prefs = TASK_PREFERENCES.get(task, {})
        print(f"Task: {task}")
        print(f"Min quality: {prefs.get('min_quality', 0.80)}")
        print(f"Preferred: {', '.join(prefs.get('prefer', []))}")
    
    elif args.command == "route":
        result = route_and_call(args.prompt, args.strategy, args.max_tokens)
        if args.json:
            print(json.dumps(result, indent=2))
        elif "error" in result:
            print(f"{RED}Error: {result['error']}{RESET}", file=sys.stderr)
            sys.exit(1)
        else:
            meta = result.get("_router_meta", {})
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"{DIM}[{meta.get('provider', '?')} | {meta.get('task_type', '?')} | {meta.get('latency_ms', 0):.0f}ms]{RESET}")
            print(content)
    
    elif args.command == "serve":
        RouterProxy.strategy = args.strategy
        server = HTTPServer(("0.0.0.0", args.port), RouterProxy)
        print(f"{BOLD}🔀 llm-router proxy{RESET}")
        print(f"  Listening on http://0.0.0.0:{args.port}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Endpoints: /v1/chat/completions, /v1/models, /health")
        print(f"  {DIM}Use as OpenAI base_url: http://localhost:{args.port}/v1{RESET}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
