
import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

# Configuration and State Management
CONFIG_DIR = os.path.expanduser("~/.multi_llm_router")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
CACHE_DIR = os.path.join(CONFIG_DIR, "cache")
STATS_FILE = os.path.join(CONFIG_DIR, "stats.json")

class LLMRouterConfig:
    def __init__(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.config = self._load_config()
        self.stats = self._load_stats()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {
            "models": {
                "openai-gpt4o": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1", "model_name": "gpt-4o", "cost_input_per_token": 0.000005, "cost_output_per_token": 0.000015, "max_tokens": 4096},
                "anthropic-claude3-opus": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com/v1", "model_name": "claude-3-opus-20240229", "cost_input_per_token": 0.000015, "cost_output_per_token": 0.000075, "max_tokens": 4096},
                "openai-gpt35-turbo": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1", "model_name": "gpt-3.5-turbo", "cost_input_per_token": 0.0000005, "cost_output_per_token": 0.0000015, "max_tokens": 4096},
                "anthropic-claude3-haiku": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com/v1", "model_name": "claude-3-haiku-20240307", "cost_input_per_token": 0.00000025, "cost_output_per_token": 0.00000125, "max_tokens": 4096},
            },
            "routes": [
                {"pattern": ".*code generation.*", "models": ["openai-gpt4o", "anthropic-claude3-opus"], "strategy": "cost_optimized"},
                {"pattern": ".*summarization.*", "models": ["anthropic-claude3-haiku", "openai-gpt35-turbo"], "strategy": "cost_optimized"},
                {"pattern": ".*", "models": ["openai-gpt35-turbo", "anthropic-claude3-haiku"], "strategy": "cost_optimized"} # Default route
            ],
            "caching": {"enabled": True, "ttl_seconds": 3600}
        }

    def _save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def _load_stats(self) -> Dict[str, Any]:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        return {"total_cost": 0.0, "total_requests": 0, "model_stats": {}, "cache_hits": 0, "cache_misses": 0}

    def _save_stats(self):
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def get_models(self) -> Dict[str, Any]:
        return self.config["models"]

    def get_routes(self) -> List[Dict[str, Any]]:
        return self.config["routes"]

    def get_caching_config(self) -> Dict[str, Any]:
        return self.config["caching"]

    def update_model(self, model_name: str, updates: Dict[str, Any]):
        if model_name not in self.config["models"]:
            raise ValueError(f"Model '{model_name}' not found.")
        self.config["models"][model_name].update(updates)
        self._save_config()

    def add_model(self, model_name: str, model_config: Dict[str, Any]):
        if model_name in self.config["models"]:
            raise ValueError(f"Model '{model_name}' already exists.")
        self.config["models"][model_name] = model_config
        self._save_config()

    def delete_model(self, model_name: str):
        if model_name not in self.config["models"]:
            raise ValueError(f"Model '{model_name}' not found.")
        del self.config["models"][model_name]
        self._save_config()

    def add_route(self, route_config: Dict[str, Any]):
        self.config["routes"].append(route_config)
        self._save_config()

    def update_stats(self, model_name: str, input_tokens: int, output_tokens: int, cost: float, latency: float, success: bool):
        self.stats["total_cost"] += cost
        self.stats["total_requests"] += 1
        if model_name not in self.stats["model_stats"]:
            self.stats["model_stats"][model_name] = {"requests": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0, "latency": 0.0, "successes": 0, "failures": 0}
        
        model_stat = self.stats["model_stats"][model_name]
        model_stat["requests"] += 1
        model_stat["cost"] += cost
        model_stat["input_tokens"] += input_tokens
        model_stat["output_tokens"] += output_tokens
        model_stat["latency"] = (model_stat["latency"] * (model_stat["requests"] - 1) + latency) / model_stat["requests"] # Simple moving average
        if success:
            model_stat["successes"] += 1
        else:
            model_stat["failures"] += 1
        self._save_stats()

    def record_cache_hit(self):
        self.stats["cache_hits"] += 1
        self._save_stats()

    def record_cache_miss(self):
        self.stats["cache_misses"] += 1
        self._save_stats()

# LLM API Clients
class LLMClient:
    def __init__(self, model_config: Dict[str, Any]):
        self.provider = model_config["provider"]
        self.api_key = os.environ.get(model_config["api_key_env"])
        if not self.api_key:
            raise ValueError(f"API key for {self.provider} not found in environment variable {model_config['api_key_env']}")
        self.base_url = model_config["base_url"]
        self.model_name = model_config["model_name"]
        self.cost_input_per_token = model_config["cost_input_per_token"]
        self.cost_output_per_token = model_config["cost_output_per_token"]
        self.max_tokens = model_config.get("max_tokens", 4096) # Default to 4096 if not specified

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.cost_input_per_token) + (output_tokens * self.cost_output_per_token)

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        start_time = time.time()
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            completion = data["choices"][0]["message"]["content"]
            input_tokens = data["usage"]["prompt_tokens"]
            output_tokens = data["usage"]["completion_tokens"]
            latency = time.time() - start_time
            cost = self._calculate_cost(input_tokens, output_tokens)
            return completion, input_tokens, output_tokens, cost, latency
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI API error: {e}")
        except KeyError as e:
            raise RuntimeError(f"OpenAI API response parse error: {e}, Response: {data}")

class AnthropicClient(LLMClient):
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        import requests
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        start_time = time.time()
        try:
            response = requests.post(f"{self.base_url}/messages", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            completion = data["content"][0]["text"]
            input_tokens = data["usage"]["input_tokens"]
            output_tokens = data["usage"]["output_tokens"]
            latency = time.time() - start_time
            cost = self._calculate_cost(input_tokens, output_tokens)
            return completion, input_tokens, output_tokens, cost, latency
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Anthropic API error: {e}")
        except KeyError as e:
            raise RuntimeError(f"Anthropic API response parse error: {e}, Response: {data}")

# Routing Logic
class LLMRouter:
    def __init__(self, config_manager: LLMRouterConfig):
        self.config_manager = config_manager
        self.clients: Dict[str, LLMClient] = self._initialize_clients()

    def _initialize_clients(self) -> Dict[str, LLMClient]:
        clients = {}
        for model_name, model_config in self.config_manager.get_models().items():
            try:
                if model_config["provider"] == "openai":
                    clients[model_name] = OpenAIClient(model_config)
                elif model_config["provider"] == "anthropic":
                    clients[model_name] = AnthropicClient(model_config)
                # Add more providers here
                else:
                    print(f"Warning: Unknown provider '{model_config['provider']}' for model '{model_name}'. Skipping.")
            except ValueError as e:
                print(f"Error initializing client for model '{model_name}': {e}")
            except ImportError:
                print(f"Missing dependencies for {model_config['provider']}. Please install 'requests'.")
        return clients

    def _get_matching_models(self, prompt: str) -> List[Tuple[str, LLMClient]]:
        import re
        
        matching_models = []
        routes = self.config_manager.get_routes()

        for route in routes:
            if re.search(route["pattern"], prompt, re.IGNORECASE):
                for model_name in route["models"]:
                    if model_name in self.clients:
                        matching_models.append((model_name, self.clients[model_name]))
                return matching_models # Return models from the first matching route

        return [] # Should not happen with a default route ".*"

    def _select_model(self, prompt: str, strategy: str, available_models: List[Tuple[str, LLMClient]]) -> Optional[Tuple[str, LLMClient]]:
        if not available_models:
            return None

        if strategy == "cost_optimized":
            # Sort by total potential cost (using a placeholder token count for comparison)
            # This is a heuristic; actual cost depends on actual input/output tokens
            # For a more accurate cost, we'd need to estimate token counts
            return min(available_models, key=lambda x: x[1].cost_input_per_token + x[1].cost_output_per_token)
        elif strategy == "latency_optimized":
            # This would require actual latency stats per model, which we don't have dynamically here
            # For now, we'll just pick the first available if latency is the strategy.
            # In a real system, you'd query historical stats or ping models.
            print("Warning: Latency-optimized strategy not fully implemented. Falling back to first available.")
            return available_models[0]
        else:
            # Default to first available if strategy is unknown
            print(f"Warning: Unknown strategy '{strategy}'. Falling back to first available.")
            return available_models[0]

    def _get_route_strategy(self, prompt: str) -> str:
        import re
        routes = self.config_manager.get_routes()
        for route in routes:
            if re.search(route["pattern"], prompt, re.IGNORECASE):
                return route["strategy"]
        return "cost_optimized" # Default strategy if no route matches

    def route(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Tuple[str, str, int, int, float, float]:
        caching_config = self.config_manager.get_caching_config()
        if caching_config["enabled"]:
            cached_response = self._get_cached_response(prompt)
            if cached_response:
                self.config_manager.record_cache_hit()
                return cached_response[0], "cache", cached_response[1], cached_response[2], 0.0, 0.0 # completion, model_name, input_tokens, output_tokens, cost, latency
            self.config_manager.record_cache_miss()

        matching_models_with_clients = self._get_matching_models(prompt)
        if not matching_models_with_clients:
            raise RuntimeError("No models configured or matched for the given prompt.")

        strategy = self._get_route_strategy(prompt)
        
        # Apply strategy to get an ordered list of models for fallback
        if strategy == "cost_optimized":
            # Sort all matched models by their per-token cost for fallback
            ordered_models = sorted(matching_models_with_clients, key=lambda x: x[1].cost_input_per_token + x[1].cost_output_per_token)
        else:
            # For other strategies or if strategy is not specifically cost/latency based,
            # use the order defined in the route.
            # Reconstruct based on route order for consistent fallback.
            ordered_models = []
            route_models_names = []
            for route in self.config_manager.get_routes():
                if import re; re.search(route["pattern"], prompt, re.IGNORECASE):
                    route_models_names = route["models"]
                    break
            
            for model_name in route_models_names:
                if model_name in self.clients:
                    ordered_models.append((model_name, self.clients[model_name]))
            
            # Ensure all matching clients are included, even if not explicitly in the route's ordered list
            # This handles cases where _get_matching_models might return more than what's strictly in a route's 'models' list
            # (though the current implementation of _get_matching_models should align them).
            # For robustness, we could merge and deduplicate.
            
            # For now, let's ensure models in ordered_models are unique and use the strategy's preference for primary
            # and then fall back based on its relative cost or just sequential order.
            # Simplified: the selection logic will pick the *best* from `ordered_models` based on strategy,
            # and then fallback will iterate through the rest of `ordered_models`.


        last_error = None
        for model_name, client in ordered_models:
            print(f"Attempting to use model: {model_name}...")
            try:
                completion, input_tokens, output_tokens, cost, latency = client.generate(prompt, max_tokens, temperature)
                self.config_manager.update_stats(model_name, input_tokens, output_tokens, cost, latency, success=True)
                if caching_config["enabled"]:
                    self._cache_response(prompt, completion, input_tokens, output_tokens)
                return completion, model_name, input_tokens, output_tokens, cost, latency
            except Exception as e:
                last_error = e
                self.config_manager.update_stats(model_name, 0, 0, 0.0, 0.0, success=False)
                print(f"Model '{model_name}' failed: {e}. Attempting fallback...")
        
        raise RuntimeError(f"All models failed for the prompt. Last error: {last_error}")

    def _get_cache_file_path(self, prompt: str) -> str:
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return os.path.join(CACHE_DIR, f"{prompt_hash}.json")

    def _get_cached_response(self, prompt: str) -> Optional[Tuple[str, int, int]]:
        cache_file = self._get_cache_file_path(prompt)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            ttl = self.config_manager.get_caching_config()["ttl_seconds"]
            if (time.time() - data["timestamp"]) < ttl:
                return data["completion"], data["input_tokens"], data["output_tokens"]
            else:
                os.remove(cache_file) # Cache expired
        return None

    def _cache_response(self, prompt: str, completion: str, input_tokens: int, output_tokens: int):
        cache_file = self._get_cache_file_path(prompt)
        data = {
            "prompt": prompt,
            "completion": completion,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": time.time()
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=4)

    def clear_cache(self):
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting cache file {file_path}: {e}")
        print("Cache cleared.")

# CLI Interface
def print_help():
    print("""
Usage: python llm_router.py <command> [options]

Commands:
  route <prompt>                - Route a prompt to an LLM based on configuration.
  config list [models|routes]   - List current models or routing rules.
  config add model <name> <json> - Add a new model configuration.
  config update model <name> <json> - Update an existing model configuration.
  config delete model <name>    - Delete a model configuration.
  config add route <json>       - Add a new routing rule.
  stats                         - Display usage statistics.
  cache clear                   - Clear the response cache.
  cache list                    - List cached prompts (shows hashes).
  help                          - Display this help message.

Examples:
  python llm_router.py route "Generate python code for a quicksort algorithm."
  python llm_router.py route "Summarize this article: ..." --max_tokens 100
  python llm_router.py config list models
  python llm_router.py config add model my-ollama '{"provider": "openai", "api_key_env": "OLLAMA_API_KEY", "base_url": "http://localhost:11434/v1", "model_name": "llama2", "cost_input_per_token": 0.0, "cost_output_per_token": 0.0, "max_tokens": 4096}'
  python llm_router.py stats
""")

def main():
    import sys
    config_manager = LLMRouterConfig()
    router = LLMRouter(config_manager)

    args = sys.argv[1:]

    if not args or args[0] == "help":
        print_help()
        return

    command = args[0]

    if command == "route":
        if len(args) < 2:
            print("Error: 'route' command requires a prompt.")
            print_help()
            return
        
        prompt = args[1]
        max_tokens = 200
        temperature = 0.7

        # Parse optional arguments
        i = 2
        while i < len(args):
            if args[i] == "--max_tokens":
                if i + 1 < len(args):
                    max_tokens = int(args[i+1])
                    i += 1
                else:
                    print("Error: --max_tokens requires a value.")
                    return
            elif args[i] == "--temperature":
                if i + 1 < len(args):
                    temperature = float(args[i+1])
                    i += 1
                else:
                    print("Error: --temperature requires a value.")
                    return
            i += 1

        try:
            completion, model_name, input_tokens, output_tokens, cost, latency = router.route(prompt, max_tokens, temperature)
            print("\n--- LLM Router Response ---")
            print(f"Model Used: {model_name}")
            print(f"Input Tokens: {input_tokens}")
            print(f"Output Tokens: {output_tokens}")
            print(f"Cost: ${cost:.8f}")
            print(f"Latency: {latency:.2f}s")
            print("\nCompletion:")
            print(completion)
        except RuntimeError as e:
            print(f"Error routing prompt: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    elif command == "config":
        if len(args) < 2:
            print("Error: 'config' command requires a subcommand.")
            print_help()
            return
        
        subcommand = args[1]

        if subcommand == "list":
            if len(args) < 3:
                print("Error: 'config list' requires 'models' or 'routes'.")
                print_help()
                return
            target = args[2]
            if target == "models":
                print(json.dumps(config_manager.get_models(), indent=4))
            elif target == "routes":
                print(json.dumps(config_manager.get_routes(), indent=4))
            else:
                print(f"Error: Unknown config list target '{target}'. Use 'models' or 'routes'.")
                print_help()
        
        elif subcommand == "add":
            if len(args) < 4:
                print("Error: 'config add' requires 'model' or 'route' and data.")
                print_help()
                return
            target = args[2]
            try:
                data = json.loads(args[3])
                if target == "model":
                    name = data.pop("name") # Expect name in the JSON for 'add model'
                    config_manager.add_model(name, data)
                    print(f"Model '{name}' added successfully.")
                elif target == "route":
                    config_manager.add_route(data)
                    print("Route added successfully.")
                else:
                    print(f"Error: Unknown config add target '{target}'. Use 'model' or 'route'.")
            except json.JSONDecodeError:
                print("Error: Invalid JSON provided.")
            except ValueError as e:
                print(f"Error: {e}")

        elif subcommand == "update":
            if len(args) < 5 or args[2] != "model":
                print("Error: 'config update model <name> <json>'")
                print_help()
                return
            model_name = args[3]
            try:
                updates = json.loads(args[4])
                config_manager.update_model(model_name, updates)
                print(f"Model '{model_name}' updated successfully.")
            except json.JSONDecodeError:
                print("Error: Invalid JSON provided.")
            except ValueError as e:
                print(f"Error: {e}")

        elif subcommand == "delete":
            if len(args) < 4 or args[2] != "model":
                print("Error: 'config delete model <name>'")
                print_help()
                return
            model_name = args[3]
            try:
                config_manager.delete_model(model_name)
                print(f"Model '{model_name}' deleted successfully.")
            except ValueError as e:
                print(f"Error: {e}")

        else:
            print(f"Error: Unknown config subcommand '{subcommand}'.")
            print_help()

    elif command == "stats":
        stats = config_manager.stats
        print("\n--- LLM Usage Statistics ---")
        print(f"Total Requests: {stats.get('total_requests', 0)}")
        print(f"Total Cost: ${stats.get('total_cost', 0.0):.8f}")
        print(f"Cache Hits: {stats.get('cache_hits', 0)}")
        print(f"Cache Misses: {stats.get('cache_misses', 0)}")
        print("\nModel Specific Stats:")
        if not stats.get('model_stats'):
            print("  No model usage data yet.")
        for model_name, model_stats in stats.get('model_stats', {}).items():
            print(f"  Model: {model_name}")
            print(f"    Requests: {model_stats.get('requests', 0)}")
            print(f"    Successes: {model_stats.get('successes', 0)}")
            print(f"    Failures: {model_stats.get('failures', 0)}")
            print(f"    Cost: ${model_stats.get('cost', 0.0):.8f}")
            print(f"    Input Tokens: {model_stats.get('input_tokens', 0)}")
            print(f"    Output Tokens: {model_stats.get('output_tokens', 0)}")
            print(f"    Avg Latency: {model_stats.get('latency', 0.0):.2f}s")
            print("-" * 30)

    elif command == "cache":
        if len(args) < 2:
            print("Error: 'cache' command requires a subcommand.")
            print_help()
            return
        
        subcommand = args[1]
        if subcommand == "clear":
            router.clear_cache()
        elif subcommand == "list":
            print("\n--- Cached Prompts ---")
            if not os.listdir(CACHE_DIR):
                print("  Cache is empty.")
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith(".json"):
                    file_path = os.path.join(CACHE_DIR, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        timestamp_dt = datetime.fromtimestamp(data["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"  Hash: {filename.replace('.json', '')}")
                        print(f"    Prompt: {data['prompt'][:70]}...")
                        print(f"    Cached at: {timestamp_dt}")
                        print("-" * 30)
                    except Exception as e:
                        print(f"  Error reading cache file {filename}: {e}")
        else:
            print(f"Error: Unknown cache subcommand '{subcommand}'.")
            print_help()

    else:
        print(f"Error: Unknown command '{command}'.")
        print_help()

if __name__ == "__main__":
    main()
