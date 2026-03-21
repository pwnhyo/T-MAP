"""
LangChain-based MCP client wrapper for T-MAP experiments.

Replaces core/mcp_client.py with LangChain's MultiServerMCPClient + create_agent.
Provides a clean interface compatible with ExperimentBase.
"""
import json
import os
import logging
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

# Suppress noisy logs
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)

MAX_HISTORY_DETAIL_CHARS = 2000

def _truncate_detail_text(text: str, max_chars: int = MAX_HISTORY_DETAIL_CHARS) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... (Content truncated due to length) ..."


def load_server_config(config_path: str) -> dict:
    """Load server configurations from a JSON file.
    
    Supports ${env.VAR_NAME} syntax for environment variable references.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Resolve environment variable references in env fields
    for server_name, server_config in config.items():
        if "env" in server_config:
            resolved_env = {}
            for k, v in server_config["env"].items():
                if isinstance(v, str) and v.startswith("${env.") and v.endswith("}"):
                    env_var = v[6:-1]  # Extract VAR_NAME from ${env.VAR_NAME}
                    resolved_env[k] = os.environ.get(env_var, "")
                else:
                    resolved_env[k] = v
            server_config["env"] = resolved_env
    
    return config


def get_server_connections(config: dict, server_names: list[str]) -> dict:
    """Extract connection configs for the specified servers.
    
    Args:
        config: Full server config dict from servers.json
        server_names: List of server names to connect to (e.g., ["Gmail", "GoogleSearch"])
    
    Returns:
        Dict of {server_name: connection_config} for MultiServerMCPClient
    """
    connections = {}
    for name in server_names:
        if name not in config:
            raise ValueError(f"Server '{name}' not found in config. Available: {list(config.keys())}")
        connections[name] = config[name]
    return connections


def convert_messages_to_history(messages) -> list[dict]:
    """Convert LangChain message objects to our clean dict format.
    
    Strips all LangChain metadata, keeping only what T-MAP needs:
    - role, content, tool_calls (for AI), tool_name (for tool results)
    """
    history = []
    for m in messages:
        msg_type = getattr(m, 'type', 'unknown')
        content = getattr(m, 'content', '')
        
        # Normalize content: if it's a list of content blocks, extract text
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts) if text_parts else str(content)
        
        # Truncate verbose inner details before history is forwarded to attacker/judge prompts.
        if msg_type in ("tool", "ai", "assistant"):
            content = _truncate_detail_text(content)
        
        # Map LangChain types to OpenAI-style roles
        role_map = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}
        role = role_map.get(msg_type, msg_type)
        
        d = {"role": role, "content": content}
        
        # Extract tool_calls from AI messages
        if hasattr(m, "tool_calls") and m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _truncate_detail_text(
                            json.dumps(tc.get("args", {}), ensure_ascii=False)
                        ),
                    },
                }
                for tc in m.tool_calls
            ]
        
        # Extract tool name and call id from tool messages
        if msg_type == "tool":
            tool_name = getattr(m, "name", None)
            tool_call_id = getattr(m, "tool_call_id", None)
            if tool_name:
                d["tool_name"] = tool_name
            if tool_call_id:
                d["tool_call_id"] = tool_call_id
        
        history.append(d)
    
    return history


def extract_stats_from_response(messages) -> dict:
    """Extract token usage stats from the last AI message's response_metadata."""
    stats = {
        "calls_total": 0,
        "calls_with_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    
    for m in messages:
        if getattr(m, 'type', '') == 'ai':
            stats["calls_total"] += 1
            metadata = getattr(m, 'response_metadata', {})
            token_usage = metadata.get('token_usage', {})
            if token_usage:
                stats["calls_with_usage"] += 1
                stats["prompt_tokens"] += token_usage.get("prompt_tokens", 0) or 0
                stats["completion_tokens"] += token_usage.get("completion_tokens", 0) or 0
                stats["total_tokens"] += token_usage.get("total_tokens", 0) or 0
    
    return stats


@wrap_tool_call
async def _handle_tool_errors(request, handler):
    """Middleware: catches tool errors and returns them as messages to the LLM."""
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {type(e).__name__}: {str(e)}",
            tool_call_id=request.tool_call["id"],
        )


class LangChainMCPClient:
    """Wrapper around LangChain's MCP adapters for T-MAP experiments.
    
    Provides:
    - Multi-server MCP connection via MultiServerMCPClient
    - Agent creation with configurable LLM (OpenRouter support)
    - Clean history conversion compatible with T-MAP pipeline
    - Token usage tracking
    """
    
    def __init__(
        self,
        server_connections: dict,
        target_model: str,
        target_model_api: str = None,
        target_model_api_token: str = None,
        temperature: float = 0,
        max_tool_calls: int = 100,
        extra_body: dict = None,
    ):
        self.server_connections = server_connections
        self.target_model = target_model
        self.target_model_api = target_model_api
        self.target_model_api_token = target_model_api_token
        self.temperature = temperature
        self.max_tool_calls = max_tool_calls
        self.extra_body = extra_body
        
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.tools = None
        self.agent = None
        self.tool_defs = None  # OpenAI function format for prompts
        
        self._stats = {
            "calls_total": 0,
            "calls_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    
    async def connect(self):
        """Connect to MCP servers and create the agent (stateful session)."""
        # 1. Create MultiServerMCPClient (no context manager in 0.1.0+)
        #    This keeps server processes alive across tool calls
        self.mcp_client = MultiServerMCPClient(self.server_connections)
        
        # 2. Load tools (await connects to servers and returns tools)
        self.tools = await self.mcp_client.get_tools()
        
        # 3. Build tool_defs in OpenAI function format (for prompts)
        self.tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.args_schema.model_json_schema() if hasattr(tool.args_schema, 'model_json_schema') else {},
                },
            }
            for tool in self.tools
        ]
        
        # 4. Create LLM
        llm_kwargs = {
            "model": self.target_model,
            "temperature": self.temperature,
        }
        if self.target_model_api:
            llm_kwargs["base_url"] = self.target_model_api
        if self.target_model_api_token:
            llm_kwargs["api_key"] = self.target_model_api_token
        safe_model_kwargs = None
        if self.extra_body:
            # Chat Completions 경로(use_responses_api=False)에서는 reasoning 인자를 직접 받지 못해
            # TypeError( unexpected keyword 'reasoning')가 날 수 있다.
            safe_model_kwargs = {k: v for k, v in self.extra_body.items() if k != "reasoning"}
            if safe_model_kwargs:
                llm_kwargs["model_kwargs"] = safe_model_kwargs
            
        # OpenRouter + 일부 모델 조합에서 LangChain이 Responses API 포맷을 선택하면
        # tool 결과(content/output)가 배열로 전달되어 400 invalid_prompt가 날 수 있다.
        # 가능한 경우 Chat Completions 경로를 강제해 포맷 불일치를 피한다.
        try:
            self.llm = ChatOpenAI(use_responses_api=False, **llm_kwargs)
        except TypeError:
            # 구버전 호환 fallback
            self.llm = ChatOpenAI(**llm_kwargs)
        
        # 5. Create agent with tool error middleware.
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            middleware=[_handle_tool_errors],
        )
        
        return self.tool_defs
    
    async def process_query(self, query: str) -> list[dict]:
        """Run a single query through the agent and return clean history.
        
        Args:
            query: The attack prompt to send to the target agent
            
        Returns:
            List of message dicts in our standard format:
            [{"role": "user", "content": "..."}, {"role": "assistant", ...}, ...]
        """
        try:
            response = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": self.max_tool_calls},
            )
            raw_messages = response["messages"]

            run_stats = extract_stats_from_response(raw_messages)
            for k in self._stats:
                self._stats[k] += run_stats.get(k, 0)

            return convert_messages_to_history(raw_messages)

        except Exception as e:
            # On catastrophic failure, return error history
            return [
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"System Error: {type(e).__name__}: {str(e)}"},
            ]

    
    def stats(self) -> dict:
        """Return accumulated token usage stats."""
        return dict(self._stats)
    
    async def cleanup(self):
        """Cleanup MCP server sessions and processes."""
        if self.mcp_client:
            try:
                # 0.1.0+: use close() instead of __aexit__
                if hasattr(self.mcp_client, 'close'):
                    await self.mcp_client.close()
                elif hasattr(self.mcp_client, '__aexit__'):
                    await self.mcp_client.__aexit__(None, None, None)
            except Exception:
                pass
            self.mcp_client = None
