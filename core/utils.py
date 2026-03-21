import argparse

def get_base_parser():
    parser = argparse.ArgumentParser()
    
    # ── Server Connection (Legacy: single server) ──
    parser.add_argument("--remote_server_token", type=str, default=None)
    parser.add_argument("--remote_server_url", type=str, default=None)
    parser.add_argument("--stdio_server_cmd", type=str, default=None)
    parser.add_argument("--stdio_server_args", type=str, default=None)
    parser.add_argument("--stdio_server_envs", type=str, default=None)

    # ── Server Connection (Indexed stdio: multi-server without server_config) ──
    # Example:
    # --server Gmail,GoogleSearch
    # --stdio_server_cmd_1 npx --stdio_server_args_1 "@gongrzhe/server-gmail-autoauth-mcp"
    # --stdio_server_cmd_2 npx --stdio_server_args_2 "/path/to/google-mcp"
    for idx in range(1, 9):
        parser.add_argument(f"--stdio_server_cmd_{idx}", type=str, default=None)
        parser.add_argument(f"--stdio_server_args_{idx}", type=str, default=None)
        parser.add_argument(f"--stdio_server_envs_{idx}", type=str, default=None)
    
    # ── Server Connection (New: config-based, supports multi-server) ──
    parser.add_argument("--server_config", type=str, default=None,
                        help="Path to servers.json config file (e.g., environments/servers.json)")
    
    # ── Models ──
    parser.add_argument("--attacker_model", type=str, default="deepseek-chat")
    parser.add_argument("--attacker_model_api", type=str, default="https://api.deepseek.com")
    parser.add_argument("--attacker_model_api_token", type=str, required=True)
    parser.add_argument("--target_model", type=str, default="gpt-5.2")
    parser.add_argument("--target_model_api", type=str, default=None)
    parser.add_argument("--target_model_api_token", type=str, default=None)
    parser.add_argument("--level_judge_model", type=str, default=None,
                        help="Optional separate model for level judging (L0-L3).")
    parser.add_argument("--level_judge_model_api", type=str, default=None,
                        help="Optional API base URL for level judge model.")
    parser.add_argument("--level_judge_model_api_token", type=str, default=None,
                        help="Optional API token for level judge model.")
    
    # ── Experiment Settings ──
    parser.add_argument("--server", type=str, required=True,
                        help="Server name(s). Single: 'Gmail'. Multi: 'Gmail,GoogleSearch'")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--query_timeout", type=int, default=300,
                        help="Per-query timeout (seconds) for target MCP execution in evaluate_batch.")
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--mutation_n", type=int, default=3)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    
    # ── Paths ──
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--log_dir", type=str, default="logs")
    
    # ── LLM Extra Options (for attacker model) ──
    parser.add_argument("--reasoning_effort", type=str, default=None, choices=["low", "medium", "high", "none"],
                        help="Reasoning effort for attacker model (low, medium, high, none)")
    # ── LLM Extra Options (for target model) ──
    parser.add_argument("--target_reasoning_effort", type=str, default=None, choices=["low", "medium", "high", "none"],
                        help="Reasoning effort for target model (low, medium, high, none)")
    
    return parser
