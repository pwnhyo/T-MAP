# Run Examples (You can change the MCP servers and API URL. )
# Slack
python run_main.py \
  --server Slack \
  --attacker_model "deepseek-chat" \
  --attacker_model_api "https://api.deepseek.com" \
  --attacker_model_api_token "[DEEPSEEK_API_KEY]" \
  --target_model "openai/gpt-5-mini" \
  --target_model_api "https://openrouter.ai/api/v1" \
  --target_model_api_token "[OPENROUTER_API_KEY]" \
  --level_judge_model "deepseek-chat" \
  --level_judge_model_api "https://api.deepseek.com" \
  --level_judge_model_api_token "[DEEPSEEK_API_KEY]" \
  --stdio_server_cmd npx \
  --stdio_server_args "-y slack-mcp-server@latest --transport stdio" \
  --stdio_server_envs "SLACK_MCP_XOXP_TOKEN=[SLACK_MCP_XOXP_TOKEN] SLACK_MCP_ADD_MESSAGE_TOOL=true" \
  --iteration 100 \
  --mutation_n 3 \
  --checkpoint_interval 20

# CodeExecutor
python run_main.py \
  --server CodeExecutor \
  --attacker_model "deepseek-chat" \
  --attacker_model_api "https://api.deepseek.com" \
  --attacker_model_api_token "[DEEPSEEK_API_KEY]" \
  --target_model "openai/gpt-5-mini" \
  --target_model_api "https://openrouter.ai/api/v1" \
  --target_model_api_token "[OPENROUTER_API_KEY]" \
  --level_judge_model "deepseek-chat" \
  --level_judge_model_api "https://api.deepseek.com" \
  --level_judge_model_api_token "[DEEPSEEK_API_KEY]" \
  --stdio_server_cmd docker \
  --stdio_server_args "run -i --rm --workdir /app mcp-code-executor:latest" \
  --iteration 100 \
  --mutation_n 3 \
  --checkpoint_interval 20 

# Slack + CodeExecutor (Multi-MCP Configuration)
python run_main.py \
  --server "Slack,CodeExecutor" \
  --attacker_model "deepseek-chat" \
  --attacker_model_api "https://api.deepseek.com" \
  --attacker_model_api_token "[DEEPSEEK_API_KEY]" \
  --target_model "openai/gpt-5-mini" \
  --target_model_api "https://openrouter.ai/api/v1" \
  --target_model_api_token "[OPENROUTER_API_KEY]" \
  --level_judge_model "deepseek-chat" \
  --level_judge_model_api "https://api.deepseek.com" \
  --level_judge_model_api_token "[DEEPSEEK_API_KEY]" \
  --stdio_server_cmd_1 npx \
  --stdio_server_args_1 "-y slack-mcp-server@latest --transport stdio" \
  --stdio_server_envs_1 "SLACK_MCP_XOXP_TOKEN=[SLACK_MCP_XOXP_TOKEN] SLACK_MCP_ADD_MESSAGE_TOOL=true" \
  --stdio_server_cmd_2 docker \
  --stdio_server_args_2 "run -i --rm --workdir /app mcp-code-executor:latest" \
  --iteration 100 \
  --mutation_n 3 \
  --checkpoint_interval 20