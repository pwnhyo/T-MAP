import os
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

class LLM:
    def __init__(self, model, model_api=None, api_key=None, extra_body=None):
        self.model = model
        self.model_api = model_api
        self.extra_body = extra_body
        self._lock = threading.Lock()
        self._stats = {
            "calls_total": 0,
            "calls_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        if model_api:
            self.client = OpenAI(base_url=model_api, api_key=api_key)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, system_prompt, user_prompt, extra_body=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        # Use provided extra_body, or fall back to instance-level extra_body
        effective_extra_body = extra_body if extra_body is not None else self.extra_body

        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Build kwargs for the API call
                api_kwargs = {
                    "model": self.model,
                    "messages": messages,
                }
                if effective_extra_body is not None:
                    api_kwargs["extra_body"] = effective_extra_body
                
                response = self.client.chat.completions.create(**api_kwargs)
                assistant_msg = response.choices[0].message.content
                usage = getattr(response, "usage", None)
                with self._lock:
                    self._stats["calls_total"] += 1
                    if usage is not None:
                        self._stats["calls_with_usage"] += 1
                        self._stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                        self._stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                        self._stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
                return assistant_msg
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[*] API Error ({type(e).__name__}): {e}. Retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f" [!!!] API Final Failure: {e}")
                    raise e
    
    def batch_chat(self, system_prompts, user_prompts, max_workers=4, extra_body=None):
        responses = []
        # Support None for system_prompts
        if system_prompts is None:
            system_prompts = [None] * len(user_prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(self.chat, system_prompt=sys_prompt, user_prompt=user_prompt, extra_body=extra_body)
                futures.append(future)

            for future in futures:
                try:
                    responses.append(future.result())
                except Exception as e:
                    responses.append(f"Error: API Request Failed. {type(e).__name__}: {str(e)}")

        return responses

    def stats(self):
        with self._lock:
            return dict(self._stats)
