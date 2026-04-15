import os

from dotenv import load_dotenv
from openai import OpenAI


class HelloAgents:
    """Simple OpenAI-compatible chat client wrapper."""

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ) -> None:
        load_dotenv(dotenv_path=".env")

        self.model = model or os.getenv("MODELSCOPE_MODEL_ID")
        apiKey = apiKey or os.getenv("MODELSCOPE_SDK_TOKEN")
        baseUrl = baseUrl or os.getenv("MODELSCOPE_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not self.model:
            raise ValueError("Missing model id")
        if not apiKey:
            raise ValueError("Missing API key")
        if not baseUrl:
            raise ValueError("Missing baseUrl")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def _normalize_messages(self, message):
        if isinstance(message, str):
            return [{"role": "user", "content": message}]

        if not isinstance(message, list):
            raise ValueError("message must be a string or a list of chat messages")

        normalized_messages = []
        for item in message:
            if not isinstance(item, dict):
                raise ValueError("each message must be a dict with role/content")

            role = str(item.get("role", "user")).strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Unsupported message role: {item.get('role')}")

            normalized_messages.append(
                {
                    "role": role,
                    "content": item.get("content", ""),
                }
            )

        return normalized_messages

    def think(self, message, temperature: float = 0):
        print(f"正在调用{self.model}模型")

        try:
            normalized_messages = self._normalize_messages(message)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                temperature=temperature,
                stream=True,
            )

            print("模型调用成功")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)

            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None


if __name__ == "__main__":
    try:
        client = HelloAgents()

        exampleMessage = [
            {"role": "system", "content": "You are a helpful assistant for Python code."},
            {"role": "user", "content": "写一个简单的递归逻辑"},
        ]

        print("--- 调用LLM ---")
        responseText = client.think(exampleMessage)
        if responseText:
            print("\n\n--- 模型输出 ---")
            print(responseText)

    except ValueError as e:
        print(e)
