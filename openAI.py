import os
from pathlib import Path

from openai import OpenAI


DEFAULT_AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"


def load_local_env(env_file: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    env_path = Path(env_file)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class OpenAICompatibleClient:
    """OpenAI-compatible client with AIHubMix defaults."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        load_local_env()

        self.model = model or os.getenv("AIHUBMIX_MODEL") or "gpt-4o-mini"
        api_key = api_key or os.getenv("AIHUBMIX_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("AIHUBMIX_BASE_URL") or DEFAULT_AIHUBMIX_BASE_URL

        if not api_key:
            raise ValueError(
                "Missing API key. Set AIHUBMIX_API_KEY in .env, "
                "or pass api_key when creating OpenAICompatibleClient."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Call the chat completions API and return plain text."""
        print("Calling model...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            answer = response.choices[0].message.content or ""
            print("Model response received.")
            return answer
        except Exception as exc:
            print(f"LLM API error: {exc}")
            return "Error: failed to call the language model service."
