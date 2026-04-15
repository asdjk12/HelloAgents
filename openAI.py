import os
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


DEFAULT_AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2


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
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        load_local_env()

        self.model = model or os.getenv("AIHUBMIX_MODEL") or "gpt-4o-mini"
        api_key = api_key or os.getenv("AIHUBMIX_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("AIHUBMIX_BASE_URL") or DEFAULT_AIHUBMIX_BASE_URL
        timeout = timeout or float(
            os.getenv("OPENAI_TIMEOUT")
            or os.getenv("LLM_TIMEOUT")
            or DEFAULT_TIMEOUT_SECONDS
        )
        max_retries = max_retries if max_retries is not None else int(
            os.getenv("OPENAI_MAX_RETRIES")
            or DEFAULT_MAX_RETRIES
        )

        if not api_key:
            raise ValueError(
                "Missing API key. Set AIHUBMIX_API_KEY in .env, "
                "or pass api_key when creating OpenAICompatibleClient."
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Call the chat completions API and return plain text."""
        print(
            f"Calling model '{self.model}' via {self.base_url} "
            f"(timeout={self.timeout}s, retries={self.max_retries})..."
        )
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
        except APITimeoutError:
            print(
                "LLM API timeout: the request exceeded the configured timeout. "
                "Try increasing OPENAI_TIMEOUT/LLM_TIMEOUT in .env or check the upstream service."
            )
            return (
                "Error: the language model request timed out. "
                "Increase OPENAI_TIMEOUT (or LLM_TIMEOUT) in .env and retry."
            )
        except APIConnectionError as exc:
            print(f"LLM API connection error: {exc}")
            return (
                "Error: failed to connect to the language model service. "
                "Check your network, API base URL, and upstream availability."
            )
        except APIStatusError as exc:
            print(f"LLM API status error ({exc.status_code}): {exc}")
            return (
                f"Error: the language model service returned HTTP {exc.status_code}. "
                "Check the API key, model name, and upstream service status."
            )
        except Exception as exc:
            print(f"LLM API error: {type(exc).__name__}: {exc}")
            return "Error: failed to call the language model service."
