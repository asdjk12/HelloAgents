from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
import os


DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2


def main():
    load_dotenv(dotenv_path='.env')
    timeout = float(os.getenv("OPENAI_TIMEOUT") or os.getenv("LLM_TIMEOUT") or DEFAULT_TIMEOUT_SECONDS)
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES") or DEFAULT_MAX_RETRIES)

    client = OpenAI(
        api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
        base_url="https://api-inference.modelscope.cn/v1",
        timeout=timeout,
        max_retries=max_retries,
    )

    try:
        models = client.models.list()
        for m in models.data:
            print(m.id)
    except APITimeoutError:
        print(
            "Request timed out while listing models. "
            "Increase OPENAI_TIMEOUT/LLM_TIMEOUT in .env and retry."
        )
        return 1
    except APIConnectionError as exc:
        print(f"Connection error while listing models: {exc}")
        return 1
    except APIStatusError as exc:
        print(f"API returned HTTP {exc.status_code} while listing models: {exc}")
        return 1
    return 0

if __name__ == "__main__":
    main()
