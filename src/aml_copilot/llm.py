import time

from openai import OpenAI

from . import config
from .observability import log_call

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def chat(messages, step, model=None, tools=None, json_mode=False, temperature=0.0):
    model = model or config.CHAT_MODEL
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.perf_counter()
    try:
        resp = _get_client().chat.completions.create(**kwargs)
    except Exception as exc:
        log_call(step, model, (time.perf_counter() - start) * 1000, 0, 0, error=str(exc)[:200])
        raise

    usage = resp.usage
    log_call(step, model, (time.perf_counter() - start) * 1000,
             usage.prompt_tokens, usage.completion_tokens)
    return resp.choices[0].message
