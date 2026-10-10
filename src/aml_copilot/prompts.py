from . import config


def load(name):
    path = config.PROMPT_DIR / f"{config.PROMPT_VERSION}_{name}.txt"
    return path.read_text(encoding="utf-8")
