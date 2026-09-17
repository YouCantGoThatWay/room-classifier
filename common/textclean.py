"""Cleaning of MUD room text: color codes, tildes, whitespace."""
import re

_CODE_RES = [
    re.compile(r"\x1b\[[0-9;]*m"),   # raw ANSI
    re.compile(r"&[a-zA-Z0-9]"),     # SMAUG &-codes
    re.compile(r"@[a-zA-Z0-9]"),     # tbaMUD @-codes
    re.compile(r"\{[a-zA-Z]"),       # ROM {-codes
]
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")

MIN_DESC_CHARS = 30


def strip_codes(text: str) -> str:
    for rx in _CODE_RES:
        text = rx.sub("", text)
    return text


def clean_text(text: str) -> str:
    text = strip_codes(text).replace("~", "")
    return _WS.sub(" ", text).strip()


def is_trivial(name: str, description: str) -> bool:
    return len(clean_text(description)) < MIN_DESC_CHARS


def build_text(name: str, description: str) -> str:
    """The one canonical model input format. C# must reproduce this."""
    return f"{clean_text(name)}\n{clean_text(description)}"


def normalized_key(name: str, description: str) -> str:
    joined = f"{name} {description}".lower()
    joined = strip_codes(joined).replace("~", "")
    return _WS.sub(" ", _PUNCT.sub("", joined)).strip()
