"""Text normalisation shared by training and inference.

Using the same function in both places avoids training/serving skew: the model is trained on
the same kind of cleaned text it sees in production (plain text, URLs replaced by a token).
"""
import html
import re

MAX_CHARS = 6000

_SCRIPT_STYLE = re.compile(r"<(script|style|head)\b.*?</\1\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LONG_TOKEN = re.compile(r"\S{40,}")          # base64 blobs, tracking ids
_SPACE = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n\s*\n+")

# Words that only identify one data source (e.g. the Enron corpus). Removing them stops the
# model from using "this looks like an Enron email" as a shortcut for "legitimate".
SOURCE_ARTIFACTS = re.compile(
    r"\b(enron|ect|hou|hpl|hplc|ees|eott|corp|enronxgate|ena|sitara|daren|lay|skilling|"
    r"vince|kaminski|louise|kitchen|phillip|gas|nom|nomination|meter|volumes?)\b", re.I)


def clean_text(text, max_chars=MAX_CHARS):
    """Return plain text: HTML removed, URLs/emails replaced by tokens, whitespace collapsed."""
    if not text:
        return ""
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _LONG_TOKEN.sub(" ", text)
    text = SOURCE_ARTIFACTS.sub(" ", text)
    text = _SPACE.sub(" ", text)
    text = _BLANKS.sub("\n", text).strip()
    return text[:max_chars]


# ---- extra normalisation used ONLY for the ML model ---------------------------------------------
# Public corpora differ in era, language and tooling. Without these steps the model learns those
# differences (years such as "2001", the maintainer's "phishing pot" tag, German words) instead of phishing.
_PLACEHOLDERS = re.compile(r"\b(urltoken|emailtoken)\b")
_DIGITS = re.compile(r"\d+")
_DATASET_MARKERS = re.compile(r"phishing[ _-]?pot|phishing", re.I)

ML_EXTRA_STOPWORDS = ["com", "http", "https", "www", "unsubscribe", "email", "emails", "mail", "sent",
                      "message", "subject", "fw", "fwd", "re", "pm", "am", "pot", "replying"]

_COMMON_ENGLISH = frozenset("the and to of a in is you your for on that this with it be are as at have or from by "
                            "not we will can if our an was has please thank thanks account".split())


def ml_text(text):
    """Cleaned text with era/source shortcuts removed. This is what the ML model reads."""
    text = _PLACEHOLDERS.sub(" ", clean_text(text))
    text = _DATASET_MARKERS.sub(" ", text)
    return _DIGITS.sub(" ", text)


def looks_english(text, min_words=8, min_ratio=0.12):
    """Cheap English check so a model trained on mixed-language data cannot use language as a shortcut."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < min_words:
        return False
    return sum(w in _COMMON_ENGLISH for w in words) / len(words) >= min_ratio
