"""Analysis engine: turns an email (or a bare link) into one explainable risk score.

Four signals are combined:
  1. ML probability   - calibrated TF-IDF + Logistic Regression on the cleaned message text
  2. Link & domain    - look-alike / typosquatted brands, IP hosts, shorteners, bad TLDs, hidden
                        redirects, anchor-text mismatch
  3. Keywords         - urgency, credential requests, threats, rewards (weak signals, capped)
  4. Headers & files  - SPF/DKIM/DMARC, sender mismatches, free-mail impersonation, risky attachments
Header and attachment checks only run when the input contains email headers (or is an .eml file).
"""
import html
import logging
import os
import re
import threading
from email import message_from_string, policy
from email.parser import BytesParser
from email.utils import parseaddr
from typing import NamedTuple
from urllib.parse import parse_qs, unquote, urlparse

import joblib
import tldextract
from rapidfuzz import fuzz

from preprocess import clean_text, ml_text

log = logging.getLogger("inboxguard")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ---- tunable scoring constants (see evaluate.py) -------------------------------------------
ML_WEIGHT = 0.5          # share of the score taken from the ML model for normal-length text
ML_WEIGHT_SHORT = 0.25   # ... for very short text (little for the model to read)
STRONG_RULES = 60        # rule score at or above this overrides the blend
WARN_AT, CRITICAL_AT = 45, 65   # chosen with evaluate.py: Critical never flagged legitimate mail in testing
KEYWORD_CAP = 15         # keywords are weak evidence: they can never create a verdict on their own
MAX_LINKS = 40
MAX_SCAN_CHARS = 50_000   # analyse at most this much text: bounds CPU time on huge or hostile input


class Flag(NamedTuple):
    label: str
    severity: str    # high | medium | low
    points: int


# ---- model (loaded lazily, never trained at import time) ------------------------------------
_bundle, _bundle_lock, _names = None, threading.Lock(), None


def get_model():
    """Return the model bundle, or None if model.pkl is missing/unusable (rules-only mode)."""
    global _bundle
    if _bundle is None:
        with _bundle_lock:
            if _bundle is None:
                if not os.path.exists(MODEL_PATH):
                    log.warning("model.pkl not found - running with rules only")
                    return None
                try:
                    _bundle = joblib.load(MODEL_PATH)
                except Exception:
                    log.exception("could not load model.pkl - running with rules only")
                    return None
    return _bundle


def ml_probability(text):
    """(probability 0-100 or None, list of words that pushed the score towards phishing)."""
    global _names
    bundle = get_model()
    if not bundle or bundle.get("demo"):         # a demo model must never influence a verdict
        return None, []
    p = float(bundle["pipeline"].predict_proba([text])[0][1]) * 100
    terms = []
    try:
        if _names is None:
            _names = bundle["vectorizer"].get_feature_names_out()
        coo = bundle["vectorizer"].transform([text]).multiply(bundle["lr"].coef_[0]).tocoo()
        ranked = sorted(zip(coo.data, coo.col), reverse=True)
        terms = [str(_names[c]) for v, c in ranked if v > 0 and _names[c] not in ("urltoken", "emailtoken")][:6]
    except Exception:
        log.debug("could not explain ML score", exc_info=True)
    return p, terms


# ---- domains --------------------------------------------------------------------------------
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)   # offline: bundled public-suffix list


def registered_domain(host):
    host = (host or "").strip(".").lower()
    r = _TLD(host)
    reg = r.top_domain_under_public_suffix if hasattr(r, "top_domain_under_public_suffix") else r.registered_domain
    return reg or host


IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
DIGIT_FIX = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a"})

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc"}
BAD_TLDS = {"xyz", "top", "tk", "gq", "ml", "cf", "ga", "loan", "icu", "buzz", "click", "zip", "mov",
            "cyou", "rest", "monster", "sbs", "cfd"}
WEAK_TLDS = {"info", "work", "support", "online", "site", "live", "shop", "store", "vip", "club"}
# Country sites of a brand (amazon.in, google.co.uk). Generic TLDs (.com/.net/.org) are deliberately absent:
# a brand's .com must be listed in OFFICIAL, otherwise "paypal.net" or "hdfc.net" would pass as official.
BRAND_TLDS = {"in", "co.in", "co.uk", "uk", "de", "fr", "ca", "com.au", "au", "co.jp", "jp", "es", "it", "nl",
              "com.br", "br", "ae", "sg", "com.sg", "mx", "com.mx", "ie", "se", "pl", "ch", "be"}
SUSPICIOUS_WORDS = {"secure", "security", "login", "signin", "verify", "verification", "support", "help",
                    "account", "accounts", "update", "billing", "service", "services", "alert", "alerts",
                    "online", "bank", "banking", "confirm", "customer", "care", "wallet", "pay", "payment",
                    "auth", "recovery", "unlock", "refund", "helpdesk", "portal", "team", "id", "mail"}

OFFICIAL = {
    "paypal": {"paypal.com", "paypal.me", "paypalobjects.com"},
    "chase": {"chase.com", "jpmorganchase.com"},
    "amazon": {"amazon.com", "amazonaws.com", "amazon-adsystem.com", "amazonses.com", "amazonpay.in"},
    "microsoft": {"microsoft.com", "microsoftonline.com", "office.com", "office365.com", "live.com", "outlook.com", "microsoftstore.com"},
    "netflix": {"netflix.com", "nflxext.com"},
    "apple": {"apple.com", "icloud.com"},
    "dhl": {"dhl.com", "dhl.de"},
    "hdfc": {"hdfcbank.com", "hdfcbank.net"},
    "google": {"google.com", "googlemail.com", "gstatic.com", "googleusercontent.com", "googleapis.com"},
    "facebook": {"facebook.com", "fb.com", "facebookmail.com"},
    "instagram": {"instagram.com"}, "whatsapp": {"whatsapp.com"}, "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"}, "docusign": {"docusign.com", "docusign.net"},
    "fedex": {"fedex.com"}, "ups": {"ups.com"}, "usps": {"usps.com"}, "irs": {"irs.gov"},
    "sbi": {"sbi.co.in", "onlinesbi.com", "onlinesbi.sbi"}, "icici": {"icicibank.com"},
    "paytm": {"paytm.com"}, "flipkart": {"flipkart.com"},
    "wellsfargo": {"wellsfargo.com"}, "bankofamerica": {"bankofamerica.com"},
}
FREEMAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "proton.me",
            "protonmail.com", "live.com", "icloud.com", "yahoo.co.in", "rediffmail.com", "mail.com"}
ROLE_RE = re.compile(r"\b(support|security|billing|helpdesk|help desk|customer (care|service)|account|admin|"
                     r"verification|alerts?|noreply|no-reply|bank|payroll|hr)\b", re.I)


def is_official(brand, reg):
    """True if `reg` (a registered domain) belongs to the brand: explicitly listed, or brand + a country TLD."""
    if reg in OFFICIAL.get(brand, ()):
        return True
    label, _, suffix = reg.partition(".")
    return label == brand and suffix in BRAND_TLDS


def _is_known_official(reg):
    return any(reg in doms for doms in OFFICIAL.values())


def impersonated_brand(host, reg=None):
    """(brand, kind) if `host` imitates a brand it does not belong to, else None.
    kind: lookalike (digit swaps like paypa1), typo (paypall), affix (paypal-secure),
          tldswap (the brand name on an unofficial TLD, e.g. paypal.net), name (brand word elsewhere in host)."""
    reg = reg or registered_domain(host)
    tokens = [t for t in re.split(r"[.\-_]", host) if t]
    for brand in OFFICIAL:
        if is_official(brand, reg):
            continue
        for tok in tokens:
            n = tok.translate(DIGIT_FIX).replace("rn", "m").replace("vv", "w")   # paypa1, arnazon (rn looks like m)
            if n == brand:
                if n != tok.lower():
                    return brand, "lookalike"
                if reg.partition(".")[0] == brand:
                    return brand, "tldswap"
                in_subdomain = host != reg and brand not in reg.split(".")[0]
                return brand, ("affix" if in_subdomain or any(t in SUSPICIOUS_WORDS for t in tokens) else "name")
            if len(brand) >= 5 and (n.startswith(brand) or n.endswith(brand)):
                rest = n.replace(brand, "", 1)
                if rest in SUSPICIOUS_WORDS:
                    return brand, "affix"
            if len(brand) >= 6 and abs(len(n) - len(brand)) <= 1 and fuzz.ratio(n, brand) >= 85:
                return brand, "typo"
    return None


# ---- links ----------------------------------------------------------------------------------
URL_RE = re.compile(r"""(?:https?://|www\.)(?:\[[0-9a-f:.]{2,45}\])?[^\s<>"')\]]*""", re.I)
BARE_RE = re.compile(r"""(?<![@\w./-])((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,8}[a-z]{2,24})(/[^\s<>"')\]]*)?""", re.I)
BARE_IP_RE = re.compile(r"""(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?/[^\s<>"')\]]*)""")
ANCHOR_RE = re.compile(r"""<a\s[^>]{0,500}?href\s*=\s*["']([^"']{1,2000})["'][^>]{0,500}>(.{0,500}?)</a>""", re.I | re.S)
REDIRECT_PARAMS = ("url", "u", "q", "redirect", "redirect_uri", "target", "dest", "destination", "link", "goto", "continue", "next", "r")


def deobfuscate(text):
    text = re.sub(r"\bhxxp(s?)://", lambda m: f"http{m.group(1)}://", text, flags=re.I)
    return re.sub(r"\[\s*\.\s*\]|\(\s*\.\s*\)|\[dot\]|\(dot\)", ".", text, flags=re.I)


_NUMERIC_PART = re.compile(r"0x[0-9a-f]+|\d+", re.I)


def is_ip_host(host):
    """True for dotted IPv4, IPv6, and the obfuscated forms browsers still accept (0x7f000001, 3232235777, 0300.0250.1.1)."""
    if not host:
        return False
    if IP_RE.match(host) or ":" in host:
        return True
    if re.fullmatch(r"0x[0-9a-f]{1,8}|\d{8,10}", host, re.I):
        return True
    parts = host.split(".")
    return 1 < len(parts) <= 4 and all(_NUMERIC_PART.fullmatch(p) for p in parts)


def _urlparse(url):
    """urlparse that never raises: malformed input (stray '[' or ']') must not crash the whole analysis."""
    u = url if "://" in url else "http://" + url
    try:
        return urlparse(u)
    except ValueError:
        return urlparse(u.replace("[", "").replace("]", ""))


def _host(url):
    parsed = _urlparse(url)
    try:
        return (parsed.hostname or "").lower()
    except ValueError:
        return ""


def extract_links(text):
    """Unique URLs in the text, including bare domains that carry a path or look suspicious."""
    text = deobfuscate(html.unescape(text))
    urls = [u.rstrip(".,;:!?)") for u in URL_RE.findall(text)]
    rest = URL_RE.sub(" ", text)
    for m in BARE_RE.finditer(rest):
        domain, path = m.group(1).lower(), m.group(2) or ""
        r = _TLD(domain)
        if not (r.suffix and r.domain):
            continue
        if path or r.suffix in BAD_TLDS or impersonated_brand(domain):
            urls.append(domain + path)
    urls += BARE_IP_RE.findall(rest)
    return list(dict.fromkeys(urls))[:MAX_LINKS]


def check_url(raw, _depth=0):
    url = raw.rstrip(".,;)")
    parsed = _urlparse(url)
    host = _host(url)
    reg = registered_domain(host)
    flags = []
    is_ip = is_ip_host(host)
    if is_ip:
        flags.append(Flag("Uses a raw IP address instead of a domain name", "high", 30))
    if "xn--" in host:
        flags.append(Flag("Punycode domain (possible look-alike characters)", "high", 25))
    if "@" in parsed.netloc:
        flags.append(Flag("Contains '@' to hide the real destination", "high", 25))
    if reg in SHORTENERS:
        flags.append(Flag("URL shortener hides the final destination", "medium", 15))
    tld = reg.rsplit(".", 1)[-1]
    if tld in BAD_TLDS:
        flags.append(Flag(f"Suspicious top-level domain (.{tld})", "medium", 15))
    elif tld in WEAK_TLDS:
        flags.append(Flag(f"Uncommon top-level domain (.{tld})", "low", 5))
    hit = None if is_ip else impersonated_brand(host, reg)
    if hit:
        brand, kind = hit
        if kind == "name":
            flags.append(Flag(f"Uses the name '{brand}' but the real domain is {reg}", "medium", 20))
        elif kind == "tldswap":
            flags.append(Flag(f"Uses the brand name '{brand}' on an unofficial domain ({reg})", "high", 30))
        else:
            flags.append(Flag(f"Impersonates '{brand}' but the real domain is {reg}", "high", 35))
    if parsed.scheme == "http":
        flags.append(Flag("Not encrypted (HTTP, not HTTPS)", "low", 8))
    sub = _TLD(host).subdomain if not is_ip else ""
    if sub.count(".") >= 1:                       # two or more subdomain labels; the public suffix (co.uk) is not counted
        flags.append(Flag("Excessive subdomains", "medium", 10))
    if host.count("-") >= 2:
        flags.append(Flag("Many hyphens in domain", "medium", 10))
    if len(url) > 75:
        flags.append(Flag("Unusually long URL", "low", 8))
    if _depth == 0:                                    # tracking links often wrap the real destination
        query = parse_qs(parsed.query)
        for key in REDIRECT_PARAMS:
            target = unquote((query.get(key) or [""])[0])
            if target.lower().startswith(("http://", "https://")):
                inner = check_url(target, _depth=1)
                if registered_domain(inner["domain"]) != reg:
                    flags += [Flag(f"Hidden redirect target: {f.label}", f.severity, f.points) for f in inner["flags"]]
                break
    return {"url": url, "domain": host, "flags": flags}


def check_anchors(body):
    """HTML links whose visible text names one site but whose href goes to another."""
    out = []
    for href, label in ANCHOR_RE.findall(body):
        text = re.sub(r"<[^>]+>", " ", label)
        target = registered_domain(_host(href))
        if not target:
            continue
        m = URL_RE.search(text) or BARE_RE.search(text)
        shown = registered_domain(_host(m.group(0))) if m else ""
        if shown and shown != target:
            out.append((Flag(f"Link text shows {shown} but actually goes to {target}", "high", 30), target))
            continue
        low = text.lower()
        for brand in OFFICIAL:
            if re.search(r"\b" + re.escape(brand) + r"\b", low) and not is_official(brand, target):
                out.append((Flag(f"Link labelled '{brand}' goes to {target}", "medium", 15), target))
                break
    return out


# ---- email parsing ---------------------------------------------------------------------------
HEADER_START = re.compile(r"^\s*(from|reply-to|return-path|received|authentication-results|subject|date|to|message-id):", re.I)
RISKY_EXT = {".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".com", ".pif", ".lnk", ".jar", ".msi", ".hta", ".ps1", ".iso", ".img"}
MEDIUM_EXT = {".html", ".htm", ".svg", ".docm", ".xlsm", ".pptm", ".zip", ".rar", ".7z", ".one", ".xll"}
DOUBLE_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|jpe?g|png|txt)\.(exe|scr|js|vbs|bat|cmd|com|lnk|html?)$", re.I)


def _hdr(msg, name):
    try:
        return str(msg.get(name) or "")
    except Exception:
        return ""


def _bodies(msg):
    """(plain text, html) of an email.message.Message; either may be empty."""
    plain = html_part = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart() or part.get_content_disposition() == "attachment":
                    continue
                kind = part.get_content_type()
                if kind == "text/plain" and not plain:
                    plain = str(part.get_content())
                elif kind == "text/html" and not html_part:
                    html_part = str(part.get_content())
        else:
            payload = str(msg.get_content())
            if msg.get_content_type() == "text/html":
                html_part = payload
            else:
                plain = payload
    except Exception:
        log.debug("could not decode message body", exc_info=True)
    return plain, html_part


def split_message(text):
    """Return (Message or None, plain, html). Headers are parsed only if the text starts like an email."""
    if not HEADER_START.match(text):
        return None, text, ""
    try:
        msg = message_from_string(text, policy=policy.default)
    except Exception:
        return None, text, ""
    if not (_hdr(msg, "From") or _hdr(msg, "Reply-To") or _hdr(msg, "Authentication-Results")):
        return None, text, ""
    plain, html_part = _bodies(msg)
    if not plain and not html_part:
        m = re.search(r"\r?\n\r?\n", text)
        plain = text[m.end():] if m else ""
    return msg, plain, html_part


def _addr_domain(value):
    addr = parseaddr(value or "")[1]
    return registered_domain(addr.split("@")[-1]) if "@" in addr else ""


def check_headers(msg):
    out = []
    from_raw = _hdr(msg, "From")
    name = parseaddr(from_raw)[0].lower()
    from_host = parseaddr(from_raw)[1].split("@")[-1].lower() if "@" in parseaddr(from_raw)[1] else ""
    from_dom = _addr_domain(from_raw)
    claimed = next((b for b in OFFICIAL if re.search(r"\b" + re.escape(b) + r"\b", name)), None)
    if claimed and from_dom and not is_official(claimed, from_dom):
        where = "a free mail service" if from_dom in FREEMAIL else from_dom
        out.append(Flag(f"Sender name says '{claimed}' but the email comes from {where}", "high", 30))
    elif from_dom in FREEMAIL and ROLE_RE.search(name):
        out.append(Flag("Sender presents as support/security/billing but uses a free mail service", "medium", 18))
    imp = impersonated_brand(from_host, from_dom) if from_host else None
    if imp:
        out.append(Flag(f"Sender domain {from_dom} imitates '{imp[0]}'", "high", 30))
    rt = _addr_domain(_hdr(msg, "Reply-To"))
    if rt and from_dom and rt != from_dom:
        if rt in FREEMAIL and from_dom not in FREEMAIL:
            out.append(Flag(f"Replies go to a free mail address ({rt}), not the sender domain ({from_dom})", "medium", 20))
        else:
            out.append(Flag(f"Reply-To ({rt}) differs from the sender domain ({from_dom})", "low", 8))
    rp = _addr_domain(_hdr(msg, "Return-Path"))
    if rp and from_dom and rp != from_dom:
        out.append(Flag(f"Return-Path ({rp}) differs from the sender domain ({from_dom})", "low", 4))
    auth = _hdr(msg, "Authentication-Results").lower()        # topmost header = added by your own mail server
    for proto in ("spf", "dkim", "dmarc"):
        m = re.search(proto + r"\s*=\s*(\w+)", auth)
        if m and m.group(1) in {"fail", "permerror"}:
            out.append(Flag(f"{proto.upper()} check failed", "high", 25))
        elif m and m.group(1) in {"softfail", "temperror"}:
            out.append(Flag(f"{proto.upper()} check returned {m.group(1)}", "medium", 15))
    if _hdr(msg, "Received-SPF").lower().startswith("fail") and "spf=fail" not in auth:
        out.append(Flag("SPF check failed", "high", 25))
    return out


def check_attachments(names):
    out = []
    for fn in names[:20]:
        low = fn.lower()
        ext = os.path.splitext(low)[1]
        if DOUBLE_EXT.search(low):
            out.append(Flag(f"Attachment '{fn}' hides an executable behind a document extension", "high", 35))
        elif ext in RISKY_EXT:
            out.append(Flag(f"Risky attachment type: {fn}", "high", 30))
        elif ext in MEDIUM_EXT:
            out.append(Flag(f"Attachment type often used to deliver phishing: {fn}", "medium", 15))
    return out


# ---- keywords --------------------------------------------------------------------------------
KEYWORDS = {
    "Urgency": ["urgent", "immediately", "within 24 hours", "act now", "final notice", "expires", "suspended", "last chance"],
    "Credential request": ["verify your", "confirm your", "update your", "password", "log in", "sign in", "ssn", "otp",
                           "security code", "credit card", "bank account"],
    "Threat": ["permanently", "unauthorized", "unusual activity", "locked", "restricted", "legal action", "suspicious"],
    "Reward / money": ["winner", "congratulations", "prize", "gift card", "payment failed", "redelivery fee", "claim your"],
}


def find_keywords(text):
    low = text.lower()
    found = {}
    for cat, words in KEYWORDS.items():
        hits = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", low)]
        if hits:
            found[cat] = hits
    return found


# ---- scoring ---------------------------------------------------------------------------------
def level_for(score):
    return "Critical" if score >= CRITICAL_AT else "Warning" if score >= WARN_AT else "Low risk"


def combine(ml, heur, words):
    """Blend ML probability (0-100 or None) with the rule score (0-100)."""
    if ml is None:
        return heur
    w = ML_WEIGHT if words >= 12 else ML_WEIGHT_SHORT if words >= 4 else 0.0
    score = w * ml + (1 - w) * heur
    return max(score, heur) if heur >= STRONG_RULES else score


def _build(msg, plain, html_part, attachments=()):
    subject = _hdr(msg, "Subject") if msg is not None else ""
    plain, html_part = plain[:MAX_SCAN_CHARS], html_part[:MAX_SCAN_CHARS]
    raw = (plain + "\n" + html_part).strip()
    text = clean_text(plain or html_part)
    content = (subject + "\n" + text).strip() or clean_text(raw)

    links = [check_url(u) for u in extract_links(raw)]
    keywords = find_keywords(content)
    indicators, heur = [], 0

    def add(flag, where):
        nonlocal heur
        indicators.append({"severity": flag.severity, "text": flag.label, "where": where})
        heur += flag.points

    if msg is not None:
        for f in check_headers(msg):
            add(f, "email headers")
    for f in check_attachments(list(attachments)):
        add(f, "attachments")
    for f, dom in check_anchors(raw):
        add(f, dom)
    for lk in links:
        for f in lk["flags"]:
            add(f, lk["domain"])
    kw_points = 0
    for cat, hits in keywords.items():
        pts = min(len(hits) * 3, 6)
        kw_points += pts
        indicators.append({"severity": "low", "text": f"{cat} language: {', '.join(hits[:4])}", "where": "message text"})
    heur = min(heur + min(kw_points, KEYWORD_CAP), 100)

    ml, terms = ml_probability(ml_text(content))
    # The model was trained on old public corpora and over-reacts to ordinary account mail. If every link
    # goes to a known official domain and no rule fired, trust the model less.
    if ml is not None and heur < 15 and links and all(_is_known_official(registered_domain(l["domain"])) for l in links):
        ml *= 0.5
    score = round(min(combine(ml, heur, len(content.split())), 100))
    if any(i["severity"] == "high" for i in indicators):     # a confirmed look-alike or failed SPF must never read "Low risk"
        score = max(score, WARN_AT)
    level = level_for(score)
    indicators.sort(key=lambda i: {"high": 0, "medium": 1, "low": 2}[i["severity"]])
    if level == "Low risk":
        summary = "No strong phishing signals were found. This is not a guarantee: only act on messages you expected."
    else:
        top = indicators[0]["text"] if indicators else "the wording of the message"
        summary = (f"{len(indicators)} warning sign(s) found. The strongest: {top}. "
                   + ("Do not click links or reply." if level == "Critical" else "Check the sender and links before acting."))
    return {"score": score, "level": level, "ml_score": None if ml is None else round(ml), "ml_available": ml is not None,
            "ml_terms": terms, "rule_score": heur, "summary": summary, "indicators": indicators,
            "links": [{"url": l["url"], "domain": l["domain"], "flags": [f._asdict() for f in l["flags"]]} for l in links],
            "keywords": keywords, "attachments": list(attachments), "has_headers": msg is not None}


def analyze(text):
    """Analyze pasted text: a bare link, an email body, or a full email with headers."""
    msg, plain, html_part = split_message(text)
    return _build(msg, plain, html_part)


def analyze_eml(data):
    """Analyze the bytes of an .eml file (headers, MIME body and attachment names)."""
    msg = BytesParser(policy=policy.default).parsebytes(data)
    plain, html_part = _bodies(msg)
    try:
        names = [p.get_filename() or "" for p in msg.iter_attachments()]
    except Exception:
        names = []
    return _build(msg, plain, html_part, [n for n in names if n])
