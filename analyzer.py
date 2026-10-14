"""Step 2 - Analysis engine.

Combines four signals into one risk score:
  1. ML probability        (TF-IDF + Logistic Regression on the message text)
  2. Link & domain checks  (look-alikes, IP hosts, shorteners, bad TLDs, anchor-text mismatch ...)
  3. Keyword checks        (urgency, credential requests, threats, rewards)
  4. Header checks         (SPF/DKIM/DMARC results, From/Reply-To/Return-Path mismatch, spoofed display name)
Header checks only run when the pasted text contains email headers.
"""
import os
import re
from email import message_from_string
from email.utils import parseaddr
from urllib.parse import urlparse

import joblib

from train_model import MODEL_PATH, train

# Load the model once; train it automatically on first run.
MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train()

URL_RE = re.compile(r"""(?:https?://|www\.)[^\s<>"')]+""", re.I)
ANCHOR_RE = re.compile(r"""<a\s[^>]*href\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""", re.I | re.S)
HEADER_START = re.compile(r"^\s*(from|reply-to|return-path|received|authentication-results|subject|date|to|message-id):", re.I)
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
DIGIT_FIX = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a"})

KEYWORDS = {
    "Urgency": ["urgent", "immediately", "within 24 hours", "act now", "final notice", "expires", "expire", "suspended", "last chance", "today"],
    "Credential request": ["verify your", "confirm your", "password", "log in", "login", "sign in", "ssn", "otp", "security code", "credit card", "bank account", "pin"],
    "Threat": ["permanently", "unauthorized", "unusual activity", "locked", "restricted", "legal action", "closed", "suspicious"],
    "Reward / money": ["winner", "congratulations", "prize", "gift card", "refund", "claim", "payment failed", "billing", "redelivery fee"],
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
BAD_TLDS = {"xyz", "top", "tk", "click", "work", "info", "zip", "gq", "ml", "cf", "ga", "support", "loan", "ru", "icu", "buzz"}

# brand -> extra official domains (the brand's own .com and country sites like amazon.co.uk are accepted automatically)
OFFICIAL = {
    "paypal": {"paypal.com", "paypal.me", "paypalobjects.com"},
    "chase": {"chase.com", "jpmorganchase.com"},
    "amazon": {"amazon.com", "amazonaws.com"},
    "microsoft": {"microsoft.com", "microsoftonline.com", "office.com", "office365.com", "live.com", "outlook.com"},
    "netflix": {"netflix.com", "nflxext.com"},
    "apple": {"apple.com", "icloud.com"},
    "dhl": {"dhl.com", "dhl.de"},
    "hdfc": {"hdfcbank.com", "hdfcbank.net"},
    "google": {"google.com", "googlemail.com", "gstatic.com", "googleusercontent.com"},
    "facebook": {"facebook.com", "fb.com", "facebookmail.com"},
    "instagram": {"instagram.com"}, "whatsapp": {"whatsapp.com"}, "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"}, "docusign": {"docusign.com", "docusign.net"},
    "fedex": {"fedex.com"}, "ups": {"ups.com"}, "usps": {"usps.com"}, "irs": {"irs.gov"},
    "sbi": {"sbi.co.in", "onlinesbi.com", "onlinesbi.sbi"}, "icici": {"icicibank.com"},
    "paytm": {"paytm.com"}, "flipkart": {"flipkart.com"},
    "wellsfargo": {"wellsfargo.com"}, "bankofamerica": {"bankofamerica.com"},
}
FREEMAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "proton.me", "protonmail.com"}


def registered_domain(host):
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "org", "gov", "ac", "net"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def is_official(brand, reg):
    if reg in OFFICIAL.get(brand, set()):
        return True
    first, _, rest = reg.partition(".")
    return first == brand and rest != "" and rest not in BAD_TLDS  # e.g. amazon.in, amazon.co.uk


def impersonated_brand(host):
    """Return the brand a host pretends to be (including digit look-alikes like paypa1), else None."""
    reg = registered_domain(host)
    for tok in re.split(r"[.\-_]", host):
        n = tok.translate(DIGIT_FIX)
        for brand in OFFICIAL:
            hit = n == brand or (len(brand) >= 6 and (n.startswith(brand) or n.endswith(brand)))
            if hit and not is_official(brand, reg):
                return brand
    return None


def check_url(raw):
    url = raw.rstrip(".,;)")
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    reg = registered_domain(host)
    flags = []  # (label, severity, points)

    if IP_RE.match(host):
        flags.append(("Uses a raw IP address instead of a domain name", "high", 30))
    if "xn--" in host:
        flags.append(("Punycode domain (possible look-alike characters)", "high", 25))
    if "@" in parsed.netloc:
        flags.append(("Contains '@' to hide the real destination", "high", 25))
    if reg in SHORTENERS:
        flags.append(("URL shortener hides the final destination", "medium", 15))
    if reg.rsplit(".", 1)[-1] in BAD_TLDS:
        flags.append((f"Suspicious top-level domain (.{reg.rsplit('.', 1)[-1]})", "medium", 15))
    brand = impersonated_brand(host) if not IP_RE.match(host) else None
    if brand:
        flags.append((f"Impersonates '{brand}' but the real domain is {reg}", "high", 35))
    if parsed.scheme == "http":
        flags.append(("Not encrypted (HTTP, not HTTPS)", "low", 8))
    if host.count(".") >= 3 and not IP_RE.match(host):
        flags.append(("Excessive subdomains", "medium", 10))
    if host.count("-") >= 2:
        flags.append(("Many hyphens in domain", "medium", 10))
    if len(url) > 75:
        flags.append(("Unusually long URL", "low", 8))
    return {"url": url, "domain": host, "flags": flags}


def check_anchors(body):
    """HTML links whose visible text shows one domain but whose href goes to another."""
    out = []
    for href, label in ANCHOR_RE.findall(body):
        shown = URL_RE.search(re.sub(r"<[^>]+>", "", label))
        if not shown:
            continue
        a = registered_domain((urlparse(shown.group(0) if "://" in shown.group(0) else "http://" + shown.group(0)).hostname or "").lower())
        b = registered_domain((urlparse(href).hostname or "").lower())
        if a and b and a != b:
            out.append((f"Link text shows {a} but actually goes to {b}", "high", 30, b))
    return out


def split_message(text):
    """Return (email Message or None, body). Headers are only parsed if the text starts like an email header."""
    if not HEADER_START.match(text):
        return None, text
    m = re.search(r"\r?\n\r?\n", text)
    head, body = (text[:m.start()], text[m.end():]) if m else (text, "")
    msg = message_from_string(head)
    if not (msg.get("From") or msg.get("Reply-To") or msg.get("Authentication-Results")):
        return None, text
    return msg, body


def addr_domain(value):
    addr = parseaddr(value or "")[1]
    return registered_domain(addr.split("@")[-1].lower()) if "@" in addr else ""


def check_headers(msg):
    out = []  # (label, severity, points)
    from_dom = addr_domain(msg.get("From"))
    name = parseaddr(msg.get("From", ""))[0].lower()

    for brand in OFFICIAL:
        if re.search(r"\b" + re.escape(brand) + r"\b", name) and from_dom and not is_official(brand, from_dom):
            out.append((f"Sender name says '{brand}' but the email comes from {from_dom}", "high", 30))
            break
    rt = addr_domain(msg.get("Reply-To"))
    if rt and from_dom and rt != from_dom:
        out.append((f"Reply-To ({rt}) differs from the sender domain ({from_dom})", "medium", 20))
    rp = addr_domain(msg.get("Return-Path"))
    if rp and from_dom and rp != from_dom:
        out.append((f"Return-Path ({rp}) differs from the sender domain ({from_dom})", "low", 6))

    auth = " ".join(msg.get_all("Authentication-Results") or []).lower()
    for proto in ("spf", "dkim", "dmarc"):
        m = re.search(proto + r"\s*=\s*(\w+)", auth)
        if m and m.group(1) in {"fail", "permerror"}:
            out.append((f"{proto.upper()} check failed", "high", 25))
        elif m and m.group(1) in {"softfail", "temperror"}:
            out.append((f"{proto.upper()} check returned {m.group(1)}", "medium", 15))
    if (msg.get("Received-SPF") or "").lower().startswith("fail"):
        out.append(("SPF check failed", "high", 25))
    return out


def find_keywords(text):
    low = text.lower()
    found = {}
    for cat, words in KEYWORDS.items():
        hits = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", low)]
        if hits:
            found[cat] = hits
    return found


def analyze(text):
    msg, body = split_message(text)
    subject = (msg.get("Subject") or "") if msg else ""
    content = (subject + "\n" + body).strip() or text

    links = [check_url(u) for u in dict.fromkeys(URL_RE.findall(content))]
    keywords = find_keywords(content)

    indicators, heur = [], 0
    if msg:
        for label, sev, pts in check_headers(msg):
            indicators.append({"severity": sev, "text": label, "where": "email headers"})
            heur += pts
    for label, sev, pts, dom in check_anchors(body):
        indicators.append({"severity": sev, "text": label, "where": dom})
        heur += pts
    for lk in links:
        for label, sev, pts in lk["flags"]:
            indicators.append({"severity": sev, "text": label, "where": lk["domain"]})
            heur += pts
    for cat, hits in keywords.items():
        pts = min(len(hits) * 6, 18)
        heur += pts
        indicators.append({"severity": "medium" if pts >= 12 else "low",
                           "text": f"{cat} language: {', '.join(hits[:4])}", "where": "message text"})
    heur = min(heur, 100)

    ml = float(MODEL.predict_proba([content])[0][1]) * 100
    w = 0.3 if len(content.split()) < 8 else 0.6  # a bare link carries little text for the model
    score = w * ml + (1 - w) * heur
    if heur >= 60:
        score = max(score, heur)
    score = round(min(score, 100))

    level = "Critical" if score >= 65 else "Warning" if score >= 35 else "Safe"
    order = {"high": 0, "medium": 1, "low": 2}
    indicators.sort(key=lambda i: order[i["severity"]])

    if level == "Safe":
        summary = "No strong phishing signals were found. Still, only act on messages you expected."
    else:
        top = indicators[0]["text"] if indicators else "the wording of the message"
        summary = (f"{len(indicators)} warning sign(s) found. The strongest: {top}. "
                   + ("Do not click links or reply." if level == "Critical" else "Check the sender and links before acting."))

    return {"score": score, "level": level, "ml_score": round(ml), "rule_score": heur,
            "summary": summary, "indicators": indicators, "links": links,
            "keywords": keywords, "has_headers": msg is not None}
