"""Unit tests for the scoring engine. Run: python -m pytest"""
import time

import pytest

import analyzer as a


def labels(result):
    return [i["text"] for i in result["indicators"]]


# ---- domains ---------------------------------------------------------------------------------
@pytest.mark.parametrize("host,expected", [
    ("www.paypal.co.uk", "paypal.co.uk"),
    ("mail.google.com", "google.com"),
    ("a.b.example.com.au", "example.com.au"),
])
def test_registered_domain_uses_public_suffix_list(host, expected):
    assert a.registered_domain(host) == expected


@pytest.mark.parametrize("brand,reg", [
    ("paypal", "paypal.com"), ("amazon", "amazon.in"), ("amazon", "amazon.co.uk"),
    ("google", "google.de"), ("sbi", "sbi.co.in"),
])
def test_official_domains_are_accepted(brand, reg):
    assert a.is_official(brand, reg)


@pytest.mark.parametrize("brand,reg", [
    ("paypal", "paypal.net"), ("paypal", "paypal.org"), ("hdfc", "hdfc.net"),
    ("amazon", "amazon.xyz"), ("paypal", "paypal-secure.com"),
])
def test_tld_swaps_are_not_official(brand, reg):
    assert not a.is_official(brand, reg)


@pytest.mark.parametrize("url", [
    "http://paypa1-secure.com/verify",      # digit look-alike
    "http://rnicrosoft.com/login",          # rn -> m
    "https://paypall.com/signin",           # typo
    "https://paypal.com.account-verify.top/x",
    "https://amazon.com.evil.ru/login",
    "http://paypal.net/login",              # TLD swap
])
def test_brand_impersonation_is_at_least_warning(url):
    r = a.analyze(url)
    assert r["level"] in {"Warning", "Critical"}, (url, r["score"], labels(r))


@pytest.mark.parametrize("url", [
    "https://www.paypal.com/signin", "https://www.amazon.in/gp/your-account/order-details",
    "https://github.com/anurag-2705/Inboxguard", "https://drive.google.com/file/d/abc",
    "https://www.sbi.co.in/", "https://www.paypal.co.uk/signin",
])
def test_legitimate_links_have_no_flags(url):
    r = a.analyze(url)
    assert r["indicators"] == [], labels(r)
    assert r["level"] == "Low risk"


def test_public_suffix_is_not_counted_as_subdomain():
    assert "Excessive subdomains" not in labels(a.analyze("https://www.sbi.co.in/"))
    assert "Excessive subdomains" in labels(a.analyze("https://a.b.evil-example.com/"))


@pytest.mark.parametrize("url", [
    "http://192.168.4.5/login", "http://0x7f000001/login", "http://3232235777/login",
    "http://0300.0250.0001.0001/login", "http://[::1]/login", "185.23.4.5/verify",
])
def test_raw_and_obfuscated_ip_hosts(url):
    r = a.analyze(url)
    assert any("raw IP" in t for t in labels(r)), (url, labels(r))
    assert r["level"] != "Low risk"


def test_userinfo_trick_and_punycode():
    assert any("'@'" in t for t in labels(a.analyze("http://paypal.com@evil.com/")))
    assert any("Punycode" in t for t in labels(a.analyze("https://xn--pypal-4ve.com")))


def test_hidden_redirect_target_is_inspected():
    r = a.analyze("https://track.example.com/c?url=http%3A%2F%2Fpaypa1.xyz%2Flogin")
    assert any(t.startswith("Hidden redirect target") for t in labels(r))


def test_defanged_links_are_found():
    r = a.analyze("visit hxxp://paypa1-secure[.]com/login now")
    assert any("Impersonates" in t for t in labels(r))


def test_shortener_flagged():
    assert any("shortener" in t for t in labels(a.analyze("see https://bit.ly/3abc")))


# ---- anchors / headers / attachments -----------------------------------------------------------
def test_anchor_text_mismatch():
    body = '<a href="http://evil-site.top/x">https://www.paypal.com/login</a>'
    assert any("Link text shows" in t for t in labels(a.analyze(body)))


def test_spoofed_sender_and_failed_auth():
    mail = ("From: PayPal Support <help@secure-mail.xyz>\nReply-To: scam@gmail.com\n"
            "Subject: Action needed\nAuthentication-Results: mx; spf=fail; dkim=fail; dmarc=fail\n\n"
            "Please verify your account.")
    r = a.analyze(mail)
    assert r["has_headers"] and r["level"] == "Critical"
    assert any("SPF check failed" in t for t in labels(r))
    assert any("Sender name says 'paypal'" in t for t in labels(r))


def test_clean_headers_not_flagged():
    mail = ("From: GitHub <noreply@github.com>\nSubject: Welcome\n"
            "Authentication-Results: mx; spf=pass; dkim=pass; dmarc=pass\n\nWelcome to GitHub.")
    r = a.analyze(mail)
    assert r["level"] == "Low risk", labels(r)


def test_eml_attachments():
    eml = (b"From: a@example.com\nSubject: invoice\nMIME-Version: 1.0\n"
           b'Content-Type: multipart/mixed; boundary="b"\n\n'
           b"--b\nContent-Type: text/plain\n\nSee attached.\n"
           b'--b\nContent-Type: application/octet-stream\nContent-Disposition: attachment; filename="invoice.pdf.exe"\n\nMZ\n--b--\n')
    r = a.analyze_eml(eml)
    assert "invoice.pdf.exe" in r["attachments"]
    assert any("hides an executable" in t for t in labels(r))
    assert r["level"] != "Low risk"


# ---- scoring -----------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Hi team, attaching the Q3 report. Let me know if you have questions. Thanks, Priya",
    "Meeting today at 3pm to discuss the budget. Urgent items first.",
    "Your order has shipped! Track it at https://www.amazon.in/gp/your-account/order-details",
])
def test_ordinary_mail_is_low_risk(text):
    assert a.analyze(text)["level"] == "Low risk"


def test_keywords_alone_cannot_create_a_verdict():
    r = a.analyze("urgent immediately act now final notice verify your password login otp "
                  "unauthorized suspicious locked prize winner")
    assert r["rule_score"] <= a.KEYWORD_CAP


def test_level_thresholds():
    assert a.level_for(a.WARN_AT - 1) == "Low risk"
    assert a.level_for(a.WARN_AT) == "Warning"
    assert a.level_for(a.CRITICAL_AT) == "Critical"


def test_combine_without_model_uses_rules_only():
    assert a.combine(None, 37, 50) == 37


def test_committed_model_is_not_a_demo():
    bundle = a.get_model()
    assert bundle is not None and not bundle.get("demo")


def test_result_shape():
    r = a.analyze("hello")
    for key in ("score", "level", "ml_score", "rule_score", "summary", "indicators", "links", "keywords"):
        assert key in r
    assert 0 <= r["score"] <= 100


# ---- performance regressions (hostile input must not pin a worker) ------------------------------
@pytest.mark.parametrize("chunk", ["a.b.", "<script ", "a-", "1."])
def test_hostile_one_megabyte_upload_is_fast(chunk):
    data = b"From: a@b.com\nSubject: hi\nContent-Type: text/html\n\n" + (chunk * (1_000_000 // len(chunk))).encode()
    a.get_model()
    t = time.time()
    a.analyze_eml(data)
    assert time.time() - t < 8, f"{chunk!r} took {time.time() - t:.1f}s"


# ---- robustness: malformed input must never crash the analysis ------------------------------------
@pytest.mark.parametrize("text", [
    "http://[::1", "http://[evil.com/x", "http://]/x", "http://[", "www.[a].com", "http://a b",
    "\x00\x01 http://x.com", "http://" + "a" * 5000 + ".com", "<a href='http://[::'>x</a>", "💥" * 100,
])
def test_malformed_input_does_not_crash(text):
    r = a.analyze(text)
    assert 0 <= r["score"] <= 100
