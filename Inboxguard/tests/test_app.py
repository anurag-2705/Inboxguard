"""HTTP-level tests. Run: python -m pytest"""
import io

import pytest

import app as app_module


@pytest.fixture()
def client():
    app_module.limiter = app_module.RateLimiter(10_000)      # isolate tests from the rate limit
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}


def test_dashboard_renders_with_nonce_csp(client):
    r = client.get("/")
    assert r.status_code == 200
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'nonce-" in csp and "default-src 'none'" in csp
    nonce = csp.split("script-src 'nonce-")[1].split("'")[0]
    assert nonce in r.get_data(as_text=True)


def test_security_headers(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"


def test_analyze_text(client):
    r = client.post("/analyze", json={"text": "http://paypa1-secure.com/verify"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["level"] in {"Warning", "Critical"}
    assert r.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("payload,code", [
    ({"text": ""}, 400), ({"text": "   "}, 400), ({"text": 123}, 400), ({"nope": 1}, 400),
    ({"text": "x" * (app_module.MAX_CHARS + 1)}, 413),
])
def test_analyze_validation(client, payload, code):
    assert client.post("/analyze", json=payload).status_code == code


def test_analyze_rejects_non_json(client):
    assert client.post("/analyze", data="hello", content_type="text/plain").status_code == 400


def test_eml_upload(client):
    eml = b"From: PayPal <x@evil.top>\nSubject: hi\n\nverify your password at http://paypa1.xyz/login"
    r = client.post("/analyze", data={"file": (io.BytesIO(eml), "m.eml")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["has_headers"]


def test_empty_upload(client):
    r = client.post("/analyze", data={"file": (io.BytesIO(b"  "), "m.eml")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_rate_limit(client):
    app_module.limiter = app_module.RateLimiter(2)
    codes = [client.post("/analyze", json={"text": "hello there"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_unknown_route_is_json(client):
    r = client.get("/nope")
    assert r.status_code == 404 and "error" in r.get_json()


def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200 and "ml_active" in r.get_json()
