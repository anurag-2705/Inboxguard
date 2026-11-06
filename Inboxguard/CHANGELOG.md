# Changelog

## Unreleased

### Security / reliability
- **Fixed a denial-of-service:** `preprocess._EMAIL` was quadratic on long runs of `[\w.+-]` without an `@`; a 1 MB `.eml` could pin the only worker for minutes. The regex is now anchored and bounded, `BARE_RE` is bounded, and each request scans at most 50,000 characters.
- **Fixed a crash:** URLs with unbalanced brackets (`http://[::1`) raised `ValueError` and returned HTTP 500. URL parsing is now safe, and IPv6 literals are recognised.

### Detection
- `is_official` no longer accepts `brand.com/.net/.org` implicitly: a brand's generic domains must be listed in `OFFICIAL`; only country sites (amazon.in, google.co.uk) are inferred. `paypal.net` / `hdfc.net` are now flagged as brand TLD-swaps (high severity).
- "Excessive subdomains" no longer counts the public suffix (`www.sbi.co.in` was wrongly flagged).
- Hex, decimal, octal and IPv6 hosts are treated as raw IPs; bare `1.2.3.4/path` links are extracted.
- Any high-severity finding raises the verdict to at least Warning (a confirmed look-alike domain previously scored 43 = "Low risk").

### Project
- Added `tests/` (pytest), `.github/workflows/ci.yml`, `templates/index.html` (missing from the repository: the Dockerfile `COPY templates` and the `/` route need it) and this changelog.

### Note
`metrics.json` and the system-level numbers in the README were produced before these rule changes. Re-run `python evaluate.py` to refresh them.
