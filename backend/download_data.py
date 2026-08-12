"""
download_data.py
------------------
Pulls REAL, public 10-K filings from SEC EDGAR (data.sec.gov) for a small
set of well-known companies and saves them as plain text into ../data/raw.
No API key required — SEC EDGAR is a free public API, but it does require
a descriptive User-Agent header (their policy, not a hack).

Run:  python download_data.py
"""

import os
import time
import requests

USER_AGENT = os.getenv("SEC_USER_AGENT", "student-project contact@example.com")
HEADERS = {"User-Agent": USER_AGENT}

# CIK (Central Index Key) numbers for a few large, well-known companies.
# Feel free to add more — full list: https://www.sec.gov/cgi-bin/browse-edgar
COMPANIES = {
    "Apple": "0000320193",
    "Microsoft": "0000789019",
    "Tesla": "0001318605",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_latest_10k_url(cik: str) -> str | None:
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(submissions_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    for form, accession, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "10-K":
            accession_nodash = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_nodash}/{doc}"
            )
    return None


def download_filing(company: str, cik: str):
    print(f"Fetching latest 10-K for {company} (CIK {cik})...")
    url = get_latest_10k_url(cik)
    if not url:
        print(f"  No 10-K found for {company}")
        return

    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    os.makedirs(OUT_DIR, exist_ok=True)
    ext = "html" if url.endswith((".htm", ".html")) else "txt"
    out_path = os.path.join(OUT_DIR, f"{company}_10K.{ext}")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    for name, cik in COMPANIES.items():
        try:
            download_filing(name, cik)
        except Exception as e:  # noqa: BLE001
            print(f"  Failed for {name}: {e}")
        time.sleep(0.5)  # be polite to SEC's rate limits
    print("Done. Now call POST /ingest (or use the Streamlit sidebar) to index these filings.")
