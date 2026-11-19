"""Fetch the public data for the islamic_malaysia_transmission run (standard library only).

Run from this folder:  <MaRCo>/.venv/Scripts/python.exe -I fetch_data.py
Every download is saved unchanged under data/raw/<source>/ and listed, with URL, UTC time, size and SHA-256,
in data/raw/manifest.csv. Nothing is written into the MaRCo folder.
"""
import csv
import datetime as dt
import hashlib
import pathlib
import subprocess
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
MANIFEST = RAW / "manifest.csv"

# Bank Negara Malaysia, Monthly Highlights & Statistics (public XLSX). "a" = series discontinued at the 2021-07 revision.
BNM = {
    "1.19.1a": "Islamic banking system: financing by purpose and sector (to 2021/22)",
    "1.19.1": "Islamic banking system: financing by purpose and sector (2021-07 on)",
    "1.19a": "Banking system: loans/financing by purpose (to 2021)",
    "1.19": "Banking system: loans/financing by purpose (2021-07 on)",
    "1.20a": "Banking system: loans/financing by sector (to 2021)",
    "1.20": "Banking system: loans/financing by sector (2021-07 on)",
    "1.24.2": "Islamic banking system: deposits by type and holder",
    "1.24": "Banking system: total deposits by type",
    "1.25": "Banking system: total deposits by holder",
    "1.32": "Islamic banking system: investment account by type and holder",
    "2.1": "Interest rates: banking institutions (conventional)",
    "2.2": "Islamic banking system: financing rate, profit rate to depositors, return to investment account holders",
    "1.33": "Financial institutions: SME loans/financing by sector",
    "1.33.1": "SME loans/financing applied by sector",
    "1.33.2": "SME loans/financing approved by sector",
    "1.33.3": "SME loans/financing disbursed by sector",
    "1.34": "SME loans/financing by purpose",
    "2.3": "Interest rates: interbank money market, incl. overnight policy rate (OPR)",
}
BNM_URL = "https://www.bnm.gov.my/documents/20124/23499006/{t}.xlsx"


def _get(url, timeout, client):
    if client == "curl":
        # BNM refuses Python's urllib (HTTP 403) but serves curl; curl is used with its own default identity,
        # no browser headers or user-agent spoofing.
        r = subprocess.run(["curl", "-s", "-f", "-L", "--max-time", str(timeout), url], capture_output=True)
        if r.returncode:
            raise RuntimeError(f"curl exit {r.returncode}")
        return r.stdout
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch(url, dest, timeout=120, retries=3, client="urllib"):
    """Download url to dest unchanged and record it in the manifest. Returns the bytes or None."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(retries):
        try:
            body = _get(url, timeout, client)
            break
        except Exception as e:  # network errors are reported, not hidden
            err = e
            time.sleep(2 * (attempt + 1))
    else:
        print(f"FAILED {url}: {err}")
        return None
    dest.write_bytes(body)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "url", "fetched_utc", "bytes", "sha256", "client"])
        w.writerow([dest.relative_to(RAW).as_posix(), url,
                    dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    len(body), hashlib.sha256(body).hexdigest(), client])
    return body


def main():
    for t in BNM:
        fetch(BNM_URL.format(t=t), RAW / "bnm" / f"{t}.xlsx", client="curl")
    # Malaysia policy rate (OPR) through BIS central bank policy rates on DBnomics (no key)
    fetch("https://api.db.nomics.world/v22/series/BIS/WS_CBPOL/M.MY?observations=1&format=json",
          RAW / "bis_cbpol" / "MY.json")
    # Jarocinski-Karadi US monetary policy surprises (authors' GitHub update); VIX from FRED
    fetch("https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/main/shocks_fed_jk_m.csv",
          RAW / "jk" / "shocks_fed_jk_m.csv")
    fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", RAW / "fred" / "VIXCLS.csv")
    print("done; manifest at", MANIFEST)


if __name__ == "__main__":
    main()
