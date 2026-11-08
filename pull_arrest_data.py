"""
Pulls monthly marijuana possession (158) and sale/manufacturing (153)
arrest counts from the FBI Crime Data Explorer API for the 222 selected
Oklahoma/Texas/Utah counties (border + control sample), one agency per
county (county sheriff, or largest available city PD as fallback --
see build_arrest_county_agency_selection.py).

Confirmed live endpoint (NOT the '/arrest/agencies/...' plural guess that
404s): GET /arrest/agency/{ori}/{offense}?from=MM-YYYY&to=MM-YYYY&type=counts

Paced deliberately (2s between calls, exponential backoff on 503) because
this API showed real rate-limit sensitivity earlier in this session --
a short burst of ~10 rapid test calls triggered a multi-minute lockout on
every endpoint, including ones confirmed working moments before.

Writes incrementally to arrest_pull_raw.csv (append mode) so a crash or
manual interrupt preserves progress; on restart, already-pulled
(fips5, offense_code) pairs are skipped.
"""
import os
import time
import json
import urllib.request
import urllib.parse
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
KEY_PATH = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\marijuana_new\data\.fbi_api_key.txt"

API_BASE = "https://api.usa.gov/crime/fbi/cde"
OFFENSE_CODES = ["153", "158"]
FROM_DATE = "01-2010"
TO_DATE = "08-2026"
OUT_PATH = os.path.join(TBL, "arrest_pull_raw.csv")
LOG_PATH = os.path.join(TBL, "arrest_pull_log.txt")

with open(KEY_PATH) as f:
    API_KEY = f.read().strip()


def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get(path, params):
    params = dict(params)
    params["api_key"] = API_KEY
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)


def fetch_agency_offense(ori, offense):
    for attempt in range(5):
        status, body = get(f"/arrest/agency/{ori}/{offense}",
                            {"from": FROM_DATE, "to": TO_DATE, "type": "counts"})
        if status == 200:
            return json.loads(body)
        if status == 404:
            log(f"  404 (no data) ori={ori} offense={offense}")
            return None
        backoff = 20 * (attempt + 1)
        log(f"  status={status} ori={ori} offense={offense}, attempt {attempt+1}/5, backing off {backoff}s")
        time.sleep(backoff)
    log(f"  GAVE UP ori={ori} offense={offense}")
    return None


def extract_agency_series(payload, agency_name):
    actuals = payload.get("actuals", {})
    candidates = [k for k in actuals if k != "United States Arrests" and not k.endswith(", United States Arrests")]
    # prefer exact agency-name match; else take whichever key isn't the state aggregate
    key = None
    for k in candidates:
        if agency_name and agency_name in k:
            key = k
            break
    if key is None:
        # fall back: the state-level key is "{State Name} Arrests"; agency key is the other one
        state_like = [k for k in candidates if k.endswith(" Arrests") and len(k.split()) <= 3]
        non_state = [k for k in candidates if k not in state_like]
        key = non_state[0] if non_state else (candidates[0] if candidates else None)
    if key is None:
        return {}
    return actuals.get(key, {})


def main():
    sel = pd.read_csv(os.path.join(TBL, "arrest_county_agency_selection.csv"), dtype={"fips5": str})

    done = set()
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH, dtype={"fips5": str, "offense_code": str})
        done = set(zip(prev["fips5"], prev["offense_code"]))
        log(f"Resuming: {len(done)} (fips5, offense) pairs already pulled")

    write_header = not os.path.exists(OUT_PATH)
    total = len(sel) * len(OFFENSE_CODES)
    done_count = 0

    with open(OUT_PATH, "a", newline="", encoding="utf-8") as out_f:
        for _, row in sel.iterrows():
            for offense in OFFENSE_CODES:
                done_count += 1
                if (row["fips5"], offense) in done:
                    continue
                log(f"[{done_count}/{total}] {row['state_abbr']} {row['county_name']} "
                    f"ori={row['ori']} offense={offense}")
                payload = fetch_agency_offense(row["ori"], offense)
                if payload is None:
                    time.sleep(2)
                    continue
                series = extract_agency_series(payload, row.get("agency_name"))
                recs = pd.DataFrame([
                    {"fips5": row["fips5"], "state_abbr": row["state_abbr"], "county_name": row["county_name"],
                     "ori": row["ori"], "offense_code": offense, "month": m, "arrests": v}
                    for m, v in series.items()
                ])
                recs.to_csv(out_f, header=write_header, index=False, lineterminator="\n")
                write_header = False
                out_f.flush()
                time.sleep(2)

    log("DONE")


if __name__ == "__main__":
    main()
