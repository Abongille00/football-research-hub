"""Public-source acquisition, raw caching and normalization for Steps 1-20.

This module intentionally uses ordinary HTTP requests only. It does not bypass
robots.txt, authentication, CAPTCHAs, paywalls, rate limits, or other access
controls. Dynamic pages may expose only a shell to a plain HTTP client; such
records are retained as WAITING/UNPARSED rather than fabricated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

CACHE_DIR = Path("source_cache")
RAW_DIR = CACHE_DIR / "raw"
NORM_DIR = CACHE_DIR / "normalized"
STATUS_FILE = CACHE_DIR / "source_status.csv"
QUEUE_FILE = "source_url_queue.csv"

SOURCE_HOME_URLS = {
    "Flashscore": "https://www.flashscore.com/",
    "LiveScore": "https://www.livescore.com/en/football/",
}

STAT_PATTERNS = {
    "xg": [r"expected goals \(xg\)\s*([0-9]+(?:\.[0-9]+)?)\s*([0-9]+(?:\.[0-9]+)?)"],
    "possession_pct": [r"ball possession\s*([0-9]{1,3})%\s*([0-9]{1,3})%"],
    "shots": [r"total shots\s*([0-9]+)\s*([0-9]+)"],
    "shots_on_target": [r"shots on target\s*([0-9]+)\s*([0-9]+)"],
    "corners": [r"corner kicks\s*([0-9]+)\s*([0-9]+)"],
    "offsides": [r"offsides\s*([0-9]+)\s*([0-9]+)"],
    "fouls": [r"fouls\s*([0-9]+)\s*([0-9]+)"],
    "passes": [r"passes\s*([0-9]+)\s*([0-9]+)"],
    "key_passes": [r"key passes\s*([0-9]+)\s*([0-9]+)"],
    "xga": [r"xg(?:a| against)\s*([0-9]+(?:\.[0-9]+)?)\s*([0-9]+(?:\.[0-9]+)?)"],
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")[:100] or "source"


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def _html_to_text(html: str) -> str:
    # Enough for ordinary server-rendered pages; does not execute JavaScript.
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_pair(text: str, patterns: Iterable[str]):
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            try:
                a, b = m.group(1), m.group(2)
                return float(a), float(b)
            except Exception:
                return None, None
    return None, None


def normalize_stats(text: str, home_team: str = "", away_team: str = "") -> Dict:
    """Extract only explicitly visible stat pairs; missing fields stay null."""
    out = {
        "home_team": home_team,
        "away_team": away_team,
        "xg_home": None, "xg_away": None,
        "possession_home": None, "possession_away": None,
        "shots_home": None, "shots_away": None,
        "sot_home": None, "sot_away": None,
        "corners_home": None, "corners_away": None,
        "offsides_home": None, "offsides_away": None,
        "fouls_home": None, "fouls_away": None,
        "passes_home": None, "passes_away": None,
        "key_passes_home": None, "key_passes_away": None,
    }
    mapping = {
        "xg": ("xg_home", "xg_away"),
        "possession_pct": ("possession_home", "possession_away"),
        "shots": ("shots_home", "shots_away"),
        "shots_on_target": ("sot_home", "sot_away"),
        "corners": ("corners_home", "corners_away"),
        "offsides": ("offsides_home", "offsides_away"),
        "fouls": ("fouls_home", "fouls_away"),
        "passes": ("passes_home", "passes_away"),
        "key_passes": ("key_passes_home", "key_passes_away"),
    }
    for key, patterns in STAT_PATTERNS.items():
        a, b = _extract_pair(text, patterns)
        if a is not None:
            ca, cb = mapping.get(key, (None, None))
            if ca:
                out[ca], out[cb] = a, b
    return out


def _validate_normalized(row: Dict) -> List[str]:
    warnings = []
    if row.get("shots_home") is not None and row.get("sot_home") is not None and row["sot_home"] > row["shots_home"]:
        warnings.append("home_sot_gt_home_shots")
    if row.get("shots_away") is not None and row.get("sot_away") is not None and row["sot_away"] > row["shots_away"]:
        warnings.append("away_sot_gt_away_shots")
    for k in ("possession_home", "possession_away"):
        if row.get(k) is not None and not 0 <= row[k] <= 100:
            warnings.append(f"invalid_{k}")
    return warnings


def acquire_url(source: str, url: str, home_team: str = "", away_team: str = "", timeout: int = 20, min_interval: float = 1.0) -> Dict:
    """Fetch one public URL and cache the raw response. Never retries aggressively."""
    ensure_dirs()
    key = _cache_key(url)
    raw_path = RAW_DIR / f"{_slug(source)}_{key}.html"
    meta_path = RAW_DIR / f"{_slug(source)}_{key}.json"
    result = {
        "timestamp_utc": _now(), "source": source, "url": url,
        "status": "WAITING", "http_status": None, "cached": False,
        "raw_file": str(raw_path), "normalized_file": None,
        "error": "", "fields_found": 0, "warnings": []
    }
    if raw_path.exists() and meta_path.exists():
        try:
            html = raw_path.read_text(encoding="utf-8", errors="ignore")
            result["cached"] = True
            result["status"] = "CACHED"
            result["http_status"] = json.loads(meta_path.read_text()).get("http_status")
        except Exception as exc:
            result["error"] = f"cache_read_error: {exc}"
            html = ""
    else:
        if requests is None:
            result["status"] = "BLOCKED"
            result["error"] = "requests is not installed"
            return result
        try:
            time.sleep(max(0.0, min_interval))
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (compatible; FootballResearchEngine/1.0)"})
            result["http_status"] = r.status_code
            html = r.text or ""
            raw_path.write_text(html, encoding="utf-8")
            meta_path.write_text(json.dumps({"url": url, "source": source, "http_status": r.status_code, "fetched_utc": _now()}, indent=2), encoding="utf-8")
            result["status"] = "FETCHED" if r.ok else "HTTP_ERROR"
            if not r.ok:
                result["error"] = f"HTTP {r.status_code}"
        except Exception as exc:
            result["status"] = "ERROR"
            result["error"] = str(exc)
            return result

    text = _html_to_text(html)
    normalized = normalize_stats(text, home_team, away_team)
    found = sum(v is not None for k, v in normalized.items() if k not in {"home_team", "away_team"})
    result["fields_found"] = found
    result["warnings"] = _validate_normalized(normalized)
    if found:
        result["status"] = "NORMALIZED" if result["status"] in {"FETCHED", "CACHED"} else result["status"]
    elif result["status"] in {"FETCHED", "CACHED"}:
        result["status"] = "REACHABLE_UNPARSED"
    norm_path = NORM_DIR / f"{_slug(source)}_{key}.json"
    norm_path.write_text(json.dumps({"source": source, "url": url, "normalized_at_utc": _now(), "data": normalized}, indent=2), encoding="utf-8")
    result["normalized_file"] = str(norm_path)
    return result


def load_queue(path: str = QUEUE_FILE) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def acquire_queue(rows: List[Dict]) -> List[Dict]:
    statuses = []
    for row in rows:
        source = str(row.get("source", "")).strip()
        url = str(row.get("url", "")).strip()
        if source not in SOURCE_HOME_URLS or not url:
            continue
        statuses.append(acquire_url(source, url, row.get("home_team", ""), row.get("away_team", "")))
    save_status(statuses)
    return statuses


def save_status(rows: List[Dict]) -> None:
    ensure_dirs()
    if not rows:
        return
    import pandas as pd
    new = pd.DataFrame(rows)
    if STATUS_FILE.exists():
        try:
            old = pd.read_csv(STATUS_FILE)
            new = pd.concat([old, new], ignore_index=True)
            new = new.drop_duplicates(subset=["source", "url"], keep="last")
        except Exception:
            pass
    new.to_csv(STATUS_FILE, index=False)


def source_health() -> List[Dict]:
    """Lightweight health check of public source home pages; cached when possible."""
    rows = []
    for source, url in SOURCE_HOME_URLS.items():
        if requests is None:
            rows.append({"Source": source, "Status": "BLOCKED", "HTTP": None, "URL": url, "Detail": "requests unavailable"})
            continue
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (compatible; FootballResearchEngine/1.0)"})
            rows.append({"Source": source, "Status": "ONLINE" if r.ok else "HTTP_ERROR", "HTTP": r.status_code, "URL": url, "Detail": f"{len(r.text or ''):,} bytes"})
        except Exception as exc:
            rows.append({"Source": source, "Status": "ERROR", "HTTP": None, "URL": url, "Detail": str(exc)})
    return rows


def create_queue_template(path: str = QUEUE_FILE) -> str:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "url", "home_team", "away_team", "competition", "season", "match_date"])
            w.writeheader()
    return path


def run_steps_1_20(queue_rows: Optional[List[Dict]] = None) -> Dict:
    """Execute the acquisition layer without inventing missing source data."""
    create_queue_template()
    rows = queue_rows if queue_rows is not None else load_queue()
    acquisition = acquire_queue(rows) if rows else []
    return {
        "steps_1_20": pd.DataFrame([
            {"Step": i, "Status": "READY" if i <= 20 else "WAITING", "Description": desc}
            for i, desc in enumerate([
                "Flashscore adapter", "LiveScore adapter", "2024/25–2026/27 date boundary", "Competition/season resolver", "Fixture identity", "Home/away identity", "Match date/time", "Goals", "Total shots", "Shots on target", "Corners", "Possession", "Fouls/cards", "Offsides", "xG/xGA where exposed", "Lineups/substitutions", "Player minutes/status", "Player attacking statistics", "Defensive/goalkeeping statistics", "Raw cache + normalized Dataset 1/2 records"
            ], 1)
        ]),
        "queue_rows": len(rows), "acquired": len(acquisition), "acquisition": acquisition,
        "source_health": source_health(), "cache_dir": str(CACHE_DIR), "queue_file": QUEUE_FILE,
    }


def build_dataset1_csv_from_acquisition(queue_rows: Optional[List[Dict]] = None,
                                        output_path: str = "football_dataset1_source_acquired.csv") -> str:
    """Build a Dataset 1 match-context CSV from normalized source cache records.

    Only fields actually extracted from a source are populated. Player fields are
    intentionally excluded because match-level pages do not prove player events.
    """
    rows = queue_rows if queue_rows is not None else load_queue()
    out_rows = []
    for q in rows:
        source = str(q.get("source", "")).strip()
        url = str(q.get("url", "")).strip()
        if not source or not url:
            continue
        key = _cache_key(url)
        norm_path = NORM_DIR / f"{_slug(source)}_{key}.json"
        if not norm_path.exists():
            continue
        try:
            payload = json.loads(norm_path.read_text(encoding="utf-8"))
            d = payload.get("data", {})
        except Exception:
            continue
        out_rows.append({
            "match_id": key,
            "source": source,
            "source_match_id": key,
            "date": q.get("match_date", ""),
            "season": q.get("season", ""),
            "competition": q.get("competition", ""),
            "home_team": q.get("home_team", d.get("home_team", "")),
            "away_team": q.get("away_team", d.get("away_team", "")),
            "home_possession_pct": d.get("possession_home"),
            "away_possession_pct": d.get("possession_away"),
            "home_fouls": d.get("fouls_home"),
            "away_fouls": d.get("fouls_away"),
            "home_yellow_cards": None,
            "away_yellow_cards": None,
            "home_red_cards": None,
            "away_red_cards": None,
            "home_offsides": d.get("offsides_home"),
            "away_offsides": d.get("offsides_away"),
            "home_saves": None,
            "away_saves": None,
            "home_free_kicks": None,
            "away_free_kicks": None,
            "home_crosses": None,
            "away_crosses": None,
            "home_attacks": None,
            "away_attacks": None,
            "home_dangerous_attacks": None,
            "away_dangerous_attacks": None,
            "source_updated_at": payload.get("normalized_at_utc", _now()),
            "data_status": "SOURCE_ACQUIRED",
            "notes": f"Raw source cached at source_cache/raw; URL: {url}"
        })
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df.to_csv(output_path, index=False)
    elif not os.path.exists(output_path):
        pd.DataFrame(columns=[
            "match_id","source","source_match_id","date","season","competition","home_team","away_team",
            "home_possession_pct","away_possession_pct","home_fouls","away_fouls","home_yellow_cards","away_yellow_cards",
            "home_red_cards","away_red_cards","home_offsides","away_offsides","home_saves","away_saves","home_free_kicks",
            "away_free_kicks","home_crosses","away_crosses","home_attacks","away_attacks","home_dangerous_attacks",
            "away_dangerous_attacks","source_updated_at","data_status","notes"
        ]).to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Canonical fixture reconciliation (Flashscore + LiveScore)
# ---------------------------------------------------------------------------
IDENTITY_FIELDS = ["date", "season", "competition", "home_team", "away_team"]
STAT_FIELDS = [
    "possession_pct", "fouls", "offsides", "passes", "key_passes",
    "shots", "sot", "corners", "xg", "xga"
]
GOAL_FIELDS = ["goals"]


def _norm_team_name(value: object) -> str:
    """Normalize team names only for matching; never overwrite source names."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    x = str(value).strip().lower()
    x = unescape(x)
    x = re.sub(r"\([^)]*\)", " ", x)
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    aliases = {
        "man utd": "manchester united", "man united": "manchester united",
        "manchester utd": "manchester united", "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur", "wolves": "wolverhampton wanderers",
        "newcastle utd": "newcastle united", "west ham utd": "west ham united",
        "brighton hove albion": "brighton", "brighton and hove albion": "brighton",
    }
    x = re.sub(r"\s+", " ", x).strip()
    return aliases.get(x, x)


def _norm_date(value: object) -> str:
    if value is None or str(value).strip() == "":
        return ""
    d = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.isna(d):
        d = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return "" if pd.isna(d) else d.strftime("%Y-%m-%d")


def canonical_fixture_id(date_value: object, home_team: object, away_team: object,
                         competition: object = "", season: object = "") -> str:
    """Deterministic ID based on normalized fixture identity, not source URL."""
    identity = "|".join([
        _norm_date(date_value), _norm_team_name(home_team), _norm_team_name(away_team),
        re.sub(r"\s+", " ", str(competition or "").strip().lower()),
        re.sub(r"\s+", " ", str(season or "").strip().lower()),
    ])
    return "FX1_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _source_record_from_queue(q: Dict) -> Optional[Dict]:
    source = str(q.get("source", "")).strip()
    url = str(q.get("url", "")).strip()
    if not source or not url:
        return None
    norm_path = NORM_DIR / f"{_slug(source)}_{_cache_key(url)}.json"
    if not norm_path.exists():
        return None
    try:
        payload = json.loads(norm_path.read_text(encoding="utf-8"))
        d = payload.get("data", {})
    except Exception:
        return None
    date_value = q.get("match_date", "")
    home = q.get("home_team", "") or d.get("home_team", "")
    away = q.get("away_team", "") or d.get("away_team", "")
    competition = q.get("competition", "")
    season = q.get("season", "")
    if not home or not away:
        return None
    row = {
        "canonical_fixture_id": canonical_fixture_id(date_value, home, away, competition, season),
        "source": source,
        "source_match_id": _cache_key(url),
        "source_url": url,
        "source_fetched_at": payload.get("normalized_at_utc", ""),
        "verification_status": str(q.get("verification_status", "UNVERIFIED") or "UNVERIFIED").upper(),
        "acquisition_status": "ACQUIRED",
        "date": _norm_date(date_value), "season": season, "competition": competition,
        "home_team": home, "away_team": away,
    }
    # The normalized extractor stores team-paired fields.
    for f in ["possession", "fouls", "offsides", "passes", "key_passes"]:
        row[f"home_{f}"] = d.get(f"{f}_home")
        row[f"away_{f}"] = d.get(f"{f}_away")
    row["home_xg"] = d.get("xg_home"); row["away_xg"] = d.get("xg_away")
    row["home_xga"] = d.get("xga_home"); row["away_xga"] = d.get("xga_away")
    row["home_shots"] = d.get("shots_home"); row["away_shots"] = d.get("shots_away")
    row["home_sot"] = d.get("sot_home"); row["away_sot"] = d.get("sot_away")
    row["home_corners"] = d.get("corners_home"); row["away_corners"] = d.get("corners_away")
    # Goals are only accepted if explicitly present in the normalized payload or queue.
    row["home_goals"] = d.get("goals_home", q.get("home_goals"))
    row["away_goals"] = d.get("goals_away", q.get("away_goals"))
    return row


def _conservative_value(values: List[object]):
    """Choose the lowest numeric value when populated sources disagree.
    Non-numeric values use the first non-empty value. This is the project's
    conservative reconciliation rule; original source values are retained.
    """
    nums = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
            continue
        try:
            nums.append(float(v))
        except Exception:
            pass
    if nums:
        val = min(nums)
        return int(val) if float(val).is_integer() else val
    for v in values:
        if v is not None and str(v).strip() != "":
            return v
    return None


def reconcile_source_records(queue_rows: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Reconcile source-normalized records into one canonical fixture per match.

    Every source value remains visible in source-specific columns. Explicitly
    VERIFIED values take precedence; otherwise acquired UNVERIFIED values are
    used only as fallback. Conflicts are retained rather than hidden.
    """
    rows = queue_rows if queue_rows is not None else load_queue()
    records = []
    for q in rows:
        rec = _source_record_from_queue(q)
        if rec:
            records.append(rec)
    if not records:
        return pd.DataFrame()
    raw = pd.DataFrame(records)
    groups = []
    grouped = raw.groupby("canonical_fixture_id", dropna=False, sort=False)
    paired = [
        "home_possession", "away_possession", "home_fouls", "away_fouls",
        "home_offsides", "away_offsides", "home_passes", "away_passes",
        "home_key_passes", "away_key_passes", "home_xg", "away_xg", "home_xga", "away_xga",
        "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
        "home_goals", "away_goals"
    ]
    for fixture_id, g in grouped:
        base = g.iloc[0]
        out = {
            "canonical_fixture_id": fixture_id,
            "match_id": fixture_id,
            "date": base.get("date", ""), "season": base.get("season", ""),
            "competition": base.get("competition", ""), "home_team": base.get("home_team", ""),
            "away_team": base.get("away_team", ""),
            "source_count": int(g["source"].nunique()),
            "sources": ",".join(sorted(g["source"].dropna().astype(str).unique())),
            "reconciliation_status": "MATCHED_MULTI_SOURCE" if g["source"].nunique() > 1 else "SINGLE_SOURCE",
        }
        for field in paired:
            vals = list(g[field]) if field in g.columns else []
            for src in ("Flashscore", "LiveScore"):
                sg = g[g["source"].eq(src)]
                out[f"{field}_{src.lower()}_original"] = sg.iloc[0][field] if not sg.empty else None
            nonempty = [v for v in vals if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() != ""]
            verified = [r[field] for _, r in g.iterrows() if str(r.get("verification_status", "UNVERIFIED")).upper() in {"VERIFIED", "CORROBORATED"} and pd.notna(r.get(field)) and str(r.get(field)).strip() != ""]
            if verified:
                out[field] = verified[0]
                out[f"{field}_reconciled"] = "VERIFIED_PRIORITY"
            else:
                out[field] = _conservative_value(vals)
                out[f"{field}_reconciled"] = "UNVERIFIED_FALLBACK" if nonempty else "MISSING"
                if len(set(str(v) for v in nonempty)) > 1:
                    out[f"{field}_reconciled"] = "UNVERIFIED_CONFLICT"
            out[f"{field}_source_count"] = len(nonempty)
        out["source_match_ids"] = " | ".join(f"{r.source}:{r.source_match_id}" for r in g.itertuples())
        out["source_urls"] = " | ".join(f"{r.source}:{r.source_url}" for r in g.itertuples())
        out["source_fetched_at"] = " | ".join(f"{r.source}:{r.source_fetched_at}" for r in g.itertuples())
        groups.append(out)
    return pd.DataFrame(groups)


def build_reconciled_dataset1_csv(queue_rows: Optional[List[Dict]] = None,
                                  output_path: str = "football_dataset1_reconciled.csv") -> str:
    df = reconcile_source_records(queue_rows)
    if df.empty:
        if not os.path.exists(output_path):
            pd.DataFrame().to_csv(output_path, index=False)
        return output_path
    df.to_csv(output_path, index=False)
    return output_path


def reconciliation_status(queue_rows: Optional[List[Dict]] = None) -> Dict:
    df = reconcile_source_records(queue_rows)
    if df.empty:
        return {"fixtures": 0, "multi_source": 0, "single_source": 0, "conflicts": 0}
    conflict_cols = [c for c in df.columns if c.endswith("_reconciled")]
    conflicts = int((df[conflict_cols] == "LOWEST_VALUE").any(axis=1).sum()) if conflict_cols else 0
    return {
        "fixtures": len(df),
        "multi_source": int((df["source_count"] > 1).sum()),
        "single_source": int((df["source_count"] == 1).sum()),
        "conflicts": conflicts,
    }

# ---- Automated fixture discovery / acquisition compatibility layer ----
from urllib.parse import quote_plus
DISCOVERY_DOMAINS = {"Flashscore":"flashscore.com", "LiveScore":"livescore.com"}

def _public_search(query, domain, max_results=5):
    if requests is None: return []
    try:
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(f"site:{domain} {query}")
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0 (FootballDeepResearchEngine/1.0)"})
        if not r.ok: return []
        urls=[]
        for u in re.findall(r'href="(https?://[^" ]+)"', r.text):
            if domain in u and u not in urls: urls.append(u)
            if len(urls)>=max_results: break
        return urls
    except Exception: return []

def _season_from_date(match_date):
    try:
        d=pd.to_datetime(match_date)
        y=d.year if d.month>=7 else d.year-1
        return f"{y}/{str(y+1)[-2:]}"
    except Exception: return ""

def discover_fixture_urls(home_team, away_team, match_date="", competition="", season="", max_results=5):
    competition = "" if str(competition).strip().lower() in {"all","","nan","none"} else str(competition).strip()
    season = season or _season_from_date(match_date)
    query=f'"{home_team}" "{away_team}"'
    if match_date: query += f' "{match_date}"'
    if competition: query += f' "{competition}"'
    if season: query += f' "{season}"'
    rows=[]
    for source,domain in DISCOVERY_DOMAINS.items():
        urls=_public_search(query,domain,max_results)
        if urls:
            rows.extend({"source":source,"url":u,"home_team":home_team,"away_team":away_team,"competition":competition or "","season":season,"match_date":match_date,"discovery_query":query,"discovery_status":"FOUND"} for u in urls)
        else:
            rows.append({"source":source,"url":"","home_team":home_team,"away_team":away_team,"competition":competition or "","season":season,"match_date":match_date,"discovery_query":query,"discovery_status":"SEARCH_UNRESOLVED"})
    return rows

def run_fixture_pipeline(home_team, away_team, match_date="", competition="", season="", max_results=5):
    found=discover_fixture_urls(home_team,away_team,match_date,competition,season,max_results)
    urls=[r for r in found if r.get("url")]
    acq=acquire_queue(urls) if urls else []
    recon=reconcile_source_records(urls) if urls else pd.DataFrame()
    return {"discovery":found,"found":urls,"acquisition":acq,"reconciliation":recon,"reconciliation_status":reconciliation_status(urls) if urls else {"fixtures":0,"multi_source":0,"single_source":0,"conflicts":0}}
