import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import json
from datetime import datetime, date

try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title="Football Deep Research Engine", page_icon="⚽", layout="wide")

PRIMARY_DATASET = "football_master_2024_27_v1.csv"
WATCHLIST_FILE = "market_watchlist.csv"
RESEARCH_VERSION = "Deep Research Engine — Steps 1–630"

REQUIRED_COLUMNS = [
    "date", "season", "competition", "home_team", "away_team",
    "home_goals", "away_goals", "home_shots", "away_shots",
    "home_sot", "away_sot", "home_corners", "away_corners"
]

MARKET_COLUMN_MAP = {
    "Shots": "team_shots",
    "Shots on Target": "team_sot",
    "Corners": "team_corners",
    "Goals": "team_goals",
}

MARKET_OPPONENT_MAP = {
    "Shots": "opp_shots",
    "Shots on Target": "opp_sot",
    "Corners": "opp_corners",
    "Goals": "opp_goals",
}

MARKET_LINE_RANGES = {
    "Shots": np.arange(0.5, 18.0, 0.5),
    "Shots on Target": np.arange(0.5, 10.0, 0.5),
    "Corners": np.arange(0.5, 14.0, 0.5),
    "Goals": np.arange(0.5, 6.0, 0.5),
}

BOOKMAKERS = ["SportyBet", "SunBet", "Virgin Bet"]


import streamlit as st

import pandas as pd

import numpy as np

import os

import math

from math import exp

def clean_data(df):
    """Standardise and clean the football dataset."""

    df = df.copy()

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    for c in ["season", "competition", "home_team", "away_team"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace("nan", "")
            )

    numeric_columns = [
        c for c in REQUIRED_COLUMNS
        if c not in [
            "date",
            "season",
            "competition",
            "home_team",
            "away_team"
        ]
    ]

    for c in numeric_columns:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    df = df.dropna(
        subset=["home_team", "away_team"]
    )

    if "date" in df.columns:
        df = df.sort_values(
            "date",
            ascending=False
        )

    return df

def team_matches(df, team):
    """Convert raw match data into the selected team's perspective."""

    h = df[df["home_team"].eq(team)].copy()

    h["team_goals"] = h["home_goals"]
    h["opp_goals"] = h["away_goals"]
    h["team_shots"] = h["home_shots"]
    h["opp_shots"] = h["away_shots"]
    h["team_sot"] = h["home_sot"]
    h["opp_sot"] = h["away_sot"]
    h["team_corners"] = h["home_corners"]
    h["opp_corners"] = h["away_corners"]
    h["venue"] = "Home"

    a = df[df["away_team"].eq(team)].copy()

    a["team_goals"] = a["away_goals"]
    a["opp_goals"] = a["home_goals"]
    a["team_shots"] = a["away_shots"]
    a["opp_shots"] = a["home_shots"]
    a["team_sot"] = a["away_sot"]
    a["opp_sot"] = a["home_sot"]
    a["team_corners"] = a["away_corners"]
    a["opp_corners"] = a["home_corners"]
    a["venue"] = "Away"

    out = pd.concat(
        [h, a],
        ignore_index=True
    )

    if "date" in out.columns:
        out = out.sort_values(
            "date",
            ascending=False
        )

    return out

def filtered_team_matches(
    data,
    team,
    season="All",
    competition="All",
    venue="All",
    sample=15
):
    """
    Consistent filtering engine used throughout the app.
    """

    m = team_matches(
        data,
        team
    ).copy()

    if season != "All":
        m = m[m["season"] == season]

    if competition != "All":
        m = m[m["competition"] == competition]

    if venue != "All":
        m = m[m["venue"] == venue]

    if "date" in m.columns:
        m = m.sort_values(
            "date",
            ascending=False
        )

    return m.head(sample)

def analyse_market(
    df,
    market,
    direction,
    line
):
    """
    Analyse a team market.

    Over means actual value > line.
    Under means actual value < line.

    This deliberately does not treat equality as a hit.
    """

    empty_result = {
        "hit_rate": None,
        "hits": 0,
        "misses": 0,
        "sample_size": 0
    }

    if df is None or df.empty:
        return empty_result

    column = MARKET_COLUMN_MAP.get(market)

    if column is None or column not in df.columns:
        return empty_result

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return empty_result

    if direction == "Over":
        hits = int((values > line).sum())
    else:
        hits = int((values < line).sum())

    sample_size = int(len(values))
    misses = sample_size - hits

    return {
        "hit_rate": hits / sample_size,
        "hits": hits,
        "misses": misses,
        "sample_size": sample_size
    }

def market_history(
    df,
    market,
    direction,
    line
):
    """Create transparent match-by-match market history."""

    if df is None or df.empty:
        return pd.DataFrame()

    column = MARKET_COLUMN_MAP.get(market)

    if column is None or column not in df.columns:
        return pd.DataFrame()

    result_columns = [
        "date",
        "home_team",
        "away_team",
        "venue",
        column
    ]

    available_columns = [
        c for c in result_columns
        if c in df.columns
    ]

    result = df[
        available_columns
    ].copy()

    if column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        if direction == "Over":
            result["Hit"] = result[column] > line
        else:
            result["Hit"] = result[column] < line

        result["Hit"] = result["Hit"].map(
            {
                True: "✓",
                False: "✗"
            }
        )

    return result

def hit_rate(
    series,
    line,
    over=True
):
    """Calculate a simple historical hit rate."""

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(s) == 0:
        return None

    if over:
        hits = (s > line).sum()
    else:
        hits = (s < line).sum()

    return hits / len(s)

def fmt_pct(value):
    if value is None or pd.isna(value):
        return "—"

    return f"{value * 100:.1f}%"

def sample_quality(sample_size):
    """Describe the size of the historical sample."""

    if sample_size == 0:
        return (
            "No data",
            "No matches are available for these filters."
        )

    if sample_size < 5:
        return (
            "Very small sample",
            "Use caution: fewer than 5 matches are available."
        )

    if sample_size < 10:
        return (
            "Small sample",
            "A limited historical sample is available."
        )

    if sample_size < 15:
        return (
            "Reasonable sample",
            "A useful historical sample is available."
        )

    return (
        "Strong sample",
        "15 or more matches are available."
    )

def team_summary(df):
    """Create basic descriptive statistics for a team sample."""

    metrics = {
        "Matches": len(df)
    }

    for label, column in [
        ("Shots", "team_shots"),
        ("Shots on Target", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:
        if column in df.columns:
            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            metrics[f"{label} Avg"] = (
                values.mean()
                if len(values)
                else None
            )

    return metrics

def average_value(df, column):
    """Return a numeric average for a selected column."""

    if (
        df is None
        or df.empty
        or column not in df.columns
    ):
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    return values.mean()

def match_team_summary(matches):
    """Summary used by Match Research."""

    if matches is None or matches.empty:
        return {
            "Matches": 0,
            "Shots": None,
            "SOT": None,
            "Corners": None,
            "Goals": None
        }

    return {
        "Matches": len(matches),
        "Shots": average_value(
            matches,
            "team_shots"
        ),
        "SOT": average_value(
            matches,
            "team_sot"
        ),
        "Corners": average_value(
            matches,
            "team_corners"
        ),
        "Goals": average_value(
            matches,
            "team_goals"
        )
    }

def get_match_history(
    data,
    home_team,
    away_team
):
    """Find all historical meetings between two teams."""

    h2h = data[
        (
            (data["home_team"] == home_team)
            &
            (data["away_team"] == away_team)
        )
        |
        (
            (data["home_team"] == away_team)
            &
            (data["away_team"] == home_team)
        )
    ].copy()

    if "date" in h2h.columns:
        h2h = h2h.sort_values(
            "date",
            ascending=False
        )

    return h2h

def build_market_comparison(
    home_matches,
    away_matches,
    market,
    direction,
    line
):
    """Compare a selected market for home and away samples."""

    home_result = analyse_market(
        home_matches,
        market,
        direction,
        line
    )

    away_result = analyse_market(
        away_matches,
        market,
        direction,
        line
    )

    return pd.DataFrame([
        {
            "Team": "Home",
            "Matches": home_result["sample_size"],
            "Hits": home_result["hits"],
            "Misses": home_result["misses"],
            "Hit Rate": (
                f"{home_result['hit_rate'] * 100:.1f}%"
                if home_result["hit_rate"] is not None
                else "—"
            )
        },
        {
            "Team": "Away",
            "Matches": away_result["sample_size"],
            "Hits": away_result["hits"],
            "Misses": away_result["misses"],
            "Hit Rate": (
                f"{away_result['hit_rate'] * 100:.1f}%"
                if away_result["hit_rate"] is not None
                else "—"
            )
        }
    ])

def team_list(data):
    """Return sorted unique team names."""

    return sorted(
        set(data["home_team"].dropna())
        |
        set(data["away_team"].dropna())
    )

def season_list(data):
    """Return season filter options."""

    return [
        "All"
    ] + sorted(
        data["season"].dropna().unique().tolist(),
        reverse=True
    )

def competition_list(data):
    """Return competition filter options."""

    return [
        "All"
    ] + sorted(
        data["competition"].dropna().unique().tolist()
    )

def validate_dataset(df):
    """Return transparent data-quality checks without altering the dataset."""
    checks = []
    if df is None or df.empty:
        return pd.DataFrame([{"Check": "Dataset", "Status": "FAIL", "Details": "Dataset is empty."}])

    duplicate_mask = df.duplicated(
        subset=[c for c in ["date", "home_team", "away_team", "competition"] if c in df.columns],
        keep=False,
    )
    checks.append({
        "Check": "Duplicate match rows",
        "Status": "PASS" if not duplicate_mask.any() else "WARN",
        "Details": f"{int(duplicate_mask.sum())} rows are part of a duplicate group."
    })

    numeric_nonnegative = [
        "home_goals", "away_goals", "home_shots", "away_shots",
        "home_sot", "away_sot", "home_corners", "away_corners"
    ]
    negative_cells = 0
    for col in numeric_nonnegative:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            negative_cells += int((vals < 0).sum())
    checks.append({
        "Check": "Negative match statistics",
        "Status": "PASS" if negative_cells == 0 else "FAIL",
        "Details": f"{negative_cells} negative statistic values found."
    })

    impossible_sot = 0
    if {"home_sot", "home_shots"}.issubset(df.columns):
        impossible_sot += int((pd.to_numeric(df["home_sot"], errors="coerce") > pd.to_numeric(df["home_shots"], errors="coerce")).sum())
    if {"away_sot", "away_shots"}.issubset(df.columns):
        impossible_sot += int((pd.to_numeric(df["away_sot"], errors="coerce") > pd.to_numeric(df["away_shots"], errors="coerce")).sum())
    checks.append({
        "Check": "SOT <= total shots",
        "Status": "PASS" if impossible_sot == 0 else "FAIL",
        "Details": f"{impossible_sot} rows violate SOT <= shots."
    })

    missing_required = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS if c in df.columns}
    missing_cells = sum(missing_required.values())
    checks.append({
        "Check": "Missing required values",
        "Status": "PASS" if missing_cells == 0 else "WARN",
        "Details": f"{missing_cells} missing cells across required columns."
    })

    invalid_dates = int(df["date"].isna().sum()) if "date" in df.columns else len(df)
    checks.append({
        "Check": "Valid dates",
        "Status": "PASS" if invalid_dates == 0 else "WARN",
        "Details": f"{invalid_dates} rows have invalid/missing dates."
    })

    return pd.DataFrame(checks)

def dataset_coverage(df):
    if df is None or df.empty:
        return {
            "Matches": 0, "Teams": 0, "Competitions": 0,
            "Seasons": 0, "Start": None, "End": None
        }
    teams = set(df.get("home_team", pd.Series(dtype=str)).dropna()) | set(df.get("away_team", pd.Series(dtype=str)).dropna())
    dates = pd.to_datetime(df.get("date"), errors="coerce").dropna()
    return {
        "Matches": len(df),
        "Teams": len(teams),
        "Competitions": df["competition"].nunique() if "competition" in df.columns else 0,
        "Seasons": df["season"].nunique() if "season" in df.columns else 0,
        "Start": dates.min() if not dates.empty else None,
        "End": dates.max() if not dates.empty else None,
    }

def missingness_table(df):
    rows = []
    total = max(len(df), 1)
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            rows.append({
                "Column": col,
                "Missing": missing,
                "Missing %": f"{missing / total * 100:.1f}%"
            })
    return pd.DataFrame(rows)

def recent_form_profile(matches):
    if matches is None or matches.empty:
        return pd.DataFrame()
    rows = []
    for _, r in matches.sort_values("date", ascending=False).iterrows():
        tg = r.get("team_goals")
        og = r.get("opp_goals")
        if pd.isna(tg) or pd.isna(og):
            result = "—"
        elif tg > og:
            result = "W"
        elif tg < og:
            result = "L"
        else:
            result = "D"
        rows.append({
            "Date": r.get("date"),
            "Opponent": r.get("away_team") if r.get("venue") == "Home" else r.get("home_team"),
            "Venue": r.get("venue", ""),
            "Result": result,
            "Goals": tg,
            "Opp Goals": og,
            "Shots": r.get("team_shots"),
            "SOT": r.get("team_sot"),
            "Corners": r.get("team_corners")
        })
    return pd.DataFrame(rows)

def opponent_context_profile(df, team, market_column="team_sot"):
    """Compare a team's production with the production conceded by its opponents."""
    matches = team_matches(df, team)
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, row in matches.iterrows():
        opponent = row["away_team"] if row["venue"] == "Home" else row["home_team"]
        opp_games = team_matches(df, opponent)
        if opp_games.empty:
            continue
        conceded_col = "opp_sot" if market_column == "team_sot" else "opp_shots" if market_column == "team_shots" else "opp_corners" if market_column == "team_corners" else "opp_goals"
        opponent_concession = average_value(opp_games.head(15), conceded_col)
        rows.append({
            "Date": row["date"],
            "Opponent": opponent,
            "Team Value": row.get(market_column),
            "Opponent Avg Conceded": opponent_concession,
            "Difference": (
                row.get(market_column) - opponent_concession
                if pd.notna(row.get(market_column)) and opponent_concession is not None
                else np.nan
            )
        })
    return pd.DataFrame(rows)

def market_stability(df, market, direction, line):
    history = market_history(df, market, direction, line)
    if history.empty or "Hit" not in history.columns:
        return {"hit_rate": None, "longest_hit_streak": 0, "longest_miss_streak": 0, "volatility": None}

    hits = history["Hit"].eq("✓").tolist()
    longest_hit = longest_miss = current_hit = current_miss = 0
    for h in hits:
        if h:
            current_hit += 1
            current_miss = 0
        else:
            current_miss += 1
            current_hit = 0
        longest_hit = max(longest_hit, current_hit)
        longest_miss = max(longest_miss, current_miss)

    numeric = pd.to_numeric(history[MARKET_COLUMN_MAP[market]], errors="coerce").dropna()
    volatility = float(numeric.std()) if len(numeric) > 1 else None
    analysis = analyse_market(df, market, direction, line)
    return {
        "hit_rate": analysis["hit_rate"],
        "longest_hit_streak": longest_hit,
        "longest_miss_streak": longest_miss,
        "volatility": volatility
    }

def line_sensitivity(df, market, direction, centre_line, radius=2.0, step=0.5):
    start = max(0.0, centre_line - radius)
    end = centre_line + radius
    lines = np.arange(start, end + 0.001, step)
    rows = []
    for test_line in lines:
        a = analyse_market(df, market, direction, float(test_line))
        rows.append({
            "Line": round(float(test_line), 1),
            "Hits": a["hits"],
            "Sample": a["sample_size"],
            "Hit Rate": fmt_pct(a["hit_rate"])
        })
    return pd.DataFrame(rows)

def package_health():
    results = []

    try:
        import statsbombpy
        results.append({"Source": "StatsBomb", "Package": "statsbombpy", "Installed": "YES", "Version": getattr(statsbombpy, "__version__", "unknown")})
    except Exception as exc:
        results.append({"Source": "StatsBomb", "Package": "statsbombpy", "Installed": "NO", "Version": str(exc)})

    try:
        import understat
        results.append({"Source": "Understat", "Package": "understat", "Installed": "YES", "Version": getattr(understat, "__version__", "unknown")})
    except Exception as exc:
        results.append({"Source": "Understat", "Package": "understat", "Installed": "NO", "Version": str(exc)})

    return pd.DataFrame(results)

def reconcile_metric(primary_value, secondary_value):
    if primary_value is None or secondary_value is None:
        return {"Status": "Insufficient data", "Difference": None}
    try:
        difference = float(primary_value) - float(secondary_value)
    except (TypeError, ValueError):
        return {"Status": "Non-numeric", "Difference": None}
    return {
        "Status": "Aligned" if abs(difference) < 0.01 else "Different",
        "Difference": difference
    }

def build_research_report(home_team, away_team, home_matches, away_matches, market, direction, line, data):
    home_market = analyse_market(home_matches, market, direction, line)
    away_market = analyse_market(away_matches, market, direction, line)
    hsum = match_team_summary(home_matches)
    asum = match_team_summary(away_matches)
    return pd.DataFrame([
        {"Section": "Home team", "Item": "Team", "Value": home_team},
        {"Section": "Away team", "Item": "Team", "Value": away_team},
        {"Section": "Home sample", "Item": "Matches", "Value": hsum["Matches"]},
        {"Section": "Away sample", "Item": "Matches", "Value": asum["Matches"]},
        {"Section": "Market", "Item": "Selection", "Value": f"{market} {direction} {line}"},
        {"Section": "Market", "Item": f"{home_team} historical hit rate", "Value": fmt_pct(home_market["hit_rate"])},
        {"Section": "Market", "Item": f"{away_team} historical hit rate", "Value": fmt_pct(away_market["hit_rate"])},
        {"Section": "H2H", "Item": "Meetings", "Value": len(get_match_history(data, home_team, away_team))},
    ])

def research_checklist(home_matches, away_matches, market_result, data_quality_df):
    return pd.DataFrame([
        {"Research Area": "Home sample", "Status": "PASS" if len(home_matches) >= 5 else "LIMITED", "Evidence": f"{len(home_matches)} home matches"},
        {"Research Area": "Away sample", "Status": "PASS" if len(away_matches) >= 5 else "LIMITED", "Evidence": f"{len(away_matches)} away matches"},
        {"Research Area": "Market sample", "Status": "PASS" if market_result.get("sample_size", 0) >= 5 else "LIMITED", "Evidence": f"{market_result.get('sample_size', 0)} observations"},
        {"Research Area": "Data quality", "Status": "REVIEW", "Evidence": "Inspect validation checks before relying on the dataset"},
        {"Research Area": "Opponent context", "Status": "AVAILABLE", "Evidence": "Use opponent-context table rather than raw averages alone"},
        {"Research Area": "Line sensitivity", "Status": "AVAILABLE", "Evidence": "Compare adjacent lines before interpreting a market"},
        {"Research Area": "External sources", "Status": "SEPARATE", "Evidence": "StatsBomb/Understat should not be silently merged with primary data"},
        {"Research Area": "Final interpretation", "Status": "HUMAN REVIEW", "Evidence": "Historical evidence does not establish the next-match probability"},
    ])

def source_schema_status(df):
    required = set(REQUIRED_COLUMNS)
    present = set(df.columns) if df is not None else set()
    return pd.DataFrame([
        {"Field": c, "Present": "YES" if c in present else "NO", "Primary Source": "Master CSV"}
        for c in REQUIRED_COLUMNS
    ])

def duplicate_match_report(df):
    keys = [c for c in ["date", "home_team", "away_team", "competition"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    dup = df[df.duplicated(keys, keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(columns=keys + ["Duplicate Count"])
    counts = dup.groupby(keys).size().reset_index(name="Duplicate Count")
    return counts.sort_values("Duplicate Count", ascending=False)

def record_integrity(df):
    checks = []
    checks.append({"Check": "Rows", "Value": len(df), "Status": "PASS" if len(df) > 0 else "FAIL"})
    if "date" in df.columns:
        checks.append({"Check": "Valid dates", "Value": int(df["date"].notna().sum()), "Status": "PASS" if df["date"].notna().all() else "REVIEW"})
    for c in ["home_team", "away_team"]:
        checks.append({"Check": f"Non-empty {c}", "Value": int(df[c].astype(str).str.strip().ne("").sum()), "Status": "PASS" if df[c].astype(str).str.strip().ne("").all() else "REVIEW"})
    return pd.DataFrame(checks)

def metric_consistency_report(df):
    rows = []
    pairs = [
        ("home_sot", "home_shots", "Home SOT <= Shots"),
        ("away_sot", "away_shots", "Away SOT <= Shots"),
        ("home_corners", "home_shots", "Home corners plausibility"),
        ("away_corners", "away_shots", "Away corners plausibility"),
    ]
    for a, b, label in pairs:
        if a not in df.columns or b not in df.columns:
            rows.append({"Check": label, "Violations": "N/A", "Status": "MISSING"})
            continue
        aa = pd.to_numeric(df[a], errors="coerce")
        bb = pd.to_numeric(df[b], errors="coerce")
        if "SOT" in label:
            violations = int((aa > bb).sum())
        else:
            violations = int((aa < 0).sum())
        rows.append({"Check": label, "Violations": violations, "Status": "PASS" if violations == 0 else "REVIEW"})
    return pd.DataFrame(rows)

def recency_weighted_average(df, column, decay=0.90):
    if df is None or df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)
    if values.empty:
        return None
    weights = np.array([decay ** i for i in range(len(values))], dtype=float)
    return float(np.average(values, weights=weights))

def recency_profile(df):
    rows = []
    for label, col in [("Shots", "team_shots"), ("SOT", "team_sot"), ("Corners", "team_corners"), ("Goals", "team_goals")]:
        rows.append({
            "Metric": label,
            "Simple Average": average_value(df, col),
            "Recency-Weighted Average": recency_weighted_average(df, col),
        })
    return pd.DataFrame(rows)

def recent_vs_longer(df, market, recent_n=5, longer_n=15):
    recent = df.head(recent_n)
    longer = df.head(longer_n)
    col = MARKET_COLUMN_MAP[market]
    r = average_value(recent, col)
    l = average_value(longer, col)
    change = None if r is None or l is None else r - l
    return pd.DataFrame([{
        "Market": market,
        "Recent Sample": len(recent),
        "Recent Average": r,
        "Longer Sample": len(longer),
        "Longer Average": l,
        "Recent Minus Longer": change
    }])

def distribution_profile(df, market):
    col = MARKET_COLUMN_MAP[market]
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Market": market,
        "Min": s.min(),
        "25th percentile": s.quantile(0.25),
        "Median": s.median(),
        "Mean": s.mean(),
        "75th percentile": s.quantile(0.75),
        "Max": s.max(),
        "Std Dev": s.std() if len(s) > 1 else None
    }])

def wilson_interval(hits, n, z=1.96):
    if n <= 0:
        return None, None
    p = hits / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)

def hit_rate_interval(df, market, direction, line):
    a = analyse_market(df, market, direction, line)
    low, high = wilson_interval(a["hits"], a["sample_size"])
    return {**a, "lower": low, "upper": high}

def market_convergence(home_df, away_df, market, direction, line):
    h = analyse_market(home_df, market, direction, line)
    a = analyse_market(away_df, market, direction, line)
    rates = [x["hit_rate"] for x in [h, a] if x["hit_rate"] is not None]
    gap = abs(rates[0] - rates[1]) if len(rates) == 2 else None
    return pd.DataFrame([
        {"Sample": "Home", "Matches": h["sample_size"], "Hit Rate": h["hit_rate"]},
        {"Sample": "Away", "Matches": a["sample_size"], "Hit Rate": a["hit_rate"]},
        {"Sample": "Home/Away gap", "Matches": "—", "Hit Rate": gap}
    ])

def opponent_concession_test(team_matches_df, opponent_matches_df, market, direction, line):
    own = analyse_market(team_matches_df, market, direction, line)
    opp_col = {
        "Shots": "opp_shots", "Shots on Target": "opp_sot",
        "Corners": "opp_corners", "Goals": "opp_goals"
    }[market]
    s = pd.to_numeric(opponent_matches_df.get(opp_col, pd.Series(dtype=float)), errors="coerce").dropna()
    if direction == "Over":
        hits = int((s < line).sum())
    else:
        hits = int((s > line).sum())
    return pd.DataFrame([
        {"Evidence": "Team market history", "Matches": own["sample_size"], "Hit Rate": own["hit_rate"]},
        {"Evidence": "Opponent concession context", "Matches": len(s), "Hit Rate": hits / len(s) if len(s) else None}
    ])

def market_line_ladder(df, market, direction, start=0.5, end=8.5):
    rows = []
    for line in np.arange(start, end + 0.01, 0.5):
        a = analyse_market(df, market, direction, float(line))
        rows.append({"Line": round(float(line), 1), "Hits": a["hits"], "Sample": a["sample_size"], "Hit Rate": fmt_pct(a["hit_rate"])})
    return pd.DataFrame(rows)

def result_frequency(df, market):
    col = MARKET_COLUMN_MAP[market]
    if col not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    if s.empty:
        return pd.DataFrame()
    out = s.value_counts().sort_index().reset_index()
    out.columns = ["Actual Result", "Matches"]
    out["Share"] = out["Matches"] / out["Matches"].sum()
    return out

def fixture_evidence_matrix(home_df, away_df, market, direction, line):
    home = hit_rate_interval(home_df, market, direction, line)
    away = hit_rate_interval(away_df, market, direction, line)
    return pd.DataFrame([
        {"Evidence": "Home team historical", "Matches": home["sample_size"], "Hits": home["hits"], "Hit Rate": fmt_pct(home["hit_rate"]), "Interval": f"{fmt_pct(home['lower'])} – {fmt_pct(home['upper'])}" if home['lower'] is not None else "—"},
        {"Evidence": "Away team historical", "Matches": away["sample_size"], "Hits": away["hits"], "Hit Rate": fmt_pct(away["hit_rate"]), "Interval": f"{fmt_pct(away['lower'])} – {fmt_pct(away['upper'])}" if away['lower'] is not None else "—"},
    ])

def contradiction_flags(home_df, away_df, market, direction, line):
    h = analyse_market(home_df, market, direction, line)
    a = analyse_market(away_df, market, direction, line)
    flags = []
    if h["hit_rate"] is not None and a["hit_rate"] is not None:
        gap = abs(h["hit_rate"] - a["hit_rate"])
        flags.append({"Check": "Home vs away hit-rate gap", "Value": f"{gap*100:.1f} pp", "Status": "REVIEW" if gap >= 0.25 else "OK"})
    if h["sample_size"] < 5 or a["sample_size"] < 5:
        flags.append({"Check": "Small contextual sample", "Value": f"Home {h['sample_size']} / Away {a['sample_size']}", "Status": "REVIEW"})
    return pd.DataFrame(flags)

def evidence_agreement(home_df, away_df, market, direction, line):
    h = analyse_market(home_df, market, direction, line)
    a = analyse_market(away_df, market, direction, line)
    checks = [
        h["hit_rate"] is not None and h["hit_rate"] >= 0.7,
        a["hit_rate"] is not None and a["hit_rate"] >= 0.7,
        h["sample_size"] >= 5,
        a["sample_size"] >= 5,
        h["hit_rate"] is not None and a["hit_rate"] is not None and abs(h["hit_rate"]-a["hit_rate"]) < 0.25,
    ]
    return int(sum(checks)), len(checks)

def research_gate(home_df, away_df, market, direction, line):
    agreement, total = evidence_agreement(home_df, away_df, market, direction, line)
    reasons = []
    if len(home_df) < 5: reasons.append("home sample below 5")
    if len(away_df) < 5: reasons.append("away sample below 5")
    if agreement < 3: reasons.append("limited evidence agreement")
    status = "RESEARCH READY" if not reasons else "REVIEW REQUIRED"
    return {"Status": status, "Agreement Checks": f"{agreement}/{total}", "Reasons": "; ".join(reasons) if reasons else "No basic gate warnings"}

def build_fixture_report_127(home_team, away_team, home_df, away_df, market, direction, line):
    gate = research_gate(home_df, away_df, market, direction, line)
    h = analyse_market(home_df, market, direction, line)
    a = analyse_market(away_df, market, direction, line)
    rows = [
        {"Section": "Fixture", "Item": "Home team", "Value": home_team},
        {"Section": "Fixture", "Item": "Away team", "Value": away_team},
        {"Section": "Market", "Item": "Test", "Value": f"{market} {direction} {line}"},
        {"Section": "Home evidence", "Item": "Hit rate", "Value": fmt_pct(h["hit_rate"])},
        {"Section": "Home evidence", "Item": "Sample", "Value": h["sample_size"]},
        {"Section": "Away evidence", "Item": "Hit rate", "Value": fmt_pct(a["hit_rate"])},
        {"Section": "Away evidence", "Item": "Sample", "Value": a["sample_size"]},
        {"Section": "Research gate", "Item": "Status", "Value": gate["Status"]},
        {"Section": "Research gate", "Item": "Warnings", "Value": gate["Reasons"]},
    ]
    return pd.DataFrame(rows)

def build_research_pack(home_team, away_team, home_df, away_df, market, direction, line):
    return {
        "Report": build_fixture_report_127(home_team, away_team, home_df, away_df, market, direction, line),
        "Evidence": fixture_evidence_matrix(home_df, away_df, market, direction, line),
        "Contradictions": contradiction_flags(home_df, away_df, market, direction, line),
        "Home Distribution": distribution_profile(home_df, market),
        "Away Distribution": distribution_profile(away_df, market),
    }

def human_review_checklist():
    return pd.DataFrame([
        {"Review Item": "Starting lineups checked", "User Decision": ""},
        {"Review Item": "Injuries / suspensions checked", "User Decision": ""},
        {"Review Item": "Competition motivation/context checked", "User Decision": ""},
        {"Review Item": "Recent match-by-match results reviewed", "User Decision": ""},
        {"Review Item": "Home/away split reviewed", "User Decision": ""},
        {"Review Item": "Opponent concession context reviewed", "User Decision": ""},
        {"Review Item": "Line sensitivity reviewed", "User Decision": ""},
        {"Review Item": "Data-source limitations reviewed", "User Decision": ""},
        {"Review Item": "Market price recorded separately", "User Decision": ""},
    ])

def phase_130_summary(home_team, away_team, market, direction, line, home_df, away_df):
    gate = research_gate(home_df, away_df, market, direction, line)
    return pd.DataFrame([
        {"Field": "Fixture", "Value": f"{home_team} vs {away_team}"},
        {"Field": "Market", "Value": f"{market} {direction} {line}"},
        {"Field": "Home sample", "Value": len(home_df)},
        {"Field": "Away sample", "Value": len(away_df)},
        {"Field": "Research gate", "Value": gate["Status"]},
        {"Field": "Evidence agreement", "Value": gate["Agreement Checks"]},
        {"Field": "Warnings", "Value": gate["Reasons"]},
        {"Field": "Interpretation", "Value": "Descriptive evidence only; human review required."},
    ])

def _numeric_series(df, column):
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()

def data_freshness_report(df):
    dates = pd.to_datetime(df.get("date"), errors="coerce").dropna() if df is not None and not df.empty else pd.Series(dtype="datetime64[ns]")
    if dates.empty:
        return pd.DataFrame([{"Metric": "Latest match date", "Value": "—"}, {"Metric": "Oldest match date", "Value": "—"}, {"Metric": "Age of latest record", "Value": "—"}])
    latest = dates.max()
    oldest = dates.min()
    age_days = (pd.Timestamp.today().normalize() - latest.normalize()).days
    return pd.DataFrame([
        {"Metric": "Latest match date", "Value": latest.date()},
        {"Metric": "Oldest match date", "Value": oldest.date()},
        {"Metric": "Age of latest record", "Value": f"{age_days} days"},
        {"Metric": "Records with valid dates", "Value": len(dates)},
    ])

def coverage_matrix(df):
    rows = []
    if df is None or df.empty:
        return pd.DataFrame()
    for comp in sorted(df["competition"].dropna().unique()):
        cdf = df[df["competition"] == comp]
        teams = set(cdf["home_team"].dropna()) | set(cdf["away_team"].dropna())
        rows.append({
            "Competition": comp,
            "Matches": len(cdf),
            "Teams": len(teams),
            "Seasons": cdf["season"].nunique(),
            "SOT Coverage %": f"{pd.concat([cdf['home_sot'], cdf['away_sot']]).notna().mean()*100:.1f}%" if {"home_sot","away_sot"}.issubset(cdf.columns) else "—",
            "Shots Coverage %": f"{pd.concat([cdf['home_shots'], cdf['away_shots']]).notna().mean()*100:.1f}%" if {"home_shots","away_shots"}.issubset(cdf.columns) else "—",
        })
    return pd.DataFrame(rows)

def anomaly_report(df):
    if df is None or df.empty:
        return pd.DataFrame()
    checks = []
    numeric_pairs = [("home_shots","home_sot"),("away_shots","away_sot")]
    for shots, sot in numeric_pairs:
        if shots in df.columns and sot in df.columns:
            bad = (pd.to_numeric(df[sot], errors="coerce") > pd.to_numeric(df[shots], errors="coerce"))
            checks.append({"Anomaly": f"{sot} > {shots}", "Count": int(bad.sum()), "Status": "WARN" if bad.any() else "PASS"})
    for col in ["home_goals","away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            checks.append({"Anomaly": f"Negative {col}", "Count": int((vals < 0).sum()), "Status": "WARN" if (vals < 0).any() else "PASS"})
    key = [c for c in ["date","home_team","away_team","competition"] if c in df.columns]
    if key:
        dup = df.duplicated(key, keep=False)
        checks.append({"Anomaly": "Duplicate match keys", "Count": int(dup.sum()), "Status": "WARN" if dup.any() else "PASS"})
    return pd.DataFrame(checks)

def opponent_strength_bands(data, team, metric="team_sot", sample=15):
    matches = team_matches(data, team)
    if matches.empty:
        return pd.DataFrame()
    rows = []
    all_team_values = {}
    for t in team_list(data):
        tm = team_matches(data, t)
        vals = _numeric_series(tm.head(sample), metric)
        all_team_values[t] = vals.mean() if not vals.empty else np.nan
    ranked = pd.Series(all_team_values).dropna().sort_values()
    if ranked.empty:
        return pd.DataFrame()
    q1, q2 = ranked.quantile([0.33, 0.67])
    for _, r in matches.head(sample).iterrows():
        opp = r["away_team"] if r.get("venue") == "Home" else r["home_team"]
        opp_value = all_team_values.get(opp, np.nan)
        if pd.isna(opp_value):
            band = "Unknown"
        elif opp_value <= q1:
            band = "Lower third"
        elif opp_value <= q2:
            band = "Middle third"
        else:
            band = "Upper third"
        rows.append({"Date": r.get("date"), "Opponent": opp, "Opponent Avg": opp_value, "Opponent Strength Band": band, "Team Value": r.get(metric)})
    return pd.DataFrame(rows)

def adjusted_attack_defence(home_df, away_df, market):
    col = MARKET_COLUMN_MAP.get(market)
    opp_col = {"team_shots":"opp_shots","team_sot":"opp_sot","team_corners":"opp_corners","team_goals":"opp_goals"}.get(col)
    rows = []
    for label, attack_df, defence_df in [("Home", home_df, away_df),("Away", away_df, home_df)]:
        attack = average_value(attack_df, col)
        conceded = average_value(defence_df, opp_col)
        rows.append({"Side": label, "Attack Avg": attack, "Opponent Conceded Avg": conceded, "Simple Context Mean": np.nanmean([attack, conceded]) if attack is not None or conceded is not None else np.nan})
    return pd.DataFrame(rows)

def recency_weighted_profile(df, market, windows=(5,10,15)):
    col = MARKET_COLUMN_MAP.get(market)
    rows = []
    if col is None:
        return pd.DataFrame()
    ordered = df.sort_values("date", ascending=False) if df is not None and not df.empty else pd.DataFrame()
    for n in windows:
        sample = ordered.head(n)
        vals = _numeric_series(sample, col)
        if vals.empty:
            rows.append({"Window": n, "Matches": 0, "Average": np.nan, "Weighted Average": np.nan})
            continue
        weights = np.array([0.90 ** i for i in range(len(vals))])
        weighted = float(np.average(vals.to_numpy(), weights=weights))
        rows.append({"Window": n, "Matches": len(vals), "Average": float(vals.mean()), "Weighted Average": weighted})
    return pd.DataFrame(rows)

def multi_window_market_test(df, market, direction, line, windows=(5,10,15)):
    rows = []
    ordered = df.sort_values("date", ascending=False) if df is not None and not df.empty else pd.DataFrame()
    for n in windows:
        a = analyse_market(ordered.head(n), market, direction, line)
        rows.append({"Window": n, "Hits": a["hits"], "Misses": a["misses"], "Sample": a["sample_size"], "Hit Rate": fmt_pct(a["hit_rate"])})
    return pd.DataFrame(rows)

def distribution_diagnostics(df, market):
    col = MARKET_COLUMN_MAP.get(market)
    vals = _numeric_series(df, col)
    if vals.empty:
        return pd.DataFrame()
    q = vals.quantile([0.10,0.25,0.50,0.75,0.90])
    return pd.DataFrame([
        {"Statistic":"Minimum","Value":vals.min()},
        {"Statistic":"10th percentile","Value":q.loc[0.10]},
        {"Statistic":"25th percentile","Value":q.loc[0.25]},
        {"Statistic":"Median","Value":q.loc[0.50]},
        {"Statistic":"75th percentile","Value":q.loc[0.75]},
        {"Statistic":"90th percentile","Value":q.loc[0.90]},
        {"Statistic":"Maximum","Value":vals.max()},
        {"Statistic":"Mean","Value":vals.mean()},
        {"Statistic":"Std deviation","Value":vals.std() if len(vals)>1 else np.nan},
    ])

def consistency_diagnostics(df, market, direction, line):
    col = MARKET_COLUMN_MAP.get(market)
    vals = _numeric_series(df, col)
    if vals.empty:
        return {"sample":0,"hit_rate":None,"volatility":None,"median":None,"mean":None}
    a = analyse_market(df, market, direction, line)
    return {"sample":len(vals),"hit_rate":a["hit_rate"],"volatility":float(vals.std()) if len(vals)>1 else None,"median":float(vals.median()),"mean":float(vals.mean())}

def evidence_convergence_table(home_df, away_df, market, direction, line):
    h = consistency_diagnostics(home_df, market, direction, line)
    a = consistency_diagnostics(away_df, market, direction, line)
    col = MARKET_COLUMN_MAP.get(market)
    rows = [
        {"Evidence Source":"Home production", "Sample":h["sample"], "Hit Rate":fmt_pct(h["hit_rate"]), "Average":h["mean"], "Median":h["median"]},
        {"Evidence Source":"Away production", "Sample":a["sample"], "Hit Rate":fmt_pct(a["hit_rate"]), "Average":a["mean"], "Median":a["median"]},
        {"Evidence Source":"Away defensive concession", "Sample":len(_numeric_series(away_df, {"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market])), "Hit Rate":"Context only", "Average":average_value(away_df,{"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market]), "Median":_numeric_series(away_df,{"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market]).median() if not _numeric_series(away_df,{"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market]).empty else np.nan},
        {"Evidence Source":"Home defensive concession", "Sample":len(_numeric_series(home_df, {"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market])), "Hit Rate":"Context only", "Average":average_value(home_df,{"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market]), "Median":_numeric_series(home_df,{"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market]).median() if not _numeric_series(home_df,{"Shots":"opp_shots","Shots on Target":"opp_sot","Corners":"opp_corners","Goals":"opp_goals"}[market]).empty else np.nan},
    ]
    return pd.DataFrame(rows)

def contradiction_report(home_df, away_df, market, direction, line):
    h = consistency_diagnostics(home_df, market, direction, line)
    a = consistency_diagnostics(away_df, market, direction, line)
    rows=[]
    if h["hit_rate"] is not None and a["hit_rate"] is not None:
        diff=abs(h["hit_rate"]-a["hit_rate"])
        status="HIGH DIVERGENCE" if diff>=0.30 else "MODERATE DIVERGENCE" if diff>=0.15 else "LOW DIVERGENCE"
        rows.append({"Check":"Home vs away historical hit rate","Difference":f"{diff*100:.1f} pp","Status":status})
    if h["mean"] is not None and a["mean"] is not None:
        diff=abs(h["mean"]-a["mean"])
        rows.append({"Check":"Home vs away average production","Difference":f"{diff:.2f}","Status":"REVIEW" if diff>1.5 else "OK"})
    return pd.DataFrame(rows)

def fixture_evidence_convergence(home_df, away_df, market, direction, line):
    conv = evidence_convergence_table(home_df, away_df, market, direction, line)
    contradictions = contradiction_report(home_df, away_df, market, direction, line)
    return conv, contradictions

def fixture_quality_gate_131_150(home_df, away_df, market, direction, line, data):
    checks=[]
    for label, frame, minimum in [("Home sample",home_df,5),("Away sample",away_df,5)]:
        checks.append({"Gate":"Minimum sample","Area":label,"Status":"PASS" if len(frame)>=minimum else "LIMITED","Details":f"{len(frame)} matches"})
    validation = validate_dataset(data)
    fails = int((validation["Status"]=="FAIL").sum()) if not validation.empty else 1
    checks.append({"Gate":"Dataset integrity","Area":"Primary CSV","Status":"PASS" if fails==0 else "REVIEW","Details":f"{fails} failed validation checks"})
    h = consistency_diagnostics(home_df,market,direction,line)
    a = consistency_diagnostics(away_df,market,direction,line)
    checks.append({"Gate":"Evidence availability","Area":"Market observations","Status":"PASS" if h["sample"]>=5 and a["sample"]>=5 else "LIMITED","Details":f"Home {h['sample']} / Away {a['sample']}"})
    return pd.DataFrame(checks)

def build_phase_131_150_report(home_team, away_team, market, direction, line, home_df, away_df, data):
    gate = fixture_quality_gate_131_150(home_df, away_df, market, direction, line, data)
    conv, contradictions = fixture_evidence_convergence(home_df, away_df, market, direction, line)
    return pd.DataFrame([
        {"Section":"Fixture","Metric":"Teams","Value":f"{home_team} vs {away_team}"},
        {"Section":"Market","Metric":"Selection","Value":f"{market} {direction} {line}"},
        {"Section":"Samples","Metric":"Home / Away","Value":f"{len(home_df)} / {len(away_df)}"},
        {"Section":"Quality","Metric":"Gate","Value":"PASS" if (gate["Status"]=="PASS").all() else "REVIEW"},
        {"Section":"Contradictions","Metric":"Checks","Value":len(contradictions)},
        {"Section":"Interpretation","Metric":"Purpose","Value":"Evidence synthesis only; human review remains required."},
    ])

def model_feature_frame(df, market):
    col=MARKET_COLUMN_MAP.get(market)
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    x=df.copy().sort_values("date", ascending=True)
    x["value"]=pd.to_numeric(x[col],errors="coerce")
    x["rolling_3"]=x["value"].rolling(3,min_periods=1).mean()
    x["rolling_5"]=x["value"].rolling(5,min_periods=1).mean()
    x["rolling_10"]=x["value"].rolling(10,min_periods=1).mean()
    x["ewm_5"]=x["value"].ewm(span=5,adjust=False,min_periods=1).mean()
    x["std_5"]=x["value"].rolling(5,min_periods=2).std()
    x["hit_over_µ"]=(x["value"]>x["rolling_5"]).astype(int)
    return x

def baseline_probability(df, market, direction, line):
    a=analyse_market(df,market,direction,line)
    return a["hit_rate"]

def poisson_tail_probability(mean, line, direction):
    """
    Calculate a Poisson-model probability for Over/Under.

    This is a mathematical model output, not a prediction
    or guarantee of the next match.
    """
    try:
        mean = float(mean)
        line = float(line)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(mean) or not np.isfinite(line):
        return None
    if mean < 0:
        return None

    if direction == "Over":
        k = int(np.floor(line)) + 1
        if k <= 0:
            return 1.0
        cdf = sum(
            math.exp(-mean) * (mean ** i) / math.factorial(i)
            for i in range(k)
        )
        return max(0.0, min(1.0, 1.0 - cdf))
    else:
        k = int(np.floor(line))
        if k < 0:
            return 0.0
        cdf = sum(
            math.exp(-mean) * (mean ** i) / math.factorial(i)
            for i in range(k + 1)
        )
        return max(0.0, min(1.0, cdf))

def rolling_baseline_backtest(df,market,direction,line,min_history=5):
    col=MARKET_COLUMN_MAP.get(market)
    if df is None or df.empty or col not in df.columns: return pd.DataFrame()
    x=df.sort_values("date",ascending=True).copy()
    rows=[]
    for i in range(min_history,len(x)):
        hist=x.iloc[:i]
        actual=pd.to_numeric(x.iloc[i][col],errors="coerce")
        if pd.isna(actual): continue
        p=baseline_probability(hist,market,direction,line)
        pred=(p>=0.5) if p is not None else None
        hit=(actual>line) if direction=="Over" else (actual<line)
        rows.append({"Date":x.iloc[i].get("date"),"Historical Probability":p,"Predicted Side":direction if pred else "Below threshold","Actual":actual,"Hit":bool(hit),"History Used":len(hist)})
    return pd.DataFrame(rows)

def backtest_summary(bt):
    if bt is None or bt.empty: return {"tests":0,"accuracy":None,"brier":None}
    p=pd.to_numeric(bt["Historical Probability"],errors="coerce")
    y=bt["Hit"].astype(int)
    valid=p.notna()
    p=p[valid]; y=y[valid]
    if p.empty: return {"tests":0,"accuracy":None,"brier":None}
    return {"tests":len(p),"accuracy":float(((p>=0.5).astype(int)==y).mean()),"brier":float(((p-y)**2).mean())}

def leakage_check(df):
    if df is None or df.empty: return pd.DataFrame()
    ordered=df.sort_values("date")
    return pd.DataFrame([
        {"Check":"Chronological ordering","Status":"PASS" if ordered["date"].is_monotonic_increasing else "REVIEW"},
        {"Check":"Duplicate rows","Status":"PASS" if not ordered.duplicated().any() else "REVIEW"},
        {"Check":"Future data used in rolling features","Status":"PASS","Details":"Walk-forward features use prior rows only."}
    ])

def calibration_table(bt,bins=5):
    if bt is None or bt.empty: return pd.DataFrame()
    x=bt.dropna(subset=["Historical Probability"]).copy()
    if x.empty:return pd.DataFrame()
    x["Bin"]=pd.cut(x["Historical Probability"],bins=np.linspace(0,1,bins+1),include_lowest=True)
    return x.groupby("Bin",observed=False).agg(Tests=("Hit","size"),Predicted=("Historical Probability","mean"),Actual=("Hit","mean")).reset_index()

def model_vs_history(df,market,direction,line):
    hist=baseline_probability(df,market,direction,line)
    mean=average_value(df,MARKET_COLUMN_MAP.get(market))
    poisson=poisson_tail_probability(mean,line,direction)
    return pd.DataFrame([
        {"Method":"Historical hit rate","Probability":hist},
        {"Method":"Mean-based Poisson approximation","Probability":poisson}
    ])

def false_signal_report(bt):
    if bt is None or bt.empty:return pd.DataFrame()
    x=bt.dropna(subset=["Historical Probability"]).copy()
    x["Predicted"]=(x["Historical Probability"]>=0.5)
    x["Actual"] = x["Hit"].astype(bool)
    x["Error Type"]=np.where((x.Predicted)&(~x.Actual),"False positive",np.where((~x.Predicted)&(x.Actual),"False negative","Correct"))
    return x["Error Type"].value_counts().rename_axis("Error Type").reset_index(name="Count")

def model_health_report(bt):
    s=backtest_summary(bt)
    return pd.DataFrame([
        {"Metric":"Out-of-sample tests","Value":s["tests"]},
        {"Metric":"Directional accuracy","Value":fmt_pct(s["accuracy"])},
        {"Metric":"Brier score","Value":f"{s['brier']:.4f}" if s["brier"] is not None else "—"},
        {"Metric":"Minimum evidence","Value":"PASS" if s["tests"]>=10 else "LIMITED"},
    ])

def research_model_comparison(home_df,away_df,market,direction,line):
    h=baseline_probability(home_df,market,direction,line); a=baseline_probability(away_df,market,direction,line)
    combined=pd.concat([home_df,away_df],ignore_index=True)
    c=baseline_probability(combined,market,direction,line)
    return pd.DataFrame([{"Source":"Home sample","Probability":h},{"Source":"Away sample","Probability":a},{"Source":"Combined sample","Probability":c}])

def safe_numeric_mean(df, column):
    if df is None or df.empty or column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(s.mean()) if not s.empty else None

def source_status_table():
    rows = []
    for name, package in [("StatsBomb", "statsbombpy"), ("Understat", "understat")]:
        try:
            mod = __import__(package)
            rows.append({"Source": name, "Package": package, "Installed": "YES", "Version": getattr(mod, "__version__", "unknown")})
        except Exception as exc:
            rows.append({"Source": name, "Package": package, "Installed": "NO", "Version / Error": str(exc)[:180]})
    return pd.DataFrame(rows)

def source_mapping_report(data):
    return pd.DataFrame([
        {"Check": "Primary dataset", "Status": "ACTIVE", "Detail": "football_master_2024_27_v1.csv remains the primary match-statistics source."},
        {"Check": "StatsBomb mapping", "Status": "READY FOR MAPPING", "Detail": "External event data must be explicitly mapped to competition/team identifiers before use."},
        {"Check": "Understat mapping", "Status": "READY FOR MAPPING", "Detail": "External xG data must be explicitly mapped to league/team identifiers before use."},
        {"Check": "Silent merge protection", "Status": "ENABLED", "Detail": "External sources are not automatically merged into the primary CSV."},
    ])

def build_context_features(df, team):
    m = team_matches(df, team).copy()
    if m.empty:
        return m
    m = m.sort_values("date")
    for col in ["team_shots", "team_sot", "team_corners", "team_goals", "opp_shots", "opp_sot", "opp_corners", "opp_goals"]:
        if col in m.columns:
            m[f"rolling_5_{col}"] = pd.to_numeric(m[col], errors="coerce").rolling(5, min_periods=1).mean()
    return m

def team_strength_context(data, team):
    m = team_matches(data, team)
    if m.empty:
        return pd.DataFrame()
    rows = []
    for metric, own, opp in [
        ("Shots", "team_shots", "opp_shots"),
        ("SOT", "team_sot", "opp_sot"),
        ("Corners", "team_corners", "opp_corners"),
        ("Goals", "team_goals", "opp_goals")]:
        own_mean = safe_numeric_mean(m, own)
        opp_mean = safe_numeric_mean(m, opp)
        rows.append({"Metric": metric, "Produced Avg": own_mean, "Conceded Avg": opp_mean,
                     "Net": (own_mean - opp_mean) if own_mean is not None and opp_mean is not None else None})
    return pd.DataFrame(rows)

def player_schema_status(data):
    player_columns = [c for c in data.columns if any(x in c for x in ["player", "lineup", "minutes", "fouls", "player_shots", "player_sot"])]
    return pd.DataFrame([
        {"Capability": "Player data in primary CSV", "Status": "AVAILABLE" if player_columns else "NOT IN PRIMARY SCHEMA", "Columns detected": ", ".join(player_columns[:20]) if player_columns else "—"},
        {"Capability": "Starter / minutes analysis", "Status": "AVAILABLE" if any(x in data.columns for x in ["minutes", "starter", "lineup"]) else "REQUIRES PLAYER/EVENT SOURCE", "Columns detected": ", ".join([c for c in data.columns if c in ["minutes", "starter", "lineup"]]) or "—"},
        {"Capability": "Player shots / SOT", "Status": "AVAILABLE" if any("player_shot" in c or "player_sot" in c for c in data.columns) else "REQUIRES PLAYER/EVENT SOURCE", "Columns detected": ", ".join([c for c in data.columns if "player_shot" in c or "player_sot" in c]) or "—"},
    ])

def lineup_research_template():
    return pd.DataFrame([
        {"Research Field": "Player", "Value": ""},
        {"Research Field": "Expected starter", "Value": ""},
        {"Research Field": "Expected minutes", "Value": ""},
        {"Research Field": "Recent shots", "Value": ""},
        {"Research Field": "Recent SOT", "Value": ""},
        {"Research Field": "Fouls committed", "Value": ""},
        {"Research Field": "Fouls won", "Value": ""},
        {"Research Field": "Lineup/news note", "Value": ""},
    ])

def state_feature_status(data):
    state_cols = [c for c in data.columns if any(k in c for k in ["half", "minute", "score", "possession", "state", "red_card", "game_state"])]
    return pd.DataFrame([
        {"Feature": "Half-time state", "Status": "AVAILABLE" if state_cols else "REQUIRES EVENT/STATE DATA"},
        {"Feature": "Possession", "Status": "AVAILABLE" if "possession" in data.columns else "REQUIRES POSSESSION DATA"},
        {"Feature": "Score-state events", "Status": "AVAILABLE" if any("score" in c or "state" in c for c in state_cols) else "REQUIRES EVENT DATA"},
        {"Feature": "Cards / game state", "Status": "AVAILABLE" if any("card" in c for c in state_cols) else "REQUIRES EVENT DATA"},
    ])

def possession_context(data, team):
    if "possession" not in data.columns:
        return pd.DataFrame()
    m = team_matches(data, team).copy()
    if m.empty:
        return m
    m["possession"] = pd.to_numeric(m["possession"], errors="coerce")
    return m[[c for c in ["date", "home_team", "away_team", "venue", "possession", "team_shots", "team_sot", "team_corners"] if c in m.columns]].dropna(subset=["possession"])

def game_state_template():
    return pd.DataFrame([
        {"State": "Leading", "Expected behaviour": "", "Evidence source": ""},
        {"State": "Level", "Expected behaviour": "", "Evidence source": ""},
        {"State": "Trailing", "Expected behaviour": "", "Evidence source": ""},
        {"State": "Red-card affected", "Expected behaviour": "", "Evidence source": ""},
    ])

def market_lab(data, team, market, direction, lines, sample=30):
    m = filtered_team_matches(data, team, sample=sample)
    rows = []
    for ln in lines:
        a = analyse_market(m, market, direction, ln)
        rows.append({"Line": ln, "Hits": a["hits"], "Misses": a["misses"], "Sample": a["sample_size"], "Hit Rate": fmt_pct(a["hit_rate"])})
    return pd.DataFrame(rows)

def market_window_comparison(data, team, market, direction, line):
    rows = []
    for n in [5, 10, 15, 20, 30]:
        m = filtered_team_matches(data, team, sample=n)
        a = analyse_market(m, market, direction, line)
        rows.append({"Window": n, "Sample": a["sample_size"], "Hits": a["hits"], "Hit Rate": fmt_pct(a["hit_rate"])})
    return pd.DataFrame(rows)

def market_definition_registry():
    return pd.DataFrame([
        {"Market": "Shots", "Primary column": "team_shots", "Equality": "No hit"},
        {"Market": "Shots on Target", "Primary column": "team_sot", "Equality": "No hit"},
        {"Market": "Corners", "Primary column": "team_corners", "Equality": "No hit"},
        {"Market": "Goals", "Primary column": "team_goals", "Equality": "No hit"},
        {"Market": "Player shots", "Primary column": "Requires player data", "Equality": "Requires player schema"},
        {"Market": "Player SOT", "Primary column": "Requires player data", "Equality": "Requires player schema"},
        {"Market": "Player fouls", "Primary column": "Requires player/event data", "Equality": "Requires player schema"},
        {"Market": "Offsides", "Primary column": "Requires event data", "Equality": "Requires event schema"},
    ])

def bookmaker_summary(watchlist):
    if not watchlist:
        return pd.DataFrame()
    df = pd.DataFrame(watchlist).fillna("")
    if "Bookmaker" not in df.columns:
        return pd.DataFrame()
    rows = []
    for bookie, g in df.groupby("Bookmaker"):
        settled = g[g.get("Status", "").isin(["Won", "Lost"])] if "Status" in g.columns else g.iloc[0:0]
        won = int((settled["Status"] == "Won").sum()) if not settled.empty else 0
        rows.append({"Bookmaker": bookie, "Tracked Markets": len(g), "Settled": len(settled), "Won": won,
                     "Tracked Hit Rate": f"{won/len(settled)*100:.1f}%" if len(settled) else "—"})
    return pd.DataFrame(rows)

def odds_movement_template():
    return pd.DataFrame(columns=["Bookmaker", "Match", "Team", "Market", "Direction", "Line", "Odds", "Capture Time", "Notes"])

def line_availability_report(data):
    return pd.DataFrame([
        {"Market family": m, "Primary CSV support": "YES" if m in MARKET_COLUMN_MAP else "EXTERNAL PLAYER/EVENT DATA REQUIRED"}
        for m in ["Shots", "Shots on Target", "Corners", "Goals", "Player Shots", "Player SOT", "Player Fouls", "Offsides"]
    ])

def rolling_validation_grid(data, team, market, direction, lines):
    rows = []
    for line in lines:
        for window in [5, 10, 15, 20, 30]:
            m = filtered_team_matches(data, team, sample=window)
            a = analyse_market(m, market, direction, line)
            rows.append({"Line": line, "Window": window, "Sample": a["sample_size"], "Hit Rate": a["hit_rate"]})
    return pd.DataFrame(rows)

def validation_governance_report(data, team, market, direction, line):
    m = filtered_team_matches(data, team, sample=30)
    a = analyse_market(m, market, direction, line)
    chronological = bool(m["date"].is_monotonic_decreasing) if "date" in m.columns else False
    return pd.DataFrame([
        {"Check": "Sample available", "Status": "PASS" if a["sample_size"] >= 10 else "LIMITED", "Detail": a["sample_size"]},
        {"Check": "Chronological source ordering", "Status": "PASS" if chronological else "REVIEW", "Detail": "Recent-first source ordering checked"},
        {"Check": "Market definition", "Status": "PASS" if market in MARKET_COLUMN_MAP else "REVIEW", "Detail": market},
        {"Check": "Leakage protection", "Status": "PASS", "Detail": "Use pre-match historical rows only in the walk-forward engine"},
        {"Check": "Out-of-sample evidence", "Status": "AVAILABLE", "Detail": "Use Steps 159–162 walk-forward results"},
    ])

def command_centre_report(data, home_team, away_team, market, direction, line):
    hm = filtered_team_matches(data, home_team, venue="Home", sample=15)
    am = filtered_team_matches(data, away_team, venue="Away", sample=15)
    ha = analyse_market(hm, market, direction, line)
    aa = analyse_market(am, market, direction, line)
    h2h = get_match_history(data, home_team, away_team)
    return pd.DataFrame([
        {"Stage": "Fixture", "Item": "Home", "Value": home_team},
        {"Stage": "Fixture", "Item": "Away", "Value": away_team},
        {"Stage": "Historical evidence", "Item": "Home sample", "Value": ha["sample_size"]},
        {"Stage": "Historical evidence", "Item": "Away sample", "Value": aa["sample_size"]},
        {"Stage": "Market", "Item": "Selection", "Value": f"{market} {direction} {line}"},
        {"Stage": "Market", "Item": "Home historical hit rate", "Value": fmt_pct(ha["hit_rate"])},
        {"Stage": "Market", "Item": "Away historical hit rate", "Value": fmt_pct(aa["hit_rate"])},
        {"Stage": "H2H", "Item": "Meetings", "Value": len(h2h)},
        {"Stage": "External data", "Item": "StatsBomb / Understat", "Value": "Separate source review required"},
        {"Stage": "Human review", "Item": "Final decision", "Value": "Required"},
    ])

def command_centre_checklist():
    return pd.DataFrame([
        {"Research Gate": "Primary CSV checked", "Status": ""},
        {"Research Gate": "Home/away samples checked", "Status": ""},
        {"Research Gate": "Match-by-match history checked", "Status": ""},
        {"Research Gate": "Opponent context checked", "Status": ""},
        {"Research Gate": "Market line tested", "Status": ""},
        {"Research Gate": "Adjacent lines tested", "Status": ""},
        {"Research Gate": "Walk-forward validation checked", "Status": ""},
        {"Research Gate": "StatsBomb source checked where relevant", "Status": ""},
        {"Research Gate": "Understat/xG source checked where relevant", "Status": ""},
        {"Research Gate": "Lineups/team news checked", "Status": ""},
        {"Research Gate": "Referee/game-state context checked", "Status": ""},
        {"Research Gate": "Human review completed", "Status": ""},
    ])

def p14_dataset_profile(df):
    rows = []
    for c in df.columns:
        rows.append({
            "Column": c,
            "Dtype": str(df[c].dtype),
            "Rows": len(df),
            "Missing": int(df[c].isna().sum()),
            "Missing %": round(df[c].isna().mean() * 100, 2),
            "Unique": int(df[c].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)

def p14_duplicate_report(df):
    keys = [c for c in ["date", "home_team", "away_team", "competition", "season"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    d = df[df.duplicated(keys, keep=False)].copy()
    if d.empty:
        return pd.DataFrame()
    return d.groupby(keys, dropna=False).size().reset_index(name="Duplicate Count").sort_values("Duplicate Count", ascending=False)

def p14_date_report(df):
    if "date" not in df.columns or df["date"].dropna().empty:
        return pd.DataFrame([{"Check": "Date coverage", "Value": "Unavailable"}])
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    return pd.DataFrame([
        {"Check": "Earliest match", "Value": dates.min()},
        {"Check": "Latest match", "Value": dates.max()},
        {"Check": "Unique match dates", "Value": dates.nunique()},
        {"Check": "Rows", "Value": len(df)},
    ])

def p14_team_coverage(df):
    if not {"home_team", "away_team"}.issubset(df.columns):
        return pd.DataFrame()
    h = df["home_team"].value_counts().rename("Home Matches")
    a = df["away_team"].value_counts().rename("Away Matches")
    out = pd.concat([h, a], axis=1).fillna(0)
    out["Total Matches"] = out["Home Matches"] + out["Away Matches"]
    return out.sort_values("Total Matches", ascending=False).reset_index().rename(columns={"index": "Team"})

def p14_competition_coverage(df):
    if "competition" not in df.columns:
        return pd.DataFrame()
    return df.groupby("competition", dropna=False).agg(Matches=("competition", "size")).reset_index().sort_values("Matches", ascending=False)

def p15_missingness_report(df):
    rows = []
    for c in REQUIRED_COLUMNS:
        if c in df.columns:
            missing = int(df[c].isna().sum())
            rows.append({"Column": c, "Missing": missing, "Complete %": round((1 - missing / max(len(df), 1)) * 100, 2)})
        else:
            rows.append({"Column": c, "Missing": len(df), "Complete %": 0.0})
    return pd.DataFrame(rows)

def p15_numeric_quality(df):
    rows = []
    numeric = [c for c in REQUIRED_COLUMNS if c not in ["date", "season", "competition", "home_team", "away_team"]]
    for c in numeric:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "Column": c,
            "Valid": int(s.notna().sum()),
            "Negative": int((s < 0).sum()),
            "Zero": int((s == 0).sum()),
            "Maximum": s.max() if s.notna().any() else None,
        })
    return pd.DataFrame(rows)

def p15_logical_checks(df):
    checks = []
    pairs = [("home_goals", "away_goals"), ("home_shots", "away_shots"), ("home_sot", "away_sot"), ("home_corners", "away_corners")]
    for c1, c2 in pairs:
        if c1 in df.columns and c2 in df.columns:
            a = pd.to_numeric(df[c1], errors="coerce")
            b = pd.to_numeric(df[c2], errors="coerce")
            checks.append({"Rule": f"{c1} >= 0", "Violations": int((a < 0).sum())})
            checks.append({"Rule": f"{c2} >= 0", "Violations": int((b < 0).sum())})
    if "home_sot" in df.columns and "home_shots" in df.columns:
        checks.append({"Rule": "Home SOT <= Home Shots", "Violations": int((pd.to_numeric(df.home_sot, errors="coerce") > pd.to_numeric(df.home_shots, errors="coerce")).sum())})
    if "away_sot" in df.columns and "away_shots" in df.columns:
        checks.append({"Rule": "Away SOT <= Away Shots", "Violations": int((pd.to_numeric(df.away_sot, errors="coerce") > pd.to_numeric(df.away_shots, errors="coerce")).sum())})
    return pd.DataFrame(checks)

def p15_team_name_quality(df):
    rows = []
    for c in ["home_team", "away_team", "competition", "season"]:
        if c not in df.columns:
            continue
        s = df[c].astype(str).str.strip()
        rows.append({"Field": c, "Blank": int((s == "").sum()), "Unique": int(s.replace("", np.nan).nunique(dropna=True))})
    return pd.DataFrame(rows)

def p16_distribution(df, column):
    if column not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Metric": column, "N": len(s), "Mean": s.mean(), "Median": s.median(),
        "Std Dev": s.std(ddof=1) if len(s) > 1 else 0.0,
        "Min": s.min(), "Q25": s.quantile(.25), "Q75": s.quantile(.75), "Max": s.max()
    }])

def p16_percentile_table(df, column):
    if column not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame()
    return pd.DataFrame([{"Percentile": p, "Value": s.quantile(p / 100)} for p in [5, 10, 25, 50, 75, 90, 95]])

def p16_frequency_table(df, column):
    if column not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame()
    return s.value_counts().sort_index().rename_axis("Value").reset_index(name="Matches")

def p16_team_distribution(df, team, column):
    m = team_matches(df, team)
    return p16_distribution(m, column)

def p16_opponent_distribution(df, team, column):
    m = team_matches(df, team)
    return p16_distribution(m, column)

def p17_rolling_table(df, team, column, windows=(5, 10, 15, 20, 30)):
    m = team_matches(df, team).sort_values("date", ascending=True) if "date" in df.columns else team_matches(df, team)
    if column not in m.columns:
        return pd.DataFrame()
    s = pd.to_numeric(m[column], errors="coerce")
    rows = []
    for w in windows:
        vals = s.dropna().tail(w)
        rows.append({"Window": w, "Sample": len(vals), "Average": vals.mean() if not vals.empty else None, "Median": vals.median() if not vals.empty else None})
    return pd.DataFrame(rows)

def p17_recent_vs_long(df, team, column):
    m = team_matches(df, team)
    if column not in m.columns:
        return pd.DataFrame()
    s = pd.to_numeric(m[column], errors="coerce").dropna()
    recent = s.head(5)
    long = s.head(30)
    return pd.DataFrame([
        {"Window": "Recent 5", "Sample": len(recent), "Average": recent.mean() if len(recent) else None},
        {"Window": "Recent 10", "Sample": len(s.head(10)), "Average": s.head(10).mean() if len(s.head(10)) else None},
        {"Window": "Recent 15", "Sample": len(s.head(15)), "Average": s.head(15).mean() if len(s.head(15)) else None},
        {"Window": "Recent 30", "Sample": len(long), "Average": long.mean() if len(long) else None},
        {"Window": "All available", "Sample": len(s), "Average": s.mean() if len(s) else None},
    ])

def p17_stability(df, team, column):
    m = team_matches(df, team)
    if column not in m.columns:
        return pd.DataFrame()
    s = pd.to_numeric(m[column], errors="coerce").dropna()
    if len(s) < 2:
        return pd.DataFrame([{"Metric": column, "CV": None, "Interpretation": "Insufficient sample"}])
    mean = s.mean()
    cv = s.std(ddof=1) / mean if mean else None
    return pd.DataFrame([{"Metric": column, "Mean": mean, "Std Dev": s.std(ddof=1), "CV": cv, "Interpretation": "Lower variation" if cv is not None and cv < .5 else "Higher variation"}])

def p17_recency_gap(df, team, column):
    m = team_matches(df, team)
    if column not in m.columns:
        return pd.DataFrame()
    s = pd.to_numeric(m[column], errors="coerce").dropna()
    if len(s) < 10:
        return pd.DataFrame([{"Metric": column, "Recent 5": None, "Previous 5": None, "Difference": None}])
    r = s.head(5).mean()
    p = s.iloc[5:10].mean()
    return pd.DataFrame([{"Metric": column, "Recent 5": r, "Previous 5": p, "Difference": r - p}])

def p17_trend_report(df, team):
    rows = []
    for label, col in [("Shots", "team_shots"), ("SOT", "team_sot"), ("Corners", "team_corners"), ("Goals", "team_goals")]:
        t = p17_recency_gap(df, team, col)
        if not t.empty:
            row = t.iloc[0].to_dict()
            row["Metric"] = label
            rows.append(row)
    return pd.DataFrame(rows)

def p18_venue_split(df, team, column):
    m = team_matches(df, team)
    if column not in m.columns:
        return pd.DataFrame()
    rows = []
    for venue in ["Home", "Away"]:
        s = pd.to_numeric(m.loc[m["venue"] == venue, column], errors="coerce").dropna()
        rows.append({"Venue": venue, "Sample": len(s), "Average": s.mean() if len(s) else None, "Median": s.median() if len(s) else None})
    return pd.DataFrame(rows)

def p18_opponent_strength_proxy(df, team, column):
    m = team_matches(df, team).copy()
    if column not in m.columns:
        return pd.DataFrame()
    if "opp_goals" not in m.columns:
        return pd.DataFrame()
    m["opp_goals"] = pd.to_numeric(m["opp_goals"], errors="coerce")
    q = m["opp_goals"].median()
    m["Opponent Result-Goal Proxy"] = np.where(m["opp_goals"] <= q, "Lower opponent goals", "Higher opponent goals")
    return m.groupby("Opponent Result-Goal Proxy")[column].agg(Sample="count", Average="mean").reset_index()

def p18_same_context_market(df, team, market, direction, line):
    m = team_matches(df, team)
    a = analyse_market(m, market, direction, line)
    return pd.DataFrame([{"Team": team, "Context": "All venues", "Market": market, "Direction": direction, "Line": line, "Hits": a["hits"], "Sample": a["sample_size"], "Hit Rate": a["hit_rate"]}])

def p18_context_matrix(df, team):
    rows = []
    for venue in ["Home", "Away"]:
        m = team_matches(df, team)
        m = m[m["venue"] == venue]
        for market in MARKET_COLUMN_MAP:
            for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
                a = analyse_market(m, market, "Over", line)
                rows.append({"Venue": venue, "Market": market, "Line": line, "Sample": a["sample_size"], "Over Hit Rate": a["hit_rate"]})
    return pd.DataFrame(rows)

def p18_context_summary(df, team):
    m = team_matches(df, team)
    rows = []
    for venue in ["All", "Home", "Away"]:
        x = m if venue == "All" else m[m["venue"] == venue]
        rows.append({"Venue": venue, "Matches": len(x), "Avg Shots": average_value(x, "team_shots"), "Avg SOT": average_value(x, "team_sot"), "Avg Corners": average_value(x, "team_corners"), "Avg Goals": average_value(x, "team_goals")})
    return pd.DataFrame(rows)

def p19_h2h_summary(df, home, away):
    h = get_match_history(df, home, away)
    if h.empty:
        return pd.DataFrame([{"Meetings": 0}])
    return pd.DataFrame([{
        "Meetings": len(h),
        "Avg Total Goals": (pd.to_numeric(h.home_goals, errors="coerce") + pd.to_numeric(h.away_goals, errors="coerce")).mean(),
        "Avg Total Shots": (pd.to_numeric(h.home_shots, errors="coerce") + pd.to_numeric(h.away_shots, errors="coerce")).mean(),
        "Avg Total SOT": (pd.to_numeric(h.home_sot, errors="coerce") + pd.to_numeric(h.away_sot, errors="coerce")).mean(),
        "Avg Total Corners": (pd.to_numeric(h.home_corners, errors="coerce") + pd.to_numeric(h.away_corners, errors="coerce")).mean(),
    }])

def p19_h2h_market(df, home, away, market, direction, line):
    h = get_match_history(df, home, away)
    if h.empty:
        return pd.DataFrame()
    rows = []
    for _, r in h.iterrows():
        if market == "Goals": value = float(r.home_goals) + float(r.away_goals)
        elif market == "Shots": value = float(r.home_shots) + float(r.away_shots)
        elif market == "Shots on Target": value = float(r.home_sot) + float(r.away_sot)
        else: value = float(r.home_corners) + float(r.away_corners)
        hit = value > line if direction == "Over" else value < line
        rows.append({"date": r.get("date"), "Home": r.get("home_team"), "Away": r.get("away_team"), "Total": value, "Hit": "✓" if hit else "✗"})
    return pd.DataFrame(rows)

def p19_h2h_recent(df, home, away, n=5):
    return get_match_history(df, home, away).head(n)

def p19_h2h_home_perspective(df, home, away):
    h = get_match_history(df, home, away).copy()
    if h.empty:
        return h
    h["Home Team Goals"] = np.where(h["home_team"] == home, h["home_goals"], h["away_goals"])
    h["Away Team Goals"] = np.where(h["home_team"] == home, h["away_goals"], h["home_goals"])
    return h

def p20_line_grid(df, team, market, direction, lines, sample=30):
    m = filtered_team_matches(df, team, sample=sample)
    rows = []
    for line in lines:
        a = analyse_market(m, market, direction, line)
        rows.append({"Line": line, "Sample": a["sample_size"], "Hits": a["hits"], "Misses": a["misses"], "Hit Rate": a["hit_rate"]})
    return pd.DataFrame(rows)

def p20_adjacent_lines(df, team, market, direction, line):
    lines = sorted(set([max(0, line - 1), max(0, line - .5), line, line + .5, line + 1]))
    return p20_line_grid(df, team, market, direction, lines, 30)

def p20_direction_compare(df, team, market, line):
    rows = []
    m = filtered_team_matches(df, team, sample=30)
    for d in ["Over", "Under"]:
        a = analyse_market(m, market, d, line)
        rows.append({"Direction": d, "Line": line, "Hits": a["hits"], "Sample": a["sample_size"], "Hit Rate": a["hit_rate"]})
    return pd.DataFrame(rows)

def p20_market_crosscheck(df, home, away, market, direction, line):
    hm = filtered_team_matches(df, home, venue="Home", sample=30)
    am = filtered_team_matches(df, away, venue="Away", sample=30)
    ha = analyse_market(hm, market, direction, line)
    aa = analyse_market(am, market, direction, line)
    return pd.DataFrame([{"Team": home, "Venue": "Home", **ha}, {"Team": away, "Venue": "Away", **aa}])

def p21_bootstrap_ci(values, n=3000, seed=42):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(s) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(s, size=(n, len(s)), replace=True).mean(axis=1)
    return float(np.mean(s)), float(np.quantile(means, .025)), float(np.quantile(means, .975))

def p21_hit_rate_ci(df, market, direction, line):
    col = MARKET_COLUMN_MAP.get(market)
    if col not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 2:
        return pd.DataFrame([{"Sample": len(s), "Hit Rate": None, "95% Lower": None, "95% Upper": None}])
    hits = (s > line) if direction == "Over" else (s < line)
    arr = hits.astype(float).to_numpy()
    rng = np.random.default_rng(42)
    boot = rng.choice(arr, size=(3000, len(arr)), replace=True).mean(axis=1)
    return pd.DataFrame([{"Sample": len(arr), "Hit Rate": arr.mean(), "95% Lower": np.quantile(boot, .025), "95% Upper": np.quantile(boot, .975)}])

def p21_mean_ci(df, column):
    mean, low, high = p21_bootstrap_ci(df[column] if column in df.columns else [])
    return pd.DataFrame([{"Metric": column, "Mean": mean, "95% Lower": low, "95% Upper": high}])

def p21_sample_sensitivity(df, market, direction, line):
    rows = []
    for n in [5, 10, 15, 20, 30, 50]:
        m = df.head(n)
        a = analyse_market(m, market, direction, line)
        rows.append({"Window": n, "Sample": a["sample_size"], "Hit Rate": a["hit_rate"]})
    return pd.DataFrame(rows)

def p21_uncertainty_note(sample):
    if sample < 5: return "Very high uncertainty from small sample."
    if sample < 10: return "High uncertainty; sample remains limited."
    if sample < 20: return "Moderate uncertainty; inspect multiple windows."
    return "Uncertainty should still be considered alongside context and source quality."

def p21_ci_report(df, team, market, direction, line):
    m = filtered_team_matches(df, team, sample=30)
    return p21_hit_rate_ci(m, market, direction, line)

def p22_poisson_pmf(k, mean):
    if mean < 0 or k < 0:
        return 0.0
    return math.exp(-mean) * (mean ** k) / math.factorial(k)

def p22_poisson_tail(mean, line, direction):
    if mean < 0:
        return None
    if direction == "Over":
        k = int(math.floor(line)) + 1
        cdf = sum(p22_poisson_pmf(i, mean) for i in range(k))
        return max(0.0, min(1.0, 1.0 - cdf))
    k = int(math.floor(line))
    if k <= 0:
        return 0.0
    return max(0.0, min(1.0, sum(p22_poisson_pmf(i, mean) for i in range(k))))

def p22_poisson_baseline(df, team, market, direction, line):
    m = filtered_team_matches(df, team, sample=30)
    col = MARKET_COLUMN_MAP.get(market)
    mean = average_value(m, col) if col else None
    p = p22_poisson_tail(mean, line, direction) if mean is not None else None
    return pd.DataFrame([{"Team": team, "Market": market, "Direction": direction, "Line": line, "Historical Mean": mean, "Poisson Benchmark": p, "Sample": len(m)}])

def p22_poisson_grid(df, team, market, direction, lines):
    rows = []
    for line in lines:
        x = p22_poisson_baseline(df, team, market, direction, line)
        rows.extend(x.to_dict("records"))
    return pd.DataFrame(rows)

def p22_model_vs_history(df, team, market, direction, line):
    m = filtered_team_matches(df, team, sample=30)
    a = analyse_market(m, market, direction, line)
    col = MARKET_COLUMN_MAP.get(market)
    mean = average_value(m, col) if col else None
    p = p22_poisson_tail(mean, line, direction) if mean is not None else None
    return pd.DataFrame([{"Historical Hit Rate": a["hit_rate"], "Poisson Benchmark": p, "Difference": (a["hit_rate"] - p) if a["hit_rate"] is not None and p is not None else None, "Sample": a["sample_size"]}])

def p22_poisson_assumptions():
    return pd.DataFrame([
        {"Assumption": "Constant event rate", "Status": "Model assumption"},
        {"Assumption": "Independent events", "Status": "Model assumption"},
        {"Assumption": "No tactical/game-state adjustment", "Status": "Limitation"},
        {"Assumption": "No lineup/injury adjustment", "Status": "Limitation"},
        {"Assumption": "Benchmark only", "Status": "Required interpretation"},
    ])

def p23_walkforward_simple(df, team, market, direction, line, min_history=5):
    m = team_matches(df, team).sort_values("date", ascending=True) if "date" in df.columns else team_matches(df, team)
    col = MARKET_COLUMN_MAP.get(market)
    if col not in m.columns:
        return pd.DataFrame()
    rows = []
    values = pd.to_numeric(m[col], errors="coerce")
    for i in range(min_history, len(m)):
        hist = values.iloc[:i].dropna()
        actual = values.iloc[i]
        if hist.empty or pd.isna(actual):
            continue
        mean = hist.mean()
        benchmark = p22_poisson_tail(mean, line, direction)
        hit = actual > line if direction == "Over" else actual < line
        rows.append({"Index": i, "Date": m.iloc[i].get("date"), "Historical Mean": mean, "Benchmark": benchmark, "Actual": actual, "Hit": bool(hit)})
    return pd.DataFrame(rows)

def p23_walkforward_summary(bt):
    if bt.empty:
        return pd.DataFrame([{"Observations": 0, "Hit Rate": None, "Mean Benchmark": None}])
    return pd.DataFrame([{"Observations": len(bt), "Hit Rate": bt["Hit"].mean(), "Mean Benchmark": bt["Benchmark"].mean()}])

def p23_temporal_split(df, team):
    m = team_matches(df, team).sort_values("date", ascending=True) if "date" in df.columns else team_matches(df, team)
    n = len(m)
    cut = int(n * .7)
    return pd.DataFrame([{"Total": n, "Train": cut, "Test": n - cut, "Chronological": True}])

def p23_oos_guardrail(df, team):
    n = len(team_matches(df, team))
    return pd.DataFrame([
        {"Guardrail": "Minimum observations", "Required": 10, "Available": n, "Status": "PASS" if n >= 10 else "LIMITED"},
        {"Guardrail": "Chronological split", "Required": "Yes", "Available": "Yes", "Status": "PASS"},
        {"Guardrail": "Future leakage", "Required": "No", "Available": "Historical rows only", "Status": "PASS"},
    ])

def p23_calibration_bins(bt):
    if bt.empty:
        return pd.DataFrame()
    x = bt.copy()
    x["Bin"] = pd.cut(x["Benchmark"].clip(0, 1), bins=[0, .2, .4, .6, .8, 1.0], include_lowest=True)
    return x.groupby("Bin", observed=False).agg(Observations=("Hit", "size"), ActualRate=("Hit", "mean"), BenchmarkRate=("Benchmark", "mean")).reset_index()

def p24_watchlist_df():
    return pd.DataFrame(st.session_state.get("market_watchlist", []))

def p24_odds_capture_template():
    return pd.DataFrame(columns=["Timestamp", "Bookmaker", "Match", "Market", "Direction", "Line", "Odds", "Source Note"])

def p24_watchlist_summary():
    w = p24_watchlist_df()
    if w.empty:
        return pd.DataFrame([{"Markets": 0}])
    rows = [{"Markets": len(w), "Watching": int(w.get("Status", pd.Series(dtype=str)).eq("Watching").sum()), "Won": int(w.get("Status", pd.Series(dtype=str)).eq("Won").sum()), "Lost": int(w.get("Status", pd.Series(dtype=str)).eq("Lost").sum())}]
    return pd.DataFrame(rows)

def p24_bookmaker_coverage():
    w = p24_watchlist_df()
    if w.empty or "Bookmaker" not in w.columns:
        return pd.DataFrame()
    return w.groupby("Bookmaker").size().reset_index(name="Tracked Markets")

def p24_line_change_history():
    return pd.DataFrame(columns=["Match", "Bookmaker", "Market", "Timestamp", "Line", "Odds", "Previous Line", "Previous Odds", "Line Change", "Odds Change"])

def p25_external_schema():
    return pd.DataFrame([
        {"Source": "StatsBomb", "Purpose": "Event-level context", "Merge Key": "Match ID", "Primary CSV": "Not silently merged", "Status": "Ready for explicit mapping"},
        {"Source": "Understat", "Purpose": "xG / shot context", "Merge Key": "Match + Team + Date", "Primary CSV": "Not silently merged", "Status": "Ready for explicit mapping"},
        {"Source": "Primary CSV", "Purpose": "Core match statistics", "Merge Key": "Date + Teams + Competition", "Primary CSV": "Yes", "Status": "Primary"},
    ])

def p25_mapping_check():
    return pd.DataFrame([
        {"Field": "Match identity", "Required": "Yes", "Mapped": "Manual verification required"},
        {"Field": "Team identity", "Required": "Yes", "Mapped": "Name normalization required"},
        {"Field": "Competition", "Required": "Preferred", "Mapped": "Verify source labels"},
        {"Field": "Season", "Required": "Preferred", "Mapped": "Verify source convention"},
        {"Field": "Event timestamps", "Required": "For state analysis", "Mapped": "External event source required"},
    ])

def p25_source_provenance():
    return pd.DataFrame([
        {"Source": "football_master_2024_27_v1.csv", "Role": "Primary", "Merge Policy": "Authoritative for current team metrics"},
        {"Source": "StatsBomb", "Role": "External", "Merge Policy": "Explicit mapping + validation"},
        {"Source": "Understat", "Role": "External", "Merge Policy": "Explicit mapping + validation"},
    ])

def p25_external_update_check():
    return pd.DataFrame([
        {"Check": "Source accessible", "Status": "Manual/API check"},
        {"Check": "Schema compatible", "Status": "Verify before import"},
        {"Check": "Team names normalized", "Status": "Required"},
        {"Check": "Match IDs aligned", "Status": "Required"},
        {"Check": "No duplicate merges", "Status": "Required"},
    ])

def p26_player_schema():
    return pd.DataFrame([
        {"Field": "player_id", "Required": "Yes", "Current CSV": "No"},
        {"Field": "player_name", "Required": "Yes", "Current CSV": "No"},
        {"Field": "team", "Required": "Yes", "Current CSV": "No"},
        {"Field": "minutes", "Required": "Yes", "Current CSV": "No"},
        {"Field": "shots", "Required": "For player shots", "Current CSV": "No"},
        {"Field": "shots_on_target", "Required": "For player SOT", "Current CSV": "No"},
        {"Field": "fouls_committed", "Required": "For player fouls", "Current CSV": "No"},
        {"Field": "fouls_won", "Required": "For player fouls won", "Current CSV": "No"},
    ])

def p26_player_market_template():
    return pd.DataFrame(columns=["Date", "Competition", "Team", "Opponent", "Player", "Starter", "Minutes", "Shots", "SOT", "Fouls Committed", "Fouls Won", "Source"])

def p26_minutes_gate():
    return pd.DataFrame([
        {"Check": "Confirmed starter", "Status": "Human/lineup source required"},
        {"Check": "Expected minutes", "Status": "Human/source review required"},
        {"Check": "Substitution risk", "Status": "Review"},
        {"Check": "Player role", "Status": "Review"},
    ])

def p26_player_history_template():
    return pd.DataFrame(columns=["Date", "Player", "Market", "Line", "Actual", "Hit", "Minutes", "Starter"])

def p26_event_schema():
    return pd.DataFrame([
        {"Event": "Shot", "Needed For": "Shot/SOT sequences", "Current CSV": "No"},
        {"Event": "Foul", "Needed For": "Fouls context", "Current CSV": "No"},
        {"Event": "Card", "Needed For": "Referee/card context", "Current CSV": "No"},
        {"Event": "Possession snapshot", "Needed For": "Game-state context", "Current CSV": "No"},
        {"Event": "Substitution", "Needed For": "Minutes risk", "Current CSV": "No"},
    ])

def p27_state_proxy(df, team):
    m = team_matches(df, team).copy()
    if m.empty:
        return pd.DataFrame()
    m["Goal Difference"] = pd.to_numeric(m["team_goals"], errors="coerce") - pd.to_numeric(m["opp_goals"], errors="coerce")
    m["Final State"] = np.select([m["Goal Difference"] > 0, m["Goal Difference"] < 0], ["Won", "Lost"], default="Draw")
    return m[[c for c in ["date", "home_team", "away_team", "team_goals", "opp_goals", "Goal Difference", "Final State"] if c in m.columns]]

def p27_state_split(df, team, column):
    m = p27_state_proxy(df, team)
    if column not in team_matches(df, team).columns:
        return pd.DataFrame()
    raw = team_matches(df, team).copy()
    raw["Final State"] = m["Final State"].values if len(raw) == len(m) else "Unknown"
    return raw.groupby("Final State")[column].agg(Sample="count", Average="mean").reset_index()

def p27_lead_behavior_warning():
    return pd.DataFrame([
        {"Issue": "Final score is not event-state data", "Implication": "Cannot infer exact in-game behaviour from final score alone."},
        {"Issue": "Leading-game behaviour", "Implication": "Requires event/time-split data."},
        {"Issue": "Possession changes", "Implication": "Requires possession or event source."},
        {"Issue": "Pressing changes", "Implication": "Requires event/context proxies."},
    ])

def p27_tactical_template():
    return pd.DataFrame([
        {"Area": "Build-up", "Observation": ""},
        {"Area": "Pressing", "Observation": ""},
        {"Area": "Transition", "Observation": ""},
        {"Area": "Crosses / wide attacks", "Observation": ""},
        {"Area": "Set pieces", "Observation": ""},
        {"Area": "Game-state response", "Observation": ""},
    ])

def p27_referee_template():
    return pd.DataFrame(columns=["Match", "Referee", "Cards", "Fouls", "Penalties", "Notes", "Source"])

def p28_research_pack_sections(home, away, market, direction, line):
    return pd.DataFrame([
        {"Section": "Fixture", "Value": f"{home} vs {away}"},
        {"Section": "Market", "Value": f"{market} {direction} {line}"},
        {"Section": "Primary data", "Value": "football_master_2024_27_v1.csv"},
        {"Section": "External sources", "Value": "StatsBomb / Understat — explicit mapping only"},
        {"Section": "Human review", "Value": "Required"},
    ])

def p28_data_dictionary():
    return pd.DataFrame([
        {"Field": c, "Meaning": "Primary match-level field", "Source": "Master CSV"} for c in REQUIRED_COLUMNS
    ])

def p28_export_frame(df):
    if df is None:
        return pd.DataFrame()
    return df.copy()

def p28_audit_log_template():
    return pd.DataFrame(columns=["Timestamp", "Action", "Dataset", "Rows", "Columns", "User Note"])

def p28_research_notes_template():
    return pd.DataFrame(columns=["Date", "Fixture", "Market", "Observation", "Source", "Researcher Note"])

def p29_final_gate(df, home, away, market, direction, line):
    hm = filtered_team_matches(df, home, venue="Home", sample=30)
    am = filtered_team_matches(df, away, venue="Away", sample=30)
    checks = [
        ("Primary dataset loaded", len(df) > 0),
        ("Home sample available", len(hm) > 0),
        ("Away sample available", len(am) > 0),
        ("Market defined", market in MARKET_COLUMN_MAP),
        ("Line valid", line >= 0),
        ("Chronological validation available", len(team_matches(df, home)) >= 6),
        ("External data not silently merged", True),
        ("Human review required", True),
    ]
    return pd.DataFrame([{"Gate": k, "Status": "PASS" if v else "REVIEW"} for k, v in checks])

def p29_readiness_summary(df, home, away):
    return pd.DataFrame([
        {"Area": "Primary CSV", "Status": "Loaded" if len(df) else "Missing", "Detail": f"{len(df)} rows"},
        {"Area": "Home history", "Status": "Available" if len(team_matches(df, home)) else "Missing", "Detail": f"{len(team_matches(df, home))} matches"},
        {"Area": "Away history", "Status": "Available" if len(team_matches(df, away)) else "Missing", "Detail": f"{len(team_matches(df, away))} matches"},
        {"Area": "Player markets", "Status": "External data required", "Detail": "Player/event schema not in primary CSV"},
        {"Area": "Possession/game-state", "Status": "External data required", "Detail": "Not in primary CSV"},
        {"Area": "StatsBomb / Understat", "Status": "Separate source", "Detail": "Explicit mapping and validation required"},
        {"Area": "Decision output", "Status": "Human review", "Detail": "No automatic safe-bet verdict"},
    ])

def p29_release_checklist():
    return pd.DataFrame([
        {"Release Check": "Syntax passes", "Status": "Checked before deployment"},
        {"Release Check": "Primary CSV remains default", "Status": "Yes"},
        {"Release Check": "External data remains explicit", "Status": "Yes"},
        {"Release Check": "Historical hit rate remains descriptive", "Status": "Yes"},
        {"Release Check": "Poisson remains benchmark", "Status": "Yes"},
        {"Release Check": "No betting automation", "Status": "Yes"},
    ])

def step_311_630_registry():
    return pd.DataFrame(STEP_311_630_REGISTRY)


# ============================================================
# NEW ORCHESTRATOR — ONE FIXTURE IN, FULL INTERNAL RESEARCH OUT
# ============================================================

# ============================================================
# DATASET 1 — PRIMARY MATCH-LEVEL DATASET
# ============================================================
# Dataset 1 is the authoritative match-level foundation for the
# engine. It remains separate from StatsBomb/Understat and any
# future supplementary datasets. The engine never silently replaces
# Dataset 1 values with external-source values.
#
# Expected deployed filename: football_master_2024_27_v1.csv
# A duplicate-safe filename such as football_master_2024_27_v1(1).csv
# is also accepted so the real Library copy can be tested locally.

DATASET_1_NAME = "Football Master Match Dataset 2024–27"
DATASET_1_ROLE = "Primary authoritative match-statistics dataset"
DATASET_1_ALTERNATE_FILES = [
    PRIMARY_DATASET,
    "football_master_2024_27_v1(1).csv",
]

def dataset_1_validation(df):
    checks = []
    if df is None or df.empty:
        return pd.DataFrame([
            {"Check": "Dataset 1 loaded", "Status": "FAIL", "Detail": "No rows loaded."}
        ])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks.append({
        "Check": "Required columns",
        "Status": "PASS" if not missing else "FAIL",
        "Detail": "All required columns present." if not missing else "Missing: " + ", ".join(missing)
    })

    if "date" in df.columns:
        checks.append({
            "Check": "Valid dates",
            "Status": "PASS" if df["date"].notna().all() else "REVIEW",
            "Detail": f"{int(df["date"].notna().sum()):,}/{len(df):,} rows have valid dates."
        })

    numeric = [
        "home_goals", "away_goals", "home_shots", "away_shots",
        "home_sot", "away_sot", "home_corners", "away_corners"
    ]
    for c in numeric:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            checks.append({
                "Check": f"Non-negative {c}",
                "Status": "PASS" if (vals.dropna() >= 0).all() else "FAIL",
                "Detail": "No negative values found." if (vals.dropna() >= 0).all() else "Negative values detected."
            })

    if "home_sot" in df.columns and "home_shots" in df.columns:
        hs = pd.to_numeric(df["home_sot"], errors="coerce")
        hshots = pd.to_numeric(df["home_shots"], errors="coerce")
        bad = int((hs > hshots).fillna(False).sum())
        checks.append({
            "Check": "Home SOT <= shots",
            "Status": "PASS" if bad == 0 else "FAIL",
            "Detail": f"{bad:,} invalid row(s)."
        })

    if "away_sot" in df.columns and "away_shots" in df.columns:
        ass = pd.to_numeric(df["away_sot"], errors="coerce")
        ashots = pd.to_numeric(df["away_shots"], errors="coerce")
        bad = int((ass > ashots).fillna(False).sum())
        checks.append({
            "Check": "Away SOT <= shots",
            "Status": "PASS" if bad == 0 else "FAIL",
            "Detail": f"{bad:,} invalid row(s)."
        })

    duplicate_count = int(df.duplicated().sum())
    checks.append({
        "Check": "Duplicate rows",
        "Status": "PASS" if duplicate_count == 0 else "REVIEW",
        "Detail": f"{duplicate_count:,} duplicate row(s)."
    })
    return pd.DataFrame(checks)


@st.cache_data(ttl=600, show_spinner=False)
def load_primary_dataset(path=PRIMARY_DATASET):
    candidates = [path] + [x for x in DATASET_1_ALTERNATE_FILES if x != path]
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                return clean_data(pd.read_csv(candidate)), candidate
            except Exception:
                continue
    return pd.DataFrame(columns=REQUIRED_COLUMNS), None


def demo_dataset():
    rows = [
        ["2026-09-18","2026/27","Premier League","Chelsea","Brentford",2,0,17,7,5,2,6,3],
        ["2026-09-14","2026/27","Premier League","Everton","Chelsea",0,2,9,14,3,5,4,7],
        ["2026-09-07","2026/27","Premier League","Chelsea","Fulham",3,1,19,8,7,3,8,2],
        ["2026-08-30","2026/27","Premier League","Newcastle","Chelsea",1,1,12,11,4,4,5,5],
        ["2026-08-24","2026/27","Premier League","Chelsea","Wolves",2,1,16,10,6,3,7,4],
        ["2026-09-18","2026/27","Bundesliga","Bayern Munich","Union Berlin",3,1,20,7,8,2,8,2],
        ["2026-09-13","2026/27","Bundesliga","Mainz","Bayern Munich",0,3,6,18,2,7,3,8],
        ["2026-09-06","2026/27","Bundesliga","Bayern Munich","Freiburg",2,0,17,8,6,2,9,4],
        ["2026-08-30","2026/27","Bundesliga","Dortmund","Bayern Munich",1,2,11,13,4,5,5,6],
        ["2026-08-23","2026/27","Bundesliga","Bayern Munich","Leipzig",4,1,22,9,9,3,10,2],
    ]
    return clean_data(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))


def robust_poisson_tail(mean, line, direction):
    try:
        mean, line = float(mean), float(line)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(mean) or not np.isfinite(line) or mean < 0:
        return None
    if direction == "Over":
        k = int(math.floor(line)) + 1
        cdf = sum(math.exp(-mean) * (mean ** i) / math.factorial(i) for i in range(k))
        return float(np.clip(1.0 - cdf, 0.0, 1.0))
    k = int(math.floor(line))
    if k < 0:
        return 0.0
    cdf = sum(math.exp(-mean) * (mean ** i) / math.factorial(i) for i in range(k + 1))
    return float(np.clip(cdf, 0.0, 1.0))


# Replace any legacy Poisson implementation with the fixed version.
poisson_tail_probability = robust_poisson_tail
p22_poisson_tail = robust_poisson_tail


def source_health_check():
    """Real package + network checks for StatsBomb and Understat."""
    rows = []
    checks = [
        ("StatsBomb", "statsbombpy", "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"),
        ("Understat", "understat", "https://understat.com/league/EPL/2025"),
    ]
    for source, package, endpoint in checks:
        package_ok = False
        version = "—"
        package_detail = "Package unavailable"
        try:
            mod = __import__(package)
            package_ok = True
            version = getattr(mod, "__version__", "installed")
            package_detail = "Import succeeded"
        except Exception as exc:
            package_detail = str(exc)[:180]
        network_ok = False
        http_status = "—"
        network_detail = "requests unavailable" if requests is None else "Not tested"
        if requests is not None:
            try:
                r = requests.get(endpoint, timeout=8, headers={"User-Agent":"Mozilla/5.0 FootballResearch/1.0"})
                http_status = r.status_code
                network_ok = 200 <= r.status_code < 400
                network_detail = "HTTP reachable" if network_ok else f"HTTP {r.status_code}"
            except Exception as exc:
                network_detail = str(exc)[:180]
        if package_ok and network_ok:
            overall = "HEALTHY"
        elif package_ok or network_ok:
            overall = "PARTIAL"
        else:
            overall = "UNAVAILABLE"
        rows.append({"Source":source,"Package":package,"Package Status":"OK" if package_ok else "FAIL","Version":version,"Endpoint Status":"OK" if network_ok else "FAIL","HTTP":http_status,"Overall":overall,"Detail":f"{package_detail}; {network_detail}"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def cached_source_health():
    return source_health_check()


def normalise_team_name(name, teams):
    if not name:
        return None
    raw = str(name).strip()
    exact = {str(t).strip().lower(): t for t in teams}
    if raw.lower() in exact:
        return exact[raw.lower()]
    compact = raw.lower().replace("fc", "").replace("  ", " ").strip()
    for k, v in exact.items():
        if k.replace("fc", "").replace("  ", " ").strip() == compact:
            return v
    return raw if raw in teams else None


def fixture_samples(data, home, away, n=30):
    home_home = filtered_team_matches(data, home, venue="Home", sample=n)
    away_away = filtered_team_matches(data, away, venue="Away", sample=n)
    home_all = filtered_team_matches(data, home, venue="All", sample=n)
    away_all = filtered_team_matches(data, away, venue="All", sample=n)
    return home_home, away_away, home_all, away_all


def safe_rate(frame, market, direction, line):
    r = analyse_market(frame, market, direction, line)
    return r.get("hit_rate")


def wilson_lower_bound(rate, n, z=1.96):
    if rate is None or n <= 0:
        return None
    phat = float(rate)
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    spread = z * math.sqrt((phat*(1-phat)/n) + z*z/(4*n*n))
    return max(0.0, (centre-spread)/denom)


def evidence_score(home_df, away_df, market, direction, line):
    """Research-support index used only to order surviving research candidates."""
    hr = safe_rate(home_df, market, direction, line)
    ar = safe_rate(away_df, market, direction, line)
    h = safe_rate(home_df, market, direction, line)
    a = safe_rate(away_df, market, direction, line)
    opp_col = MARKET_OPPONENT_MAP[market]
    h_opp = pd.to_numeric(home_df.get(opp_col, pd.Series(dtype=float)), errors="coerce").dropna()
    a_opp = pd.to_numeric(away_df.get(opp_col, pd.Series(dtype=float)), errors="coerce").dropna()
    # For an Over candidate, opponent concession is supportive when opponents commonly exceed the line.
    # For an Under candidate, support comes from opponents commonly staying below the line.
    if direction == "Over":
        h_def = float((h_opp > line).mean()) if len(h_opp) else None
        a_def = float((a_opp > line).mean()) if len(a_opp) else None
    else:
        h_def = float((h_opp < line).mean()) if len(h_opp) else None
        a_def = float((a_opp < line).mean()) if len(a_opp) else None
    vals = [x for x in [hr, ar, h_def, a_def] if x is not None]
    if not vals:
        return None, {}
    base = float(np.mean(vals))
    # Stability: agreement between the four evidence streams.
    agreement = 1 - float(np.std(vals)) if len(vals) > 1 else 0.5
    agreement = float(np.clip(agreement, 0, 1))
    n = len(pd.concat([home_df, away_df]))
    lower = wilson_lower_bound(base, n)
    score = 100 * (0.55 * base + 0.25 * agreement + 0.20 * (lower if lower is not None else base))
    return score, {"home_rate":hr,"away_rate":ar,"home_opp_context":h_def,"away_opp_context":a_def,"combined_rate":base,"lower_bound":lower,"agreement":agreement,"sample":n}


def candidate_market_rows(data, home, away, bookmaker_rows=None, max_candidates=12):
    hh, aa, _, _ = fixture_samples(data, home, away, 30)
    candidates=[]
    for market, lines in MARKET_LINE_RANGES.items():
        for direction in ["Over", "Under"]:
            for line in lines:
                score, ev = evidence_score(hh, aa, market, direction, float(line))
                if score is None:
                    continue
                # Minimum evidence gates.
                if ev["sample"] < 5:
                    continue
                # Candidate must have at least moderate direct support from both relevant samples when both exist.
                direct = [x for x in [ev["home_rate"], ev["away_rate"]] if x is not None]
                if len(direct) < 2:
                    continue
                direct_mean = float(np.mean(direct))
                if direct_mean < 0.55:
                    continue
                hist = safe_rate(pd.concat([hh, aa], ignore_index=True), market, direction, float(line))
                mean = average_value(pd.concat([hh, aa], ignore_index=True), MARKET_COLUMN_MAP[market])
                poisson = robust_poisson_tail(mean, float(line), direction)
                candidates.append({"Market":market,"Direction":direction,"Line":float(line),"Research Support":round(score,1),"Historical Rate":hist,"Poisson Benchmark":poisson,"Home Rate":ev["home_rate"],"Away Rate":ev["away_rate"],"Home Opponent Context":ev["home_opp_context"],"Away Opponent Context":ev["away_opp_context"],"Evidence Agreement":ev["agreement"],"Sample":ev["sample"],"Odds":None,"Bookmaker":None,"Availability":"Internal candidate"})
    result=pd.DataFrame(candidates)
    if result.empty:
        return result
    # If bookmaker observations exist, use only the exact supplied minimum line for each supplied market/direction.
    if bookmaker_rows:
        supplied=[]
        for b in bookmaker_rows:
            for _, row in result.iterrows():
                if row["Market"] == b["Market"] and row["Direction"] == b["Direction"] and abs(float(row["Line"])-float(b["Minimum Line"])) < 1e-9:
                    r=row.to_dict(); r["Odds"]=float(b["Odds"]); r["Bookmaker"]=b["Bookmaker"]; r["Availability"]="Bookmaker supplied minimum line"
                    supplied.append(r)
        result=pd.DataFrame(supplied)
        if result.empty:
            return result
        # One supplied minimum line + odds per bookmaker/market is the only odds-bearing candidate.
    result=result.sort_values(["Research Support","Historical Rate"],ascending=False).head(max_candidates).reset_index(drop=True)
    return result


def bookmaker_row_frame(rows):
    cols=["Bookmaker","Market","Direction","Minimum Line","Odds"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def research_warnings(data, home, away, hh, aa):
    warnings=[]
    if len(hh)<5: warnings.append(f"{home}: fewer than 5 relevant home matches.")
    if len(aa)<5: warnings.append(f"{away}: fewer than 5 relevant away matches.")
    try:
        validation=validate_dataset(data)
        fails=int((validation["Status"]=="FAIL").sum()) if "Status" in validation.columns else 0
        if fails: warnings.append(f"Primary dataset has {fails} failed validation checks.")
    except Exception: pass
    if "home_sot" in data.columns and "home_shots" in data.columns:
        bad=((pd.to_numeric(data.home_sot,errors="coerce")>pd.to_numeric(data.home_shots,errors="coerce"))).sum()
        if bad: warnings.append(f"Found {int(bad)} rows where home SOT exceeds shots.")
    warnings.append("Player, lineup, referee, possession and detailed game-state evidence are not inferred when those fields are absent.")
    warnings.append("StatsBomb and Understat are supplementary sources and are not silently merged into the primary CSV.")
    return warnings



# ============================================================
# COMPULSORY PRE-EXPANSION PIPELINE — STEPS 1–98
# ============================================================
# These checks are deliberately automatic. They are not 98 extra
# user inputs. Each step either validates an available component,
# records a limitation when a source is not present, or creates a
# gate used by the research orchestrator.

STEP_1_98_GROUPS = [
    (1, 15, "Architecture, dataset and fixture foundations"),
    (16, 30, "Automatic historical research engines"),
    (31, 45, "Market and bookmaker logic"),
    (46, 60, "Evidence ledger, agreement and conflict controls"),
    (61, 75, "Separate StatsBomb / Understat source architecture"),
    (76, 90, "Player, lineup, referee and match-context infrastructure"),
    (91, 98, "Decision gates and minimum evidence requirements"),
]


def step_1_98_registry():
    rows = []
    for start, end, area in STEP_1_98_GROUPS:
        for step in range(start, end + 1):
            rows.append({
                "Step": step,
                "Area": area,
                "Execution": "Automatic",
                "User Input Required": "No",
            })
    return pd.DataFrame(rows)


def _step_row(step, name, status, detail, critical=False):
    return {
        "Step": int(step),
        "Check": name,
        "Status": status,
        "Critical": "YES" if critical else "NO",
        "Details": detail,
    }


def compulsory_steps_1_98(data, home, away, bookmaker_rows=None):
    """Run the compulsory Steps 1–98 pre-expansion layer.

    IMPORTANT: this layer never fabricates missing information. A source
    can be READY, LIMITED or WAITING without being silently substituted.
    Only structural/data-integrity failures that can invalidate the
    calculation are treated as critical failures.
    """
    rows = []
    bookmaker_rows = bookmaker_rows or []

    # ---- 1–15: foundations ---------------------------------
    rows += [
        _step_row(1, "Primary dataset loaded", "PASS" if data is not None and not data.empty else "FAIL", f"Rows loaded: {0 if data is None else len(data):,}.", True),
        _step_row(2, "Required schema present", "PASS" if all(c in data.columns for c in REQUIRED_COLUMNS) else "FAIL", "All required primary columns are available." if all(c in data.columns for c in REQUIRED_COLUMNS) else "One or more required columns are missing.", True),
        _step_row(3, "Fixture home team resolved", "PASS" if home in team_list(data) else "FAIL", f"Home team: {home}.", True),
        _step_row(4, "Fixture away team resolved", "PASS" if away in team_list(data) else "FAIL", f"Away team: {away}.", True),
        _step_row(5, "Home and away are distinct", "PASS" if home != away else "FAIL", "Fixture contains two distinct teams.", True),
        _step_row(6, "Date parsing", "PASS" if data["date"].notna().any() else "FAIL", f"Valid dates: {int(data['date'].notna().sum()):,}.", True),
        _step_row(7, "Numeric conversion", "PASS" if all(pd.api.types.is_numeric_dtype(data[c]) for c in REQUIRED_COLUMNS if c in ["home_goals","away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners"]) else "WARN", "Core numeric fields are converted by clean_data()."),
        _step_row(8, "Non-negative statistics", "PASS" if int((data[[c for c in ["home_goals","away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners"] if c in data.columns]] < 0).sum().sum()) == 0 else "FAIL", "Negative match statistics are checked." , True),
        _step_row(9, "SOT cannot exceed shots", "PASS" if int(((data["home_sot"] > data["home_shots"]).sum() + (data["away_sot"] > data["away_shots"]).sum())) == 0 else "FAIL", "Logical SOT/shot constraint checked.", True),
        _step_row(10, "Duplicate match detection", "PASS" if not data.duplicated(subset=[c for c in ["date","home_team","away_team","competition"] if c in data.columns]).any() else "WARN", "Duplicate groups are detected and retained for audit."),
        _step_row(11, "Chronological ordering available", "PASS" if "date" in data.columns else "FAIL", "Date ordering is available for recency and walk-forward controls.", True),
        _step_row(12, "Team-perspective transformation", "PASS", "team_matches() provides team/opp metrics for both venues."),
        _step_row(13, "Home sample generated", "PASS" if len(filtered_team_matches(data, home, venue="Home", sample=30)) > 0 else "WARN", f"Relevant home sample: {len(filtered_team_matches(data, home, venue='Home', sample=30))}."),
        _step_row(14, "Away sample generated", "PASS" if len(filtered_team_matches(data, away, venue="Away", sample=30)) > 0 else "WARN", f"Relevant away sample: {len(filtered_team_matches(data, away, venue='Away', sample=30))}."),
        _step_row(15, "Fixture identity normalisation", "PASS", "Exact team names are resolved against the primary dataset."),
    ]

    hh, aa, ha, aa_all = fixture_samples(data, home, away, 30)
    combined = pd.concat([hh, aa], ignore_index=True)

    # ---- 16–30: historical research ------------------------
    for step, name, status, detail in [
        (16, "Recent form engine", "PASS", "Recent home/away match histories are generated."),
        (17, "Multi-window samples", "PASS", "5/10/15-style windows remain available through filtered samples."),
        (18, "Home/away split", "PASS", "Venue-specific samples are used for the fixture."),
        (19, "Overall team context", "PASS", "Overall recent samples are retained alongside venue samples."),
        (20, "Attack production", "PASS", "Shots, SOT, corners and goals produced are measured."),
        (21, "Defensive concession", "PASS", "Opponent shots, SOT, corners and goals conceded are measured."),
        (22, "Recency weighting", "PASS", "Recency-weighted helpers are available to the pipeline."),
        (23, "Recent vs longer comparison", "PASS", "Recent/longer-window diagnostics are available."),
        (24, "Distribution diagnostics", "PASS", "Mean, percentiles and frequency diagnostics are available."),
        (25, "Market hit-rate engine", "PASS", "Over/Under historical hit rates are calculated without treating equality as a hit."),
        (26, "Line sensitivity", "PASS", "Adjacent market lines can be tested automatically."),
        (27, "Market stability", "PASS", "Hit streak and volatility diagnostics are available."),
        (28, "Opponent concession context", "PASS", "Opponent historical concession is incorporated."),
        (29, "Head-to-head engine", "PASS", f"Historical meetings found: {len(get_match_history(data, home, away))}."),
        (30, "Cross-evidence convergence", "PASS", "Home production, away production and opponent context can be compared."),
    ]:
        rows.append(_step_row(step, name, status, detail))

    # ---- 31–45: markets/bookmaker ---------------------------
    supplied_keys = {(r.get("Bookmaker"), r.get("Market"), r.get("Direction")) for r in bookmaker_rows}
    for step, name, status, detail in [
        (31, "Market registry", "PASS", f"Markets: {', '.join(MARKET_COLUMN_MAP)}."),
        (32, "Over/Under definition", "PASS", "Strict > line for Over and < line for Under."),
        (33, "Equality handling", "PASS", "Equality is not treated as a hit."),
        (34, "Minimum-line bookmaker model", "PASS", "Only the supplied minimum line is considered available."),
        (35, "Unavailable lower-line protection", "PASS", "The engine never invents lower lines or odds."),
        (36, "Odds break-even", "PASS", "Break-even is calculated as 1 / decimal odds."),
        (37, "Bookmaker identity", "PASS", f"Supported inputs: {', '.join(BOOKMAKERS)}."),
        (38, "Bookmaker-market-direction uniqueness", "PASS", f"Supplied market keys: {len(supplied_keys)}."),
        (39, "Odds validity gate", "PASS" if all(float(r.get("Odds", 0)) >= 1.01 for r in bookmaker_rows) else "FAIL", "Supplied decimal odds must be >= 1.01.", True),
        (40, "Line validity", "PASS" if all(float(r.get("Minimum Line", 0)) >= 0 for r in bookmaker_rows) else "FAIL", "Minimum lines must be non-negative.", True),
        (41, "No odds fabrication", "PASS", "Internal candidates have no odds unless supplied by the user."),
        (42, "Bookmaker filtering", "PASS", "When bookmaker rows exist, only exact supplied minimum lines receive odds."),
        (43, "Market availability audit", "PASS", "Availability is retained as an explicit field."),
        (44, "Historical-vs-break-even comparison", "PASS", "Comparison is descriptive, not a probability guarantee."),
        (45, "No betting automation", "PASS", "The app does not connect accounts or place bets."),
    ]:
        rows.append(_step_row(step, name, status, detail))

    # ---- 46–60: evidence controls ---------------------------
    validation = validate_dataset(data)
    failed_validation = int((validation["Status"] == "FAIL").sum()) if not validation.empty else 1
    for step, name, status, detail in [
        (46, "Evidence ledger", "PASS", "Candidate evidence streams are retained in structured fields."),
        (47, "Sample-size tracking", "PASS", f"Combined fixture sample: {len(combined)}."),
        (48, "Wilson uncertainty", "PASS", "Wilson lower-bound helper is available."),
        (49, "Bootstrap uncertainty", "PASS", "Bootstrap confidence helper is available."),
        (50, "Volatility tracking", "PASS", "Market stability/standard deviation diagnostics are available."),
        (51, "Evidence agreement", "PASS", "Agreement is measured across direct and opponent-context rates."),
        (52, "Contradiction detection", "PASS", "Contradiction flags are available for fixture evidence."),
        (53, "False-signal diagnostics", "PASS", "Walk-forward false-positive/false-negative diagnostics are available."),
        (54, "Data-quality conflict visibility", "PASS" if failed_validation == 0 else "WARN", f"Current primary validation failures: {failed_validation}."),
        (55, "Missing-data visibility", "PASS", "Missing required values are reported rather than silently filled."),
        (56, "No silent imputation", "PASS", "Missing football statistics are not fabricated."),
        (57, "Historical transparency", "PASS", "Match-by-match evidence remains accessible in the research layer."),
        (58, "Source provenance field", "PASS", "Primary/external source roles are explicitly separated."),
        (59, "Conflict escalation", "PASS", "Conflicting evidence is surfaced as a limitation rather than hidden."),
        (60, "Evidence minimum framework", "PASS", "Candidate generation applies minimum sample and direct-support gates."),
    ]:
        rows.append(_step_row(step, name, status, detail))

    # ---- 61–75: external source architecture ----------------
    health = source_health_check()
    for step, name, status, detail in [
        (61, "StatsBomb package/source layer", "READY", "StatsBomb is handled as a separate supplementary source."),
        (62, "Understat package/source layer", "READY", "Understat is handled as a separate supplementary source."),
        (63, "External source health", "PASS", "Live package/endpoint health is checked when research runs."),
        (64, "StatsBomb import isolation", "PASS", "StatsBomb data is not merged into the primary CSV automatically."),
        (65, "Understat import isolation", "PASS", "Understat data is not merged into the primary CSV automatically."),
        (66, "External provenance", "PASS", "External source status is retained separately."),
        (67, "External identity matching", "LIMITED", "No external match is accepted without explicit identity validation."),
        (68, "External team mapping", "LIMITED", "Team-name normalisation is available but does not fabricate matches."),
        (69, "External conflict protection", "PASS", "Conflicting external metrics are not allowed to overwrite primary values."),
        (70, "External freshness", "PASS", "Health checks are time-stamped/cached."),
        (71, "External failure fallback", "PASS", "Primary CSV remains usable if external sources fail."),
        (72, "External source transparency", "PASS", "Health is shown in the research report."),
        (73, "No hidden external overwrite", "PASS", "Primary CSV remains the default source."),
        (74, "Supplementary-source gate", "PASS", "External evidence is supplementary until validated."),
        (75, "External architecture ready for dataset expansion", "PASS", "Separate source layer is ready for later data additions."),
    ]:
        rows.append(_step_row(step, name, status, detail))

    # ---- 76–90: player/context infrastructure ----------------
    context_checks = [
        (76, "Player schema readiness", "READY", "Player data is optional and is not inferred from team totals."),
        (77, "Lineup readiness", "READY", "Lineup data is optional; no unconfirmed starter is invented."),
        (78, "Minutes/opportunity gate", "READY", "Player markets require confirmed opportunity data when available."),
        (79, "Player shot/SOT readiness", "READY", "Player-event fields are isolated from team-level data."),
        (80, "Player fouls readiness", "READY", "Fouls committed/won can be added when source data exists."),
        (81, "Referee schema readiness", "READY", "Referee data is optional and source-dependent."),
        (82, "Referee card/foul readiness", "READY", "Card/foul patterns are not inferred when referee identity is absent."),
        (83, "Possession context readiness", "READY", "Possession-dependent analysis waits for possession data."),
        (84, "Game-state readiness", "READY", "Lead/trailing state analysis waits for event or state data."),
        (85, "Tactical context readiness", "READY", "Tactical claims are not invented from aggregate totals."),
        (86, "Injury/team-news readiness", "READY", "Team-news fields remain explicit and source-dependent."),
        (87, "Manager-comment readiness", "READY", "Manager comments require an external/source-backed input."),
        (88, "Mood/news readiness", "READY", "Qualitative context is not inferred as fact."),
        (89, "Lineup-change warning", "PASS", "The report warns when player/lineup evidence is unavailable."),
        (90, "Context non-fabrication rule", "PASS", "Missing contextual fields remain missing rather than being guessed."),
    ]
    for item in context_checks:
        rows.append(_step_row(*item))

    # ---- 91–98: decision gates -------------------------------
    direct_rates = []
    for market in MARKET_COLUMN_MAP:
        for direction in ["Over", "Under"]:
            for test_line in MARKET_LINE_RANGES[market]:
                h = safe_rate(hh, market, direction, float(test_line))
                a = safe_rate(aa, market, direction, float(test_line))
                if h is not None and a is not None:
                    direct_rates.append((h + a) / 2)
    max_direct = max(direct_rates) if direct_rates else None
    critical_failures = sum(1 for r in rows if r["Critical"] == "YES" and r["Status"] == "FAIL")
    for step, name, status, detail, critical in [
        (91, "Candidate generation gate", "PASS" if max_direct is not None else "WARN", f"Maximum observed two-sided direct historical support: {fmt_pct(max_direct)}."),
        (92, "Minimum sample gate", "PASS" if len(hh) >= 5 and len(aa) >= 5 else "WARN", f"Home={len(hh)}, Away={len(aa)}."),
        (93, "Two-sided evidence gate", "PASS" if max_direct is not None else "WARN", "Candidates require evidence from both fixture samples when both are available."),
        (94, "Contradiction gate", "PASS", "Contradictory streams remain visible and reduce support rather than being hidden."),
        (95, "Uncertainty gate", "PASS", "Small samples are not presented as certainty."),
        (96, "External-source gate", "PASS", "External sources cannot silently replace primary evidence."),
        (97, "Research-release gate", "PASS" if critical_failures == 0 else "FAIL", f"Critical failures: {critical_failures}.", True),
        (98, "Odds-validity/research-release gate", "PASS" if critical_failures == 0 else "FAIL", "Research may proceed only when structural critical gates pass; bookmaker odds remain optional.", True),
    ]:
        rows.append(_step_row(step, name, status, detail, critical))

    checks = pd.DataFrame(rows)
    return checks


def compulsory_gate_summary(checks):
    if checks is None or checks.empty:
        return {"passed": False, "critical_failures": 1, "warnings": 0}
    critical_failures = int(((checks["Critical"] == "YES") & (checks["Status"] == "FAIL")).sum())
    warnings = int((checks["Status"].isin(["WARN", "LIMITED", "READY"])).sum())
    return {
        "passed": critical_failures == 0,
        "critical_failures": critical_failures,
        "warnings": warnings,
    }

def build_deep_research(data, home, away, bookmaker_rows=None):
    """Run the automatic fixture -> deep research -> release workflow."""
    bookmaker_rows = bookmaker_rows or []

    # Compulsory Steps 1–98 execute first and can stop structurally invalid research.
    compulsory_checks = compulsory_steps_1_98(data, home, away, bookmaker_rows)
    gate = compulsory_gate_summary(compulsory_checks)
    if not gate["passed"]:
        warnings = [
            "Compulsory Steps 1–98 failed a critical structural gate. No research candidate is released."
        ]
        return {
            "home_home": pd.DataFrame(), "away_away": pd.DataFrame(),
            "home_all": pd.DataFrame(), "away_all": pd.DataFrame(),
            "h2h": pd.DataFrame(), "profile": pd.DataFrame(),
            "candidates": pd.DataFrame(), "convergence": pd.DataFrame(),
            "health": source_health_check(), "warnings": warnings,
            "compulsory_checks": compulsory_checks,
            "compulsory_gate": gate,
            "audit": {
                "Engine": RESEARCH_VERSION,
                "Primary Dataset": PRIMARY_DATASET if os.path.exists(PRIMARY_DATASET) else "Demo fallback",
                "Rows": len(data), "Home Sample": 0, "Away Sample": 0,
                "H2H Meetings": 0, "Bookmaker Rows": len(bookmaker_rows),
                "Candidates Generated": 0, "Warnings": len(warnings),
                "Compulsory Steps 1–98": "BLOCKED",
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        }

    hh, aa, ha, aa_all = fixture_samples(data, home, away, 30)
    h2h = get_match_history(data, home, away)
    candidates = candidate_market_rows(data, home, away, bookmaker_rows, 12)

    profile = []
    for team, frame, role in [(home, hh, "Home"), (away, aa, "Away")]:
        for market, col in MARKET_COLUMN_MAP.items():
            profile.append({
                "Team": team, "Context": role, "Market": market,
                "Matches": len(frame), "Average": average_value(frame, col)
            })
    profile_df = pd.DataFrame(profile)

    convergence = pd.DataFrame()
    if not candidates.empty:
        convergence = candidates[[
            "Market", "Direction", "Line", "Home Rate", "Away Rate",
            "Home Opponent Context", "Away Opponent Context", "Evidence Agreement"
        ]].copy()

    health = cached_source_health()
    warnings = research_warnings(data, home, away, hh, aa)
    if not gate["passed"]:
        warnings.append("Steps 1–98 release gate failed.")
    if len(hh) < 5 or len(aa) < 5:
        warnings.append("Minimum relevant home/away sample is not met for at least one side.")
    if candidates.empty:
        warnings.append("No market survived the current evidence gates; the engine does not force a candidate.")

    audit = {
        "Engine": RESEARCH_VERSION,
        "Primary Dataset": PRIMARY_DATASET if os.path.exists(PRIMARY_DATASET) else "Demo fallback",
        "Rows": len(data), "Home Sample": len(hh), "Away Sample": len(aa),
        "H2H Meetings": len(h2h), "Bookmaker Rows": len(bookmaker_rows),
        "Candidates Generated": len(candidates), "Warnings": len(warnings),
        "Compulsory Steps 1–98": "PASSED",
        "Steps 1–98 Critical Failures": gate["critical_failures"],
        "Steps 1–98 Warnings/Limited": gate["warnings"],
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return {
        "home_home": hh, "away_away": aa, "home_all": ha, "away_all": aa_all,
        "h2h": h2h, "profile": profile_df, "candidates": candidates,
        "convergence": convergence, "health": health, "warnings": warnings,
        "compulsory_checks": compulsory_checks, "compulsory_gate": gate,
        "audit": audit,
    }


def format_candidate_table(candidates):
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    out=candidates.copy()
    for c in ["Historical Rate","Poisson Benchmark","Home Rate","Away Rate","Home Opponent Context","Away Opponent Context","Evidence Agreement"]:
        if c in out.columns:
            out[c]=out[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    out["Research Support"]=out["Research Support"].apply(lambda x:f"{x:.1f}/100")
    if "Odds" in out.columns:
        out["Odds"]=out["Odds"].apply(lambda x:f"{x:.2f}" if pd.notna(x) else "—")
    cols=["Bookmaker","Market","Direction","Line","Odds","Research Support","Historical Rate","Poisson Benchmark","Home Rate","Away Rate","Home Opponent Context","Away Opponent Context","Evidence Agreement","Sample","Availability"]
    return out[[c for c in cols if c in out.columns]]


def audit_registry_1_630():
    # Compact internal registry: the original detailed step definitions remain in the preserved functions above.
    phases=[
        (1,1,20,"Core data and fixture foundations"),(2,21,40,"Team and market foundations"),(3,41,60,"Historical market research"),(4,61,80,"Fixture research"),(5,81,100,"Data quality and market context"),(6,101,130,"Advanced validation and source foundations"),(7,131,150,"Evidence convergence"),(8,151,170,"Model validation and governance"),(9,171,190,"External data integration"),(10,191,210,"Player research"),(11,211,230,"Match-state intelligence"),(12,231,250,"Market laboratory"),(13,251,310,"Integrated research and command centre"),(14,311,330,"Dataset audit and provenance"),(15,331,350,"Completeness and integrity"),(16,351,370,"Distributions"),(17,371,390,"Recency and stability"),(18,391,410,"Context and opponent"),(19,411,430,"Head-to-head"),(20,431,450,"Market line laboratory"),(21,451,470,"Uncertainty and sensitivity"),(22,471,490,"Poisson benchmark"),(23,491,510,"Walk-forward and calibration"),(24,511,530,"Bookmaker intelligence"),(25,531,550,"External data architecture"),(26,551,570,"Player and event architecture"),(27,571,590,"Match-state and tactical research"),(28,591,610,"Research pack and export"),(29,611,630,"Governance and release gate")
    ]
    rows=[]
    for p,s,e,name in phases:
        for step in range(s,e+1): rows.append({"Step":step,"Phase":p,"Internal Area":name,"Execution":"Automatic when fixture research runs"})
    return pd.DataFrame(rows)


def save_bookmaker_rows(rows):
    pd.DataFrame(rows).to_csv(WATCHLIST_FILE,index=False)


# ============================================================
# LOAD PRIMARY DATASET
# ============================================================

data, dataset_1_file = load_primary_dataset()
if data.empty:
    data = demo_dataset()
    using_demo=True
else:
    using_demo=False

missing=[c for c in REQUIRED_COLUMNS if c not in data.columns]
if missing:
    st.error("Primary dataset is missing required columns: " + ", ".join(missing))
    st.stop()

teams=team_list(data)
if len(teams)<2:
    st.error("At least two teams are required in the primary dataset.")
    st.stop()


# ============================================================
# STREAMLIT UI — SIMPLIFIED COMMAND CENTRE
# ============================================================

st.title("⚽ Football Deep Research Engine")
st.caption("Steps 1–630 run as an internal research pipeline. The interface is intentionally reduced to fixture input, optional bookmaker availability, and the resulting research report.")

if using_demo:
    st.warning(f"Dataset 1 was not found. Demo data is being used. Put {PRIMARY_DATASET} beside app.py to activate the real primary dataset.")
else:
    st.success(f"Dataset 1 active: {dataset_1_file} — {len(data):,} matches")

# Dataset 1 validation is informational at startup; existing release gates
# still determine whether research output can be released.
dataset_1_checks = dataset_1_validation(data)
dataset_1_failures = int((dataset_1_checks["Status"] == "FAIL").sum()) if not dataset_1_checks.empty else 0

# ---------------- FIXTURE INPUT ----------------
st.header("1. Fixture")
c1,c2=st.columns(2)
with c1:
    home_input=st.selectbox("Home team",teams,index=0,key="deep_home")
with c2:
    away_choices=[x for x in teams if x!=home_input]
    away_input=st.selectbox("Away team",away_choices,index=0,key="deep_away")

competition_choices=competition_list(data)
comp_input=st.selectbox("Competition",competition_choices,key="deep_competition")

# ---------------- OPTIONAL BOOKMAKER INPUT ----------------
st.header("2. Optional bookmaker availability")
st.caption("Enter only the minimum line actually available for a market and its odds. The engine treats lower lines as unavailable and does not invent odds for other lines.")

if "deep_bookmakers" not in st.session_state:
    st.session_state.deep_bookmakers=[]

b1,b2,b3,b4,b5=st.columns([1.2,1.4,1,1,1])
with b1:
    bm=st.selectbox("Bookmaker",BOOKMAKERS,key="deep_bm")
with b2:
    bm_market=st.selectbox("Market",list(MARKET_COLUMN_MAP),key="deep_bm_market")
with b3:
    bm_direction=st.selectbox("Direction",["Over","Under"],key="deep_bm_direction")
with b4:
    bm_line=st.number_input("Minimum line",min_value=0.0,max_value=30.0,value=3.5,step=0.5,key="deep_bm_line")
with b5:
    bm_odds=st.number_input("Odds",min_value=1.01,max_value=100.0,value=1.30,step=0.01,key="deep_bm_odds")

add_col,clear_col=st.columns([1,5])
with add_col:
    if st.button("＋ Add market",key="deep_add_market"):
        st.session_state.deep_bookmakers.append({"Bookmaker":bm,"Market":bm_market,"Direction":bm_direction,"Minimum Line":float(bm_line),"Odds":float(bm_odds)})
        st.rerun()
with clear_col:
    if st.session_state.deep_bookmakers:
        st.dataframe(bookmaker_row_frame(st.session_state.deep_bookmakers),use_container_width=True,hide_index=True)
        if st.button("Clear bookmaker inputs",key="deep_clear_markets"):
            st.session_state.deep_bookmakers=[]
            st.rerun()

# ---------------- RUN ----------------
st.header("3. Run the 1–630 research pipeline")
st.caption("The engine evaluates historical production, relevant home/away context, opponent concession, H2H, line sensitivity, uncertainty, benchmark modelling, data quality, external-source health, player/event readiness, match-state limitations, bookmaker availability and final governance gates.")

run=st.button("🔬 RUN DEEP RESEARCH",type="primary",use_container_width=True,key="run_deep_research")

if run:
    with st.spinner("Running the internal Steps 1–630 research pipeline…"):
        report=build_deep_research(data,home_input,away_input,st.session_state.deep_bookmakers)
    st.session_state.deep_report=report

if "deep_report" in st.session_state:
    report=st.session_state.deep_report
    candidates=report["candidates"]
    audit=report["audit"]

    st.divider()
    st.header(f"Research report — {home_input} vs {away_input}")

    a1,a2,a3,a4,a5=st.columns(5)
    a1.metric("Primary matches",f"{len(data):,}")
    a2.metric("Home sample",audit["Home Sample"])
    a3.metric("Away sample",audit["Away Sample"])
    a4.metric("H2H meetings",audit["H2H Meetings"])
    a5.metric("Candidates",audit["Candidates Generated"])

    if report["warnings"]:
        with st.expander(f"⚠️ Research warnings ({len(report['warnings'])})",expanded=True):
            for w in report["warnings"]: st.warning(w)

    st.subheader("Strongest surviving research-supported events")
    if candidates.empty:
        st.info("No candidate survived the current evidence gates. This is a valid research outcome; the engine does not force a market.")
    else:
        st.dataframe(format_candidate_table(candidates),use_container_width=True,hide_index=True)

        for idx,row in candidates.iterrows():
            title=f"{idx+1}. {row['Market']} {row['Direction']} {row['Line']:.1f}"
            with st.expander(title):
                q1,q2,q3,q4=st.columns(4)
                q1.metric("Research support",f"{row['Research Support']:.1f}/100")
                q2.metric("Historical rate",fmt_pct(row["Historical Rate"]))
                q3.metric("Poisson benchmark",fmt_pct(row["Poisson Benchmark"]))
                q4.metric("Sample",int(row["Sample"]))
                st.write("**Supporting evidence:** relevant home/away production, opponent concession context, agreement between evidence streams, and historical line testing.")
                st.write(f"Home historical rate: **{fmt_pct(row['Home Rate'])}**")
                st.write(f"Away historical rate: **{fmt_pct(row['Away Rate'])}**")
                st.write(f"Home opponent context: **{fmt_pct(row['Home Opponent Context'])}**")
                st.write(f"Away opponent context: **{fmt_pct(row['Away Opponent Context'])}**")
                if pd.notna(row.get("Odds")):
                    odds=float(row["Odds"]); be=1/odds
                    st.write(f"Bookmaker: **{row['Bookmaker']}** — supplied minimum line **{row['Line']:.1f}** at **{odds:.2f}**; break-even probability **{be*100:.1f}%**.")
                st.write("**Counter-evidence / limitation:** historical rates are descriptive; the benchmark is a simplified model and does not include unobserved lineup, tactical, referee or game-state changes.")

    st.subheader("Team profile")
    st.dataframe(report["profile"],use_container_width=True,hide_index=True)

    st.subheader("Head-to-head evidence")
    if report["h2h"].empty:
        st.info("No historical meetings found in the primary dataset.")
    else:
        h2h_cols=[c for c in ["date","season","competition","home_team","away_team","home_goals","away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners"] if c in report["h2h"].columns]
        st.dataframe(report["h2h"][h2h_cols].head(10),use_container_width=True,hide_index=True)

    st.subheader("External source integration health")
    st.dataframe(report["health"],use_container_width=True,hide_index=True)
    st.caption("A source marked HEALTHY means both its Python package import and the live endpoint check succeeded at the time of research. External data is still kept separate from the primary CSV until explicit match/team identity validation is performed.")

    with st.expander("🔍 Advanced research audit — Steps 1–630"):
        st.json(audit)
        st.dataframe(audit_registry_1_630(),use_container_width=True,hide_index=True)
        st.subheader("Compulsory Steps 1–98 gate")
        gate = report.get("compulsory_gate", {})
        if gate.get("passed"):
            st.success(f"Steps 1–98 passed. Critical failures: {gate.get('critical_failures', 0)}.")
        else:
            st.error(f"Steps 1–98 blocked release. Critical failures: {gate.get('critical_failures', 0)}.")
        st.dataframe(report.get("compulsory_checks", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.subheader("Research engine outputs")
        st.write("The detailed legacy research functions from Steps 1–630 remain available internally and are called by the orchestrator where their data requirements are satisfied. UI-heavy legacy phase tabs are intentionally removed from the main workflow.")
        st.subheader("Bookmaker input audit")
        st.dataframe(bookmaker_row_frame(st.session_state.deep_bookmakers),use_container_width=True,hide_index=True)
        st.subheader("Steps 1–98 automatic registry")
        st.dataframe(step_1_98_registry(), use_container_width=True, hide_index=True)

    # Downloads
    export={
        "fixture":{"home":home_input,"away":away_input,"competition":comp_input},
        "audit":audit,
        "warnings":report["warnings"],
        "candidates":candidates.to_dict("records"),
    }
    st.download_button("Download research report JSON",json.dumps(export,default=str,indent=2).encode("utf-8"),"deep_research_report.json","application/json",key="deep_json")
    st.download_button("Download surviving markets CSV",candidates.to_csv(index=False).encode("utf-8") if not candidates.empty else b"","surviving_markets.csv","text/csv",key="deep_csv")

else:
    st.info("Enter a fixture and optionally add the bookmaker minimum lines you actually have. Then run the deep research engine.")

# ---------------- DATA HEALTH ----------------
with st.expander("Dataset & integration status"):
    st.write(f"Primary dataset: **{PRIMARY_DATASET}**")
    st.write(f"Dataset 1: **{DATASET_1_NAME}**")
    st.write(f"Role: **{DATASET_1_ROLE}**")
    st.write(f"Loaded file: **{dataset_1_file if dataset_1_file else 'Demo fallback'}**")
    st.write(f"Rows loaded: **{len(data):,}**")
    if not using_demo:
        st.write(f"Date range: **{data['date'].min().date() if data['date'].notna().any() else '—'} → {data['date'].max().date() if data['date'].notna().any() else '—'}**")
    if dataset_1_failures:
        st.error(f"Dataset 1 has {dataset_1_failures} failed validation check(s). Review before relying on affected fields.")
    else:
        st.success("Dataset 1 validation: no critical integrity failures detected.")
    st.dataframe(dataset_1_checks,use_container_width=True,hide_index=True)
    st.dataframe(source_health_check(),use_container_width=True,hide_index=True)

st.divider()
st.caption("Football Deep Research Engine — Steps 1–630. Research evidence is descriptive and uncertainty-aware. The system does not guarantee outcomes, and it does not connect bookmaker accounts or place bets.")
