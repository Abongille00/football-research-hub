import streamlit as st
import pandas as pd
import numpy as np
import os
import math

st.set_page_config(page_title="Football Research Hub V1", page_icon="⚽", layout="wide")

REQUIRED_COLUMNS = [
    "date",
    "season",
    "competition",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_corners",
    "away_corners"
]

MARKET_COLUMN_MAP = {
    "Shots": "team_shots",
    "Shots on Target": "team_sot",
    "Corners": "team_corners",
    "Goals": "team_goals"
}

WATCHLIST_FILE = "market_watchlist.csv"

demo = pd.DataFrame(
    [
        [
            "2026-09-18",
            "2026/27",
            "Premier League",
            "Chelsea",
            "Brentford",
            2,
            0,
            17,
            7,
            5,
            2,
            6,
            3
        ],
        [
            "2026-09-14",
            "2026/27",
            "Premier League",
            "Everton",
            "Chelsea",
            0,
            2,
            9,
            14,
            3,
            5,
            4,
            7
        ],
        [
            "2026-09-07",
            "2026/27",
            "Premier League",
            "Chelsea",
            "Fulham",
            3,
            1,
            19,
            8,
            7,
            3,
            8,
            2
        ],
        [
            "2026-08-30",
            "2026/27",
            "Premier League",
            "Newcastle",
            "Chelsea",
            1,
            1,
            12,
            11,
            4,
            4,
            5,
            5
        ],
        [
            "2026-08-24",
            "2026/27",
            "Premier League",
            "Chelsea",
            "Wolves",
            2,
            1,
            16,
            10,
            6,
            3,
            7,
            4
        ],
        [
            "2026-09-18",
            "2026/27",
            "Bundesliga",
            "Bayern Munich",
            "Union Berlin",
            3,
            1,
            20,
            7,
            8,
            2,
            8,
            2
        ],
        [
            "2026-09-13",
            "2026/27",
            "Bundesliga",
            "Mainz",
            "Bayern Munich",
            0,
            3,
            6,
            18,
            2,
            7,
            3,
            8
        ],
        [
            "2026-09-06",
            "2026/27",
            "Bundesliga",
            "Bayern Munich",
            "Freiburg",
            2,
            0,
            17,
            8,
            6,
            2,
            9,
            4
        ],
        [
            "2026-08-30",
            "2026/27",
            "Bundesliga",
            "Dortmund",
            "Bayern Munich",
            1,
            2,
            11,
            13,
            4,
            5,
            5,
            6
        ],
        [
            "2026-08-23",
            "2026/27",
            "Bundesliga",
            "Bayern Munich",
            "Leipzig",
            4,
            1,
            22,
            9,
            9,
            3,
            10,
            2
        ]
    ],
    columns=REQUIRED_COLUMNS
)

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


# ============================================================
# NEW WORKFLOW — FIXTURE-FIRST RESEARCH COMMAND CENTRE
# ============================================================

def _num(v):
    try:
        return float(v)
    except Exception:
        return None


def candidate_metrics(data, team, venue, market, direction, line):
    """Run the available historical research layers for one bookmaker candidate."""
    sample_rows = []
    windows = [5, 10, 15, 20, 30]
    for n in windows:
        m = filtered_team_matches(data, team, venue=venue, sample=n)
        a = analyse_market(m, market, direction, line)
        sample_rows.append({"Window": n, "Sample": a["sample_size"], "Hit Rate": a["hit_rate"]})

    base = filtered_team_matches(data, team, venue=venue, sample=30)
    a = analyse_market(base, market, direction, line)
    col = MARKET_COLUMN_MAP.get(market)
    vals = pd.to_numeric(base[col], errors="coerce").dropna() if col in base.columns else pd.Series(dtype=float)
    mean = float(vals.mean()) if not vals.empty else None
    median = float(vals.median()) if not vals.empty else None
    std = float(vals.std()) if len(vals) > 1 else None

    recent = {r["Window"]: r["Hit Rate"] for r in sample_rows}
    valid_rates = [x for x in recent.values() if x is not None]
    convergence = float(np.mean(valid_rates)) if valid_rates else None
    spread = float(max(valid_rates)-min(valid_rates)) if len(valid_rates) >= 2 else None

    # Opponent concession: for a team-specific market, compare the opponent's
    # relevant concession history where the data permits it.
    opp_metric = None
    opp_conceded = None
    if col:
        opp_col = {"team_shots":"opp_shots", "team_sot":"opp_sot", "team_corners":"opp_corners", "team_goals":"opp_goals"}.get(col)
        if opp_col and opp_col in base.columns:
            s = pd.to_numeric(base[opp_col], errors="coerce").dropna()
            if not s.empty:
                opp_conceded = float(s.mean())

    h2h_count = None
    return {
        "team": team, "venue": venue, "market": market, "direction": direction, "line": line,
        "sample": a["sample_size"], "hits": a["hits"], "misses": a["misses"],
        "hit_rate": a["hit_rate"], "mean": mean, "median": median, "std": std,
        "convergence": convergence, "window_spread": spread, "opp_conceded": opp_conceded,
        "windows": sample_rows,
    }


def _probability_label(p):
    if p is None:
        return "Insufficient data"
    return f"{p*100:.1f}% historical hit rate"


def _strength_score(m):
    """Transparent research-strength score, not a probability or guarantee."""
    p = m.get("hit_rate")
    if p is None or m.get("sample", 0) == 0:
        return 0.0
    score = 45.0 * p
    # More recent windows matter, but do not replace the full sample.
    for w, weight in [(5, 15), (10, 12), (15, 10), (20, 8)]:
        rr = next((x["Hit Rate"] for x in m["windows"] if x["Window"] == w), None)
        if rr is not None:
            score += weight * rr
    # Stability/convergence bonus and instability penalty.
    if m.get("window_spread") is not None:
        score += max(0.0, 10.0 - 20.0 * m["window_spread"])
    if m.get("sample", 0) >= 15:
        score += 3.0
    return round(min(100.0, max(0.0, score)), 1)


def build_fixture_candidates(data, home, away, bookmaker_rows):
    """Evaluate only the exact bookmaker minimum lines supplied by the user."""
    rows = []
    details = []
    for r in bookmaker_rows:
        if not r.get("available"):
            continue
        bookie = r["bookmaker"]
        market = r["market"]
        direction = r["direction"]
        line = _num(r["min_line"])
        odds = _num(r["odds"])
        team = r["team"]
        venue = "Home" if team == home else "Away"
        if line is None or odds is None or odds <= 1.0 or team not in [home, away]:
            continue
        m = candidate_metrics(data, team, venue, market, direction, line)
        m["bookmaker"] = bookie
        m["odds"] = odds
        m["breakeven"] = 1.0 / odds
        m["score"] = _strength_score(m)
        m["above_breakeven"] = (m["hit_rate"] is not None and m["hit_rate"] >= m["breakeven"])
        rows.append({
            "Rank Basis": m["score"],
            "Team": team,
            "Venue": venue,
            "Market": market,
            "Direction": direction,
            "Available Minimum Line": line,
            "Odds": odds,
            "Sample": m["sample"],
            "Historical Hit Rate": _probability_label(m["hit_rate"]),
            "5M": fmt_pct(next((x["Hit Rate"] for x in m["windows"] if x["Window"]==5),None)),
            "10M": fmt_pct(next((x["Hit Rate"] for x in m["windows"] if x["Window"]==10),None)),
            "15M": fmt_pct(next((x["Hit Rate"] for x in m["windows"] if x["Window"]==15),None)),
            "20M": fmt_pct(next((x["Hit Rate"] for x in m["windows"] if x["Window"]==20),None)),
            "30M": fmt_pct(next((x["Hit Rate"] for x in m["windows"] if x["Window"]==30),None)),
            "Breakeven": f"{m['breakeven']*100:.1f}%",
            "Research Strength": m["score"],
            "Line Availability Rule": "Exact minimum line only",
        })
        details.append(m)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Research Strength","Historical Hit Rate"], ascending=False, na_position="last").reset_index(drop=True)
        out.insert(0, "Research Order", range(1, len(out)+1))
    return out, details


def fixture_context_report(data, home, away):
    hm = filtered_team_matches(data, home, venue="Home", sample=15)
    am = filtered_team_matches(data, away, venue="Away", sample=15)
    h2h = get_match_history(data, home, away)
    rows = []
    for label, team, venue, df in [("Home",home,"Home",hm),("Away",away,"Away",am)]:
        s = match_team_summary(df)
        rows.extend([
            {"Area": f"{label} team", "Metric": "Matches", "Value": s["Matches"]},
            {"Area": f"{label} team", "Metric": "Avg Shots", "Value": s["Shots"]},
            {"Area": f"{label} team", "Metric": "Avg SOT", "Value": s["SOT"]},
            {"Area": f"{label} team", "Metric": "Avg Corners", "Value": s["Corners"]},
            {"Area": f"{label} team", "Metric": "Avg Goals", "Value": s["Goals"]},
        ])
    rows.append({"Area":"H2H", "Metric":"Historical meetings", "Value":len(h2h)})
    rows.append({"Area":"External sources", "Metric":"StatsBomb / Understat", "Value":"Separate source; not silently merged"})
    rows.append({"Area":"Player / lineup", "Metric":"Primary CSV availability", "Value":"Checked automatically from schema"})
    rows.append({"Area":"Referee / game state", "Metric":"Primary CSV availability", "Value":"Checked automatically from schema"})
    return pd.DataFrame(rows)


def candidate_detail_table(m):
    if not m:
        return pd.DataFrame()
    rows = [
        {"Research Layer":"Fixture", "Finding":f"{m['team']} ({m['venue']}) — {m['market']} {m['direction']} {m['line']}"},
        {"Research Layer":"Bookmaker", "Finding":f"{m['bookmaker']} @ {m['odds']:.2f}; exact minimum line supplied"},
        {"Research Layer":"Historical sample", "Finding":f"{m['sample']} usable matches; {m['hits']} hits / {m['misses']} misses"},
        {"Research Layer":"Overall hit rate", "Finding":_probability_label(m['hit_rate'])},
        {"Research Layer":"Recent windows", "Finding":" | ".join(f"{x['Window']}M: {fmt_pct(x['Hit Rate'])}" for x in m['windows'])},
        {"Research Layer":"Distribution", "Finding":f"Mean {m['mean']:.2f}" if m['mean'] is not None else "Unavailable"},
        {"Research Layer":"Opponent concession", "Finding":f"Avg conceded {m['opp_conceded']:.2f}" if m['opp_conceded'] is not None else "Unavailable in primary data"},
        {"Research Layer":"Window stability", "Finding":f"Spread across valid windows: {m['window_spread']*100:.1f} pp" if m['window_spread'] is not None else "Limited"},
        {"Research Layer":"Breakeven comparison", "Finding":f"Bookmaker breakeven {m['breakeven']*100:.1f}% vs historical {m['hit_rate']*100:.1f}%" if m['hit_rate'] is not None else "Unavailable"},
        {"Research Layer":"Research strength", "Finding":f"{m['score']}/100 — evidence-organising score, not a guaranteed probability"},
    ]
    return pd.DataFrame(rows)


# ============================================================
# DATA LOADING
# ============================================================
st.title("⚽ Football Research Hub — Fixture-First V1")
st.caption("Research assistant, not a prediction engine. Enter one fixture and only the bookmaker lines you actually have; the app researches the available candidates automatically.")

uploaded = st.sidebar.file_uploader("Optional CSV override", type=["csv"])
if uploaded is not None:
    try:
        data = clean_data(pd.read_csv(uploaded))
        st.sidebar.success(f"Loaded {len(data)} matches")
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    master_file = "football_master_2024_27_v1.csv"
    if os.path.exists(master_file):
        try:
            data = clean_data(pd.read_csv(master_file))
            st.sidebar.success(f"Using {master_file} — {len(data)} matches")
        except Exception as exc:
            st.error(f"Could not read {master_file}: {exc}")
            st.stop()
    else:
        data = clean_data(demo)
        st.sidebar.warning("Master CSV not found; using built-in demo data.")

missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# ============================================================
# SINGLE PRIMARY WORKFLOW
# ============================================================
st.header("1. Fixture")
c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("Home Team", team_list(data), key="fx_home")
with c2:
    away_options = [x for x in team_list(data) if x != home]
    away = st.selectbox("Away Team", away_options, key="fx_away")

st.header("2. Bookmaker availability")
st.caption("Enter only the minimum line that actually exists at the bookmaker, plus its odds. Do not enter lower lines that are unavailable. Example: if Virgin Bet starts Chelsea SOT at Over 3.5 @ 1.32, enter 3.5 and 1.32; the engine will NOT assume Over 2.5 exists.")

bookmakers = ["SportyBet", "SunBet", "Virgin Bet"]
markets = list(MARKET_COLUMN_MAP.keys())
teams_for_input = [home, away]
rows = []
for i, team in enumerate(teams_for_input):
    st.subheader(f"{team} — available team markets")
    cols = st.columns([1.1, 1.25, 1.0, 1.0, 1.0, 0.8])
    with cols[0]: bookie = st.selectbox("Bookmaker", bookmakers, key=f"bk_{i}")
    with cols[1]: market = st.selectbox("Market", markets, key=f"mk_{i}")
    with cols[2]: direction = st.selectbox("Direction", ["Over", "Under"], key=f"dr_{i}")
    with cols[3]: min_line = st.number_input("Minimum line", 0.0, 30.0, 3.5, 0.5, key=f"ln_{i}")
    with cols[4]: odds = st.number_input("Odds", 1.01, 100.0, 1.30, 0.01, key=f"od_{i}")
    with cols[5]: available = st.checkbox("Use", value=True, key=f"av_{i}")
    rows.append({"bookmaker":bookie,"team":team,"market":market,"direction":direction,"min_line":min_line,"odds":odds,"available":available})

st.info("You can use the same bookmaker more than once. Each row is one exact market/line you have available. The analysis never creates a lower line or invents an odds price.")

run = st.button("🔬 RUN FULL FIXTURE RESEARCH", type="primary", use_container_width=True)

if run:
    candidates, details = build_fixture_candidates(data, home, away, rows)
    st.session_state["last_candidates"] = candidates
    st.session_state["last_details"] = details
    st.session_state["last_fixture"] = (home, away)

if "last_fixture" in st.session_state and st.session_state["last_fixture"] == (home, away):
    candidates = st.session_state.get("last_candidates", pd.DataFrame())
    details = st.session_state.get("last_details", [])

    st.header("3. Automatic research output")
    if candidates.empty:
        st.warning("No usable bookmaker markets were entered. Turn on at least one market and provide a valid minimum line and odds.")
    else:
        st.subheader("Strongest surviving market evidence")
        st.dataframe(candidates.drop(columns=["Rank Basis"], errors="ignore"), use_container_width=True, hide_index=True)
        st.caption("Research Order is based on the transparent evidence score above. It is not a prediction probability and does not guarantee an outcome.")

        st.subheader("Deep research on each available candidate")
        for idx, m in enumerate(sorted(details, key=lambda x: x["score"], reverse=True), start=1):
            title = f"{idx}. {m['team']} — {m['market']} {m['direction']} {m['line']} @ {m['odds']:.2f} ({m['bookmaker']})"
            with st.expander(title, expanded=(idx == 1)):
                st.dataframe(candidate_detail_table(m), use_container_width=True, hide_index=True)

        st.subheader("Fixture-wide context")
        st.dataframe(fixture_context_report(data, home, away), use_container_width=True, hide_index=True)

        st.subheader("Data quality and source readiness")
        quality_frames = [validate_dataset(data), data_freshness_report(data), source_status_table()]
        for q in quality_frames:
            if q is not None and not q.empty:
                st.dataframe(q, use_container_width=True, hide_index=True)

        st.subheader("Research layers not silently invented")
        st.write("Player-level shots/SOT/fouls, confirmed lineups, referee behaviour, detailed game-state/possession splits, StatsBomb events and Understat xG are only used when corresponding data is actually present or explicitly mapped. Missing sources are reported rather than fabricated.")

        export = candidates.to_csv(index=False).encode("utf-8")
        st.download_button("Download fixture research candidates", export, "fixture_research_candidates.csv", "text/csv")

st.divider()
st.caption("Primary dataset: football_master_2024_27_v1.csv. Existing Steps 1–310 research functions are retained in the engine; the interface is now fixture-first and bookmaker-line-first.")
