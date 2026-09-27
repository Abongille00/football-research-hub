import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================
# FOOTBALL BETTING RESEARCH HUB
# STEPS 55–82
#
# Research assistant, not a prediction engine.
#
# Preserves:
#   Team Research
#   Market Tester
#   Bookmaker Monitor
#   Data Format
#   Match Research
#
# Adds:
#   Steps 55–82
#   Data integrity
#   Completeness
#   Trends
#   Comparable matches
#   Game-state analysis
#   Market stability
#   Line sensitivity
#   Odds sensitivity
#   Evidence framework
#   Multi-market comparison
#   Contradiction detection
#   Pre-match worksheet
#   Research reports
#   Research history
#   Post-match audit
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Betting Research Hub",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Betting Research Hub")
st.caption(
    "Match-by-match research, market testing and decision auditing. "
    "Historical evidence is descriptive and does not establish "
    "the probability of a future match."
)


# ============================================================
# CONFIGURATION
# ============================================================

MASTER_FILE = "football_master_2024_27_v1.csv"
WATCHLIST_FILE = "market_watchlist.csv"
RESEARCH_LOG_FILE = "research_history.csv"

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

MARKETS = [
    "Shots",
    "Shots on Target",
    "Corners",
    "Goals"
]

LINES = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
    5.5,
    6.5,
    7.5,
    8.5,
    9.5
]


# ============================================================
# OPTIONAL COLUMN MAP
# ============================================================

OPTIONAL_COLUMN_GROUPS = {
    "First-half shots": [
        "home_1h_shots",
        "away_1h_shots"
    ],
    "First-half SOT": [
        "home_1h_sot",
        "away_1h_sot"
    ],
    "First-half corners": [
        "home_1h_corners",
        "away_1h_corners"
    ],
    "First-half goals": [
        "home_1h_goals",
        "away_1h_goals"
    ],
    "Second-half shots": [
        "home_2h_shots",
        "away_2h_shots"
    ],
    "Second-half SOT": [
        "home_2h_sot",
        "away_2h_sot"
    ],
    "Second-half corners": [
        "home_2h_corners",
        "away_2h_corners"
    ],
    "Second-half goals": [
        "home_2h_goals",
        "away_2h_goals"
    ],
    "Possession": [
        "home_possession",
        "away_possession"
    ],
    "Game state": [
        "game_state",
        "score_state",
        "state"
    ],
    "Opponent strength": [
        "opponent_strength",
        "opponent_rating",
        "elo_opponent"
    ]
}


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
    """Standardise and clean football data."""

    df = df.copy()

    df.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        for c in df.columns
    ]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    text_columns = [
        "season",
        "competition",
        "home_team",
        "away_team"
    ]

    for c in text_columns:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "nan": "",
                        "None": ""
                    }
                )
            )

    numeric_columns = [
        c
        for c in df.columns
        if c not in text_columns
        and c != "date"
    ]

    for c in numeric_columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "home_team",
            "away_team"
        ]
    )

    if "date" in df.columns:
        df = df.sort_values(
            "date",
            ascending=False
        )

    return df.reset_index(drop=True)


# ============================================================
# TEAM PERSPECTIVE
# ============================================================

def team_matches(df, team):

    h = df[
        df["home_team"].eq(team)
    ].copy()

    h["team_goals"] = h["home_goals"]
    h["opp_goals"] = h["away_goals"]

    h["team_shots"] = h["home_shots"]
    h["opp_shots"] = h["away_shots"]

    h["team_sot"] = h["home_sot"]
    h["opp_sot"] = h["away_sot"]

    h["team_corners"] = h["home_corners"]
    h["opp_corners"] = h["away_corners"]

    h["venue"] = "Home"

    a = df[
        df["away_team"].eq(team)
    ].copy()

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

    return out.reset_index(drop=True)


def filtered_team_matches(
    data,
    team,
    season="All",
    competition="All",
    venue="All",
    sample=15
):

    m = team_matches(
        data,
        team
    ).copy()

    if season != "All":
        m = m[
            m["season"] == season
        ]

    if competition != "All":
        m = m[
            m["competition"] == competition
        ]

    if venue != "All":
        m = m[
            m["venue"] == venue
        ]

    if "date" in m.columns:
        m = m.sort_values(
            "date",
            ascending=False
        )

    return m.head(sample).copy()


# ============================================================
# BASIC HELPERS
# ============================================================

def average_value(df, column):

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

    return float(values.mean())


def median_value(df, column):

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

    return float(values.median())


def fmt_pct(value):

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except:
        pass

    return f"{float(value) * 100:.1f}%"


def fmt_num(value):

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except:
        pass

    return f"{float(value):.2f}"


def sample_quality(n):

    if n == 0:
        return (
            "No data",
            "No matches are available."
        )

    if n < 5:
        return (
            "Very small sample",
            "Fewer than 5 matches."
        )

    if n < 10:
        return (
            "Small sample",
            "A limited sample is available."
        )

    if n < 15:
        return (
            "Reasonable sample",
            "A useful historical sample is available."
        )

    return (
        "Strong sample",
        "15 or more matches are available."
    )


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyse_market(
    df,
    market,
    direction,
    line
):

    empty = {
        "hit_rate": None,
        "hits": 0,
        "misses": 0,
        "sample_size": 0
    }

    if df is None or df.empty:
        return empty

    column = MARKET_COLUMN_MAP.get(
        market
    )

    if column not in df.columns:
        return empty

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return empty

    if direction == "Over":
        hits = int(
            (values > line).sum()
        )
    else:
        hits = int(
            (values < line).sum()
        )

    n = len(values)

    return {
        "hit_rate": hits / n,
        "hits": hits,
        "misses": n - hits,
        "sample_size": n
    }


def market_history(
    df,
    market,
    direction,
    line
):

    if df is None or df.empty:
        return pd.DataFrame()

    column = MARKET_COLUMN_MAP.get(
        market
    )

    if column not in df.columns:
        return pd.DataFrame()

    cols = [
        "date",
        "home_team",
        "away_team",
        "venue",
        column
    ]

    available = [
        c
        for c in cols
        if c in df.columns
    ]

    result = df[
        available
    ].copy()

    result[column] = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    if direction == "Over":
        result["Hit"] = (
            result[column] > line
        )
    else:
        result["Hit"] = (
            result[column] < line
        )

    result["Hit"] = result[
        "Hit"
    ].map(
        {
            True: "✓",
            False: "✗"
        }
    )

    return result


# ============================================================
# STEP 55
# DATA INTEGRITY AUDIT
# ============================================================

def data_integrity_report(df):

    report = []

    report.append(
        {
            "Check": "Total rows",
            "Value": len(df),
            "Status": "OK"
        }
    )

    duplicates = int(
        df.duplicated().sum()
    )

    report.append(
        {
            "Check": "Duplicate rows",
            "Value": duplicates,
            "Status": (
                "OK"
                if duplicates == 0
                else "Review"
            )
        }
    )

    missing_date = int(
        df["date"].isna().sum()
    )

    report.append(
        {
            "Check": "Missing dates",
            "Value": missing_date,
            "Status": (
                "OK"
                if missing_date == 0
                else "Review"
            )
        }
    )

    for c in REQUIRED_COLUMNS:

        if c in df.columns:

            missing = int(
                df[c].isna().sum()
            )

            report.append(
                {
                    "Check": f"Missing {c}",
                    "Value": missing,
                    "Status": (
                        "OK"
                        if missing == 0
                        else "Review"
                    )
                }
            )

    impossible = 0

    goal_columns = [
        "home_goals",
        "away_goals"
    ]

    for c in goal_columns:

        if c in df.columns:

            impossible += int(
                (
                    pd.to_numeric(
                        df[c],
                        errors="coerce"
                    ) < 0
                ).sum()
            )

    report.append(
        {
            "Check": "Negative goals",
            "Value": impossible,
            "Status": (
                "OK"
                if impossible == 0
                else "Review"
            )
        }
    )

    return pd.DataFrame(report)


# ============================================================
# STEP 56
# MATCH COMPLETENESS
# ============================================================

def completeness_report(df):

    metric_columns = [
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "home_corners",
        "away_corners",
        "home_goals",
        "away_goals"
    ]

    rows = []

    for c in metric_columns:

        if c in df.columns:

            missing = int(
                df[c].isna().sum()
            )

            zero = int(
                (
                    pd.to_numeric(
                        df[c],
                        errors="coerce"
                    ) == 0
                ).sum()
            )

            rows.append(
                {
                    "Column": c,
                    "Missing": missing,
                    "Recorded Zero": zero,
                    "Available": len(df) - missing
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# STEP 57
# TREND CONSISTENCY
# ============================================================

def trend_analysis(
    data,
    team,
    season="All",
    competition="All",
    venue="All"
):

    samples = {}

    for n in [5, 10, 15]:

        samples[n] = filtered_team_matches(
            data,
            team,
            season,
            competition,
            venue,
            n
        )

    rows = []

    for label, column in [
        ("Shots", "team_shots"),
        ("SOT", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:

        v5 = average_value(
            samples[5],
            column
        )

        v10 = average_value(
            samples[10],
            column
        )

        v15 = average_value(
            samples[15],
            column
        )

        if (
            v5 is not None
            and v10 is not None
        ):

            change = v5 - v10

            if change > 0.5:
                trend = "Higher recently"
            elif change < -0.5:
                trend = "Lower recently"
            else:
                trend = "Stable"

        else:
            change = None
            trend = "Insufficient data"

        rows.append(
            {
                "Metric": label,
                "Last 5": fmt_num(v5),
                "Last 10": fmt_num(v10),
                "Last 15": fmt_num(v15),
                "5 vs 10 Change": (
                    f"{change:+.2f}"
                    if change is not None
                    else "—"
                ),
                "Trend": trend
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 58
# HOME / AWAY TREND
# ============================================================

def home_away_trend(
    data,
    team,
    season="All",
    competition="All"
):

    all_df = filtered_team_matches(
        data,
        team,
        season,
        competition,
        "All",
        15
    )

    home_df = filtered_team_matches(
        data,
        team,
        season,
        competition,
        "Home",
        15
    )

    away_df = filtered_team_matches(
        data,
        team,
        season,
        competition,
        "Away",
        15
    )

    rows = []

    for label, column in [
        ("Shots", "team_shots"),
        ("SOT", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:

        all_avg = average_value(
            all_df,
            column
        )

        home_avg = average_value(
            home_df,
            column
        )

        away_avg = average_value(
            away_df,
            column
        )

        rows.append(
            {
                "Metric": label,
                "Overall": fmt_num(all_avg),
                "Home": fmt_num(home_avg),
                "Away": fmt_num(away_avg),
                "Home vs Overall": (
                    f"{home_avg - all_avg:+.2f}"
                    if home_avg is not None
                    and all_avg is not None
                    else "—"
                ),
                "Away vs Overall": (
                    f"{away_avg - all_avg:+.2f}"
                    if away_avg is not None
                    and all_avg is not None
                    else "—"
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 59
# OPPONENT STRENGTH
# ============================================================

def detect_strength_column(df):

    for c in [
        "opponent_strength",
        "opponent_rating",
        "elo_opponent"
    ]:

        if c in df.columns:
            return c

    return None


def opponent_strength_context(
    team_df
):

    strength_column = detect_strength_column(
        team_df
    )

    if strength_column is None:
        return None

    values = pd.to_numeric(
        team_df[strength_column],
        errors="coerce"
    )

    valid = team_df[
        values.notna()
    ].copy()

    if len(valid) < 3:
        return None

    median_strength = (
        pd.to_numeric(
            valid[strength_column],
            errors="coerce"
        ).median()
    )

    valid["Strength Group"] = np.where(
        pd.to_numeric(
            valid[strength_column],
            errors="coerce"
        ) >= median_strength,
        "Stronger opponent",
        "Lower-strength opponent"
    )

    rows = []

    for group, subset in valid.groupby(
        "Strength Group"
    ):

        rows.append(
            {
                "Opponent Group": group,
                "Matches": len(subset),
                "Avg Shots": fmt_num(
                    average_value(
                        subset,
                        "team_shots"
                    )
                ),
                "Avg SOT": fmt_num(
                    average_value(
                        subset,
                        "team_sot"
                    )
                ),
                "Avg Corners": fmt_num(
                    average_value(
                        subset,
                        "team_corners"
                    )
                ),
                "Avg Goals": fmt_num(
                    average_value(
                        subset,
                        "team_goals"
                    )
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 60
# COMPARABLE MATCH FILTER
# ============================================================

def comparable_matches(
    team_df,
    opponent_group=None,
    min_opponent_value=None,
    max_opponent_value=None
):

    if team_df is None or team_df.empty:
        return pd.DataFrame()

    result = team_df.copy()

    strength_column = detect_strength_column(
        result
    )

    if strength_column is not None:

        values = pd.to_numeric(
            result[strength_column],
            errors="coerce"
        )

        if min_opponent_value is not None:
            result = result[
                values >= min_opponent_value
            ]

        if max_opponent_value is not None:
            result = result[
                values <= max_opponent_value
            ]

    return result


# ============================================================
# STEP 61
# HALF ANALYSIS
# ============================================================

def half_analysis(df):

    mappings = [
        ("First-half Shots", "home_1h_shots", "away_1h_shots"),
        ("Second-half Shots", "home_2h_shots", "away_2h_shots"),
        ("First-half SOT", "home_1h_sot", "away_1h_sot"),
        ("Second-half SOT", "home_2h_sot", "away_2h_sot"),
        ("First-half Corners", "home_1h_corners", "away_1h_corners"),
        ("Second-half Corners", "home_2h_corners", "away_2h_corners"),
        ("First-half Goals", "home_1h_goals", "away_1h_goals"),
        ("Second-half Goals", "home_2h_goals", "away_2h_goals")
    ]

    rows = []

    for label, home_col, away_col in mappings:

        if (
            home_col not in df.columns
            or away_col not in df.columns
        ):
            continue

        home_avg = average_value(
            df,
            home_col
        )

        away_avg = average_value(
            df,
            away_col
        )

        rows.append(
            {
                "Metric": label,
                "Home Avg": fmt_num(home_avg),
                "Away Avg": fmt_num(away_avg)
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 62
# GAME STATE
# ============================================================

def game_state_analysis(df):

    state_column = None

    for c in [
        "game_state",
        "score_state",
        "state"
    ]:

        if c in df.columns:
            state_column = c
            break

    if state_column is None:
        return None

    result = df.copy()

    result["Game State"] = (
        result[state_column]
        .astype(str)
        .str.strip()
    )

    rows = []

    for state, subset in result.groupby(
        "Game State"
    ):

        rows.append(
            {
                "Game State": state,
                "Matches": len(subset),
                "Avg Shots": fmt_num(
                    average_value(
                        subset,
                        "team_shots"
                    )
                ),
                "Avg SOT": fmt_num(
                    average_value(
                        subset,
                        "team_sot"
                    )
                ),
                "Avg Corners": fmt_num(
                    average_value(
                        subset,
                        "team_corners"
                    )
                ),
                "Avg Goals": fmt_num(
                    average_value(
                        subset,
                        "team_goals"
                    )
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 63
# GAME STATE WARNING
# ============================================================

def game_state_warning(df):

    state_df = game_state_analysis(
        df
    )

    if state_df is None or state_df.empty:
        return (
            "No game-state data",
            "The dataset does not contain usable game-state information."
        )

    if len(state_df) <= 1:
        return (
            "Limited state variation",
            "Only one game-state category is available."
        )

    return (
        "Game-state dependency should be checked",
        "Historical production may vary by score state."
    )


# ============================================================
# STEP 64
# MARKET STABILITY
# ============================================================

def market_stability(
    df,
    market,
    direction,
    line
):

    history = market_history(
        df,
        market,
        direction,
        line
    )

    if history.empty:
        return None

    hit_column = history["Hit"].map(
        {
            "✓": 1,
            "✗": 0
        }
    )

    longest_hit = 0
    longest_miss = 0
    current_hit = 0
    current_miss = 0

    for value in hit_column:

        if value == 1:
            current_hit += 1
            current_miss = 0
        else:
            current_miss += 1
            current_hit = 0

        longest_hit = max(
            longest_hit,
            current_hit
        )

        longest_miss = max(
            longest_miss,
            current_miss
        )

    recent_hits = int(
        hit_column.head(5).sum()
    )

    recent_n = min(
        5,
        len(hit_column)
    )

    return {
        "longest_hit_streak": longest_hit,
        "longest_miss_streak": longest_miss,
        "recent_hits": recent_hits,
        "recent_sample": recent_n,
        "recent_rate": (
            recent_hits / recent_n
            if recent_n
            else None
        )
    }


# ============================================================
# STEP 65
# LINE SENSITIVITY
# ============================================================

def line_sensitivity(
    df,
    market,
    direction,
    center_line,
    radius=2.0
):

    start = max(
        0,
        center_line - radius
    )

    end = center_line + radius

    test_lines = np.arange(
        start,
        end + 0.01,
        0.5
    )

    rows = []

    for test_line in test_lines:

        result = analyse_market(
            df,
            market,
            direction,
            float(test_line)
        )

        rows.append(
            {
                "Line": float(test_line),
                "Hits": result["hits"],
                "Sample": result["sample_size"],
                "Hit Rate": fmt_pct(
                    result["hit_rate"]
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 66
# ODDS SENSITIVITY
# ============================================================

def odds_sensitivity(
    historical_rate
):

    odds_list = [
        1.10,
        1.15,
        1.20,
        1.25,
        1.30,
        1.35,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        2.00
    ]

    rows = []

    for odds in odds_list:

        breakeven = 1 / odds

        difference = (
            historical_rate
            -
            breakeven
            if historical_rate is not None
            else None
        )

        rows.append(
            {
                "Odds": odds,
                "Break-even": fmt_pct(
                    breakeven
                ),
                "Historical Rate": fmt_pct(
                    historical_rate
                ),
                "Historical Difference": (
                    f"{difference * 100:+.1f} pp"
                    if difference is not None
                    else "—"
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 67
# EVIDENCE STRENGTH
# ============================================================

def evidence_strength(
    sample_size,
    recent_rate=None,
    longer_rate=None,
    home_away_rate=None,
    opponent_context_rate=None
):

    factors = []

    if sample_size >= 15:
        factors.append(
            "Adequate sample"
        )
    elif sample_size >= 10:
        factors.append(
            "Moderate sample"
        )
    elif sample_size >= 5:
        factors.append(
            "Small sample"
        )
    else:
        factors.append(
            "Very small sample"
        )

    if (
        recent_rate is not None
        and longer_rate is not None
    ):

        difference = abs(
            recent_rate
            -
            longer_rate
        )

        if difference <= 0.10:
            factors.append(
                "Recent/longer samples broadly aligned"
            )
        else:
            factors.append(
                "Recent/longer samples diverge"
            )

    if home_away_rate is not None:
        factors.append(
            "Venue-specific evidence available"
        )

    if opponent_context_rate is not None:
        factors.append(
            "Opponent-context evidence available"
        )

    return factors


# ============================================================
# STEP 68
# RESEARCH CHECKLIST
# ============================================================

CHECKLIST_ITEMS = [
    "Adequate sample checked",
    "Home/Away context checked",
    "Recent form checked",
    "Comparable opponents checked",
    "Opponent defensive context checked",
    "Market line tested",
    "Market stability checked",
    "Game-state considerations checked",
    "Team/news/lineup notes recorded",
    "Contradictions recorded"
]


# ============================================================
# STEP 69
# MULTI-MARKET COMPARISON
# ============================================================

def multi_market_analysis(
    df,
    direction,
    lines
):

    rows = []

    for market in MARKETS:

        for line in lines:

            result = analyse_market(
                df,
                market,
                direction,
                line
            )

            rows.append(
                {
                    "Market": market,
                    "Direction": direction,
                    "Line": line,
                    "Hits": result["hits"],
                    "Sample": result["sample_size"],
                    "Hit Rate": (
                        result["hit_rate"]
                        if result["hit_rate"] is not None
                        else np.nan
                    )
                }
            )

    result_df = pd.DataFrame(
        rows
    )

    if not result_df.empty:
        result_df["Hit Rate"] = (
            result_df["Hit Rate"] * 100
        ).round(1)

    return result_df


# ============================================================
# STEP 70
# MARKET MATRIX
# ============================================================

def market_matrix(
    df,
    direction
):

    rows = []

    for market in MARKETS:

        row = {
            "Market": market
        }

        for line in [
            1.5,
            2.5,
            3.5,
            4.5,
            5.5
        ]:

            result = analyse_market(
                df,
                market,
                direction,
                line
            )

            row[
                f"{direction} {line}"
            ] = (
                fmt_pct(
                    result["hit_rate"]
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# STEP 71
# CONTRADICTION DETECTOR
# ============================================================

def contradiction_detector(
    attacking_rate,
    defensive_rate,
    recent_rate,
    longer_rate,
    venue_rate
):

    contradictions = []

    if (
        attacking_rate is not None
        and defensive_rate is not None
    ):

        if (
            attacking_rate >= 0.70
            and defensive_rate < 0.50
        ):

            contradictions.append(
                "Team production and opponent defensive context disagree."
            )

    if (
        recent_rate is not None
        and longer_rate is not None
    ):

        if abs(
            recent_rate
            -
            longer_rate
        ) >= 0.20:

            contradictions.append(
                "Recent form differs materially from the longer sample."
            )

    if (
        venue_rate is not None
        and longer_rate is not None
    ):

        if abs(
            venue_rate
            -
            longer_rate
        ) >= 0.20:

            contradictions.append(
                "Venue-specific evidence differs materially from the broader sample."
            )

    if not contradictions:

        contradictions.append(
            "No major statistical contradiction detected in the selected inputs."
        )

    return contradictions


# ============================================================
# STEP 72
# EVIDENCE AGREEMENT
# ============================================================

def evidence_agreement(rates):

    valid = [
        r
        for r in rates
        if r is not None
    ]

    if len(valid) < 2:
        return (
            "Insufficient evidence",
            "Not enough independent historical rates are available."
        )

    spread = max(valid) - min(valid)

    if spread <= 0.10:
        return (
            "Broad agreement",
            "The selected historical rates are relatively close."
        )

    if spread <= 0.20:
        return (
            "Some disagreement",
            "The historical views are not identical."
        )

    return (
        "Material disagreement",
        "The historical evidence varies substantially between contexts."
    )


# ============================================================
# STEP 73
# DEFENSIVE VULNERABILITY
# ============================================================

def defensive_vulnerability(
    home_matches,
    away_matches
):

    rows = []

    for label, home_col, away_col in [
        (
            "Shots",
            "opp_shots",
            "opp_shots"
        ),
        (
            "SOT",
            "opp_sot",
            "opp_sot"
        ),
        (
            "Corners",
            "opp_corners",
            "opp_corners"
        ),
        (
            "Goals",
            "opp_goals",
            "opp_goals"
        )
    ]:

        rows.append(
            {
                "Metric": label,
                "Home Team Concedes": fmt_num(
                    average_value(
                        home_matches,
                        home_col
                    )
                ),
                "Away Team Concedes": fmt_num(
                    average_value(
                        away_matches,
                        away_col
                    )
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 74
# ATTACKING OPPORTUNITY
# ============================================================

def attacking_opportunity(
    home_matches,
    away_matches
):

    rows = []

    for label, column in [
        ("Shots", "team_shots"),
        ("SOT", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:

        rows.append(
            {
                "Metric": label,
                "Home Attack": fmt_num(
                    average_value(
                        home_matches,
                        column
                    )
                ),
                "Away Defence Conceded": fmt_num(
                    average_value(
                        away_matches,
                        {
                            "Shots": "opp_shots",
                            "SOT": "opp_sot",
                            "Corners": "opp_corners",
                            "Goals": "opp_goals"
                        }[label]
                    )
                ),
                "Away Attack": fmt_num(
                    average_value(
                        away_matches,
                        column
                    )
                ),
                "Home Defence Conceded": fmt_num(
                    average_value(
                        home_matches,
                        {
                            "Shots": "opp_shots",
                            "SOT": "opp_sot",
                            "Corners": "opp_corners",
                            "Goals": "opp_goals"
                        }[label]
                    )
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 75
# MARKET DEPENDENCY
# ============================================================

def market_dependency(
    df,
    market,
    direction,
    line
):

    column = MARKET_COLUMN_MAP.get(
        market
    )

    if (
        column not in df.columns
        or df.empty
    ):
        return None

    result = df.copy()

    result["Market Value"] = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    result = result.dropna(
        subset=["Market Value"]
    )

    if result.empty:
        return None

    if direction == "Over":
        result["Hit"] = (
            result["Market Value"] > line
        )
    else:
        result["Hit"] = (
            result["Market Value"] < line
        )

    dependencies = {}

    for other_market, other_column in {
        "Shots": "team_shots",
        "SOT": "team_sot",
        "Corners": "team_corners",
        "Goals": "team_goals"
    }.items():

        if other_column not in result.columns:
            continue

        values = pd.to_numeric(
            result[other_column],
            errors="coerce"
        )

        if values.notna().sum() < 3:
            continue

        try:
            correlation = (
                result[
                    ["Market Value", other_column]
                ]
                .corr()
                .iloc[0, 1]
            )
        except:
            correlation = np.nan

        dependencies[
            other_market
        ] = correlation

    rows = []

    for label, corr in dependencies.items():

        rows.append(
            {
                "Related Market": label,
                "Correlation": (
                    f"{corr:.2f}"
                    if pd.notna(corr)
                    else "—"
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 76
# RECENT VS LONGER DIVERGENCE
# ============================================================

def sample_divergence(
    data,
    team,
    market,
    direction,
    line,
    season="All",
    competition="All",
    venue="All"
):

    rates = []

    for n in [5, 10, 15]:

        sample = filtered_team_matches(
            data,
            team,
            season,
            competition,
            venue,
            n
        )

        result = analyse_market(
            sample,
            market,
            direction,
            line
        )

        rates.append(
            {
                "Sample": n,
                "Hit Rate": (
                    result["hit_rate"]
                    if result["hit_rate"] is not None
                    else np.nan
                ),
                "Matches": result["sample_size"]
            }
        )

    result_df = pd.DataFrame(
        rates
    )

    result_df["Hit Rate"] = (
        result_df["Hit Rate"] * 100
    ).round(1)

    return result_df


# ============================================================
# STEP 77
# THRESHOLD STRESS TEST
# ============================================================

def threshold_stress_test(
    df,
    market,
    direction,
    line
):

    test_lines = [
        max(0, line - 1.0),
        max(0, line - 0.5),
        line,
        line + 0.5,
        line + 1.0
    ]

    rows = []

    for test_line in sorted(
        set(test_lines)
    ):

        result = analyse_market(
            df,
            market,
            direction,
            test_line
        )

        rows.append(
            {
                "Line": test_line,
                "Hits": result["hits"],
                "Misses": result["misses"],
                "Sample": result["sample_size"],
                "Hit Rate": fmt_pct(
                    result["hit_rate"]
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 78
# RESEARCH NOTES
# ============================================================

def notes_section(
    key
):

    return st.text_area(
        "Research notes",
        height=180,
        placeholder=(
            "Record lineup information, injuries, tactical observations, "
            "manager comments, game-state concerns, market movement, "
            "or anything that statistical data does not capture."
        ),
        key=key
    )


# ============================================================
# STEP 79
# PRE-MATCH WORKSHEET
# ============================================================

def render_checklist(prefix):

    completed = []

    for i, item in enumerate(
        CHECKLIST_ITEMS
    ):

        completed.append(
            st.checkbox(
                item,
                key=f"{prefix}_{i}"
            )
        )

    return completed


# ============================================================
# STEP 80
# STRUCTURED RESEARCH REPORT
# ============================================================

def build_report(
    home_team,
    away_team,
    market,
    direction,
    line,
    odds,
    home_rate,
    away_rate,
    home_sample,
    away_sample,
    contradictions,
    checklist,
    notes
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    lines = []

    lines.append(
        "FOOTBALL MATCH RESEARCH REPORT"
    )

    lines.append(
        f"Generated: {timestamp}"
    )

    lines.append(
        f"Fixture: {home_team} vs {away_team}"
    )

    lines.append(
        ""
    )

    lines.append(
        "MARKET"
    )

    lines.append(
        f"{market} {direction} {line}"
    )

    lines.append(
        f"Bookmaker odds recorded: {odds:.2f}"
    )

    lines.append(
        ""
    )

    lines.append(
        "HISTORICAL CONTEXT"
    )

    lines.append(
        f"{home_team}: "
        f"{fmt_pct(home_rate)} "
        f"from {home_sample} matches"
    )

    lines.append(
        f"{away_team}: "
        f"{fmt_pct(away_rate)} "
        f"from {away_sample} matches"
    )

    lines.append(
        ""
    )

    lines.append(
        "CONTRADICTIONS / WARNINGS"
    )

    for item in contradictions:
        lines.append(
            f"- {item}"
        )

    lines.append(
        ""
    )

    lines.append(
        "RESEARCH CHECKLIST"
    )

    for item, done in zip(
        CHECKLIST_ITEMS,
        checklist
    ):

        lines.append(
            f"[{'X' if done else ' '}] {item}"
        )

    lines.append(
        ""
    )

    lines.append(
        "NOTES"
    )

    lines.append(
        notes
        if notes
        else "No notes recorded."
    )

    lines.append(
        ""
    )

    lines.append(
        "IMPORTANT"
    )

    lines.append(
        "This report describes historical evidence. "
        "It does not establish the probability of the next match "
        "and does not automatically classify a market as safe."
    )

    return "\n".join(lines)


# ============================================================
# STEP 81
# RESEARCH HISTORY
# ============================================================

def load_research_history():

    if not os.path.exists(
        RESEARCH_LOG_FILE
    ):
        return []

    try:

        df = pd.read_csv(
            RESEARCH_LOG_FILE
        ).fillna("")

        return df.to_dict(
            "records"
        )

    except:
        return []


def save_research_history(
    entry
):

    history = load_research_history()

    history.append(
        entry
    )

    pd.DataFrame(
        history
    ).to_csv(
        RESEARCH_LOG_FILE,
        index=False
    )


# ============================================================
# STEP 82
# POST-MATCH AUDIT
# ============================================================

def settle_research_record(
    actual_value,
    direction,
    line
):

    if actual_value is None:
        return "Unsettled"

    if direction == "Over":

        return (
            "Won"
            if actual_value > line
            else "Lost"
        )

    return (
        "Won"
        if actual_value < line
        else "Lost"
    )


# ============================================================
# DEMO DATA
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

st.sidebar.header("1. DATA")

uploaded = st.sidebar.file_uploader(
    "Upload match CSV",
    type=["csv"]
)

if uploaded is not None:

    try:

        data = clean_data(
            pd.read_csv(
                uploaded
            )
        )

        st.sidebar.success(
            f"Loaded {len(data)} matches"
        )

    except Exception as exc:

        st.error(
            f"Could not read uploaded CSV: {exc}"
        )

        st.stop()

elif os.path.exists(
    MASTER_FILE
):

    try:

        data = clean_data(
            pd.read_csv(
                MASTER_FILE
            )
        )

        st.sidebar.success(
            f"Using master dataset — {len(data)} matches"
        )

    except Exception as exc:

        st.error(
            f"Could not read {MASTER_FILE}: {exc}"
        )

        st.stop()

else:

    st.warning(
        f"{MASTER_FILE} was not found. "
        "The built-in demo dataset is being used."
    )

    data = clean_data(
        demo
    )

    st.sidebar.info(
        f"Demo dataset — {len(data)} matches"
    )


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

missing = [
    c
    for c in REQUIRED_COLUMNS
    if c not in data.columns
]

if missing:

    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing)
    )

    st.stop()


# ============================================================
# SIDEBAR DATA INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.write(
    f"**Rows:** {len(data):,}"
)

st.sidebar.write(
    f"**Teams:** {len(set(data.home_team) | set(data.away_team)):,}"
)

if "season" in data.columns:

    st.sidebar.write(
        f"**Seasons:** {data['season'].nunique()}"
    )

if "competition" in data.columns:

    st.sidebar.write(
        f"**Competitions:** {data['competition'].nunique()}"
    )


# ============================================================
# LIST HELPERS
# ============================================================

def team_list(data):

    return sorted(
        set(
            data["home_team"].dropna()
        )
        |
        set(
            data["away_team"].dropna()
        )
    )


def season_list(data):

    return [
        "All"
    ] + sorted(
        data["season"]
        .dropna()
        .unique()
        .tolist(),
        reverse=True
    )


def competition_list(data):

    return [
        "All"
    ] + sorted(
        data["competition"]
        .dropna()
        .unique()
        .tolist()
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🔎 Team Research",
        "🎯 Market Tester",
        "📊 Bookmaker Monitor",
        "📥 Data Format",
        "🔬 Match Research",
        "🧠 Decision Workspace",
        "🧪 Data & Audit"
    ]
)


# ============================================================
# TAB 1
# TEAM RESEARCH
# ============================================================

with tab1:

    st.subheader(
        "Team Research Dashboard"
    )

    team = st.selectbox(
        "Team",
        team_list(data),
        key="team_research_team"
    )

    season = st.selectbox(
        "Season",
        season_list(data),
        key="team_research_season"
    )

    competition = st.selectbox(
        "Competition",
        competition_list(data),
        key="team_research_competition"
    )

    venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="team_research_venue"
    )

    sample = st.select_slider(
        "Recent sample",
        options=[5, 10, 15],
        value=10,
        key="team_research_sample"
    )

    df = filtered_team_matches(
        data,
        team,
        season,
        competition,
        venue,
        sample
    )

    quality_label, quality_message = sample_quality(
        len(df)
    )

    if len(df) < 5:
        st.warning(
            f"{quality_label}: {quality_message}"
        )
    elif len(df) < 10:
        st.info(
            f"{quality_label}: {quality_message}"
        )
    else:
        st.success(
            f"{quality_label}: {quality_message}"
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Matches",
        len(df)
    )

    c2.metric(
        "Avg Shots",
        fmt_num(
            average_value(
                df,
                "team_shots"
            )
        )
    )

    c3.metric(
        "Avg SOT",
        fmt_num(
            average_value(
                df,
                "team_sot"
            )
        )
    )

    c4.metric(
        "Avg Corners",
        fmt_num(
            average_value(
                df,
                "team_corners"
            )
        )
    )

    c5.metric(
        "Avg Goals",
        fmt_num(
            average_value(
                df,
                "team_goals"
            )
        )
    )

    st.divider()

    st.subheader(
        "Recent Match History"
    )

    display_columns = [
        "date",
        "home_team",
        "away_team",
        "venue",
        "team_goals",
        "opp_goals",
        "team_shots",
        "team_sot",
        "team_corners"
    ]

    available = [
        c
        for c in display_columns
        if c in df.columns
    ]

    st.dataframe(
        df[available],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Steps 57–58: Trend Analysis"
    )

    st.dataframe(
        trend_analysis(
            data,
            team,
            season,
            competition,
            venue
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Home / Away Trend"
    )

    st.dataframe(
        home_away_trend(
            data,
            team,
            season,
            competition
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 59: Opponent Strength"
    )

    strength = opponent_strength_context(
        df
    )

    if strength is None:

        st.info(
            "No opponent-strength column is available in the current dataset."
        )

    else:

        st.dataframe(
            strength,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2
# MARKET TESTER
# ============================================================

with tab2:

    st.subheader(
        "🎯 Market Tester"
    )

    team = st.selectbox(
        "Team",
        team_list(data),
        key="market_test_team"
    )

    season = st.selectbox(
        "Season",
        season_list(data),
        key="market_test_season"
    )

    competition = st.selectbox(
        "Competition",
        competition_list(data),
        key="market_test_competition"
    )

    venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="market_test_venue"
    )

    market = st.selectbox(
        "Market",
        MARKETS,
        key="market_test_market"
    )

    direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="market_test_direction"
    )

    line = st.number_input(
        "Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5,
        key="market_test_line"
    )

    odds = st.number_input(
        "Decimal odds",
        min_value=1.01,
        max_value=100.0,
        value=1.30,
        step=0.01,
        key="market_test_odds"
    )

    sample = st.select_slider(
        "Sample",
        options=[5, 10, 15],
        value=10,
        key="market_test_sample"
    )

    df = filtered_team_matches(
        data,
        team,
        season,
        competition,
        venue,
        sample
    )

    result = analyse_market(
        df,
        market,
        direction,
        line
    )

    rate = result["hit_rate"]

    breakeven = 1 / odds

    a, b, c, d = st.columns(4)

    a.metric(
        "Historical Hit Rate",
        fmt_pct(rate)
    )

    b.metric(
        "Break-even",
        fmt_pct(breakeven)
    )

    c.metric(
        "Hits",
        result["hits"]
    )

    d.metric(
        "Sample",
        result["sample_size"]
    )

    if rate is not None:

        difference = (
            rate
            -
            breakeven
        )

        st.metric(
            "Historical Difference vs Break-even",
            f"{difference * 100:+.1f} pp"
        )

    st.caption(
        "This comparison is descriptive. A historical rate above "
        "a bookmaker break-even point does not prove future probability."
    )

    st.divider()

    st.subheader(
        "Steps 65–66: Sensitivity"
    )

    st.write(
        "Line sensitivity"
    )

    st.dataframe(
        line_sensitivity(
            df,
            market,
            direction,
            line
        ),
        use_container_width=True,
        hide_index=True
    )

    if rate is not None:

        st.write(
            "Odds sensitivity"
        )

        st.dataframe(
            odds_sensitivity(
                rate
            ),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Step 64: Market Stability"
    )

    stability = market_stability(
        df,
        market,
        direction,
        line
    )

    if stability:

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Longest Hit Streak",
            stability["longest_hit_streak"]
        )

        s2.metric(
            "Longest Miss Streak",
            stability["longest_miss_streak"]
        )

        s3.metric(
            "Recent Hits",
            stability["recent_hits"]
        )

        s4.metric(
            "Recent Rate",
            fmt_pct(
                stability["recent_rate"]
            )
        )

    st.divider()

    st.subheader(
        "Step 76: Recent vs Longer Sample"
    )

    st.dataframe(
        sample_divergence(
            data,
            team,
            market,
            direction,
            line,
            season,
            competition,
            venue
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 77: Threshold Stress Test"
    )

    st.dataframe(
        threshold_stress_test(
            df,
            market,
            direction,
            line
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Market History"
    )

    st.dataframe(
        market_history(
            df,
            market,
            direction,
            line
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3
# BOOKMAKER MONITOR
# ============================================================

with tab3:

    st.subheader(
        "📊 Bookmaker Market Monitor"
    )

    if (
        "market_watchlist"
        not in st.session_state
    ):

        if os.path.exists(
            WATCHLIST_FILE
        ):

            try:

                loaded = pd.read_csv(
                    WATCHLIST_FILE
                ).fillna("")

                if "Status" not in loaded.columns:
                    loaded["Status"] = "Watching"

                if "Actual Result" not in loaded.columns:
                    loaded["Actual Result"] = ""

                st.session_state.market_watchlist = (
                    loaded.to_dict(
                        "records"
                    )
                )

            except:
                st.session_state.market_watchlist = []

        else:

            st.session_state.market_watchlist = []

    col1, col2 = st.columns(2)

    with col1:

        bookmaker = st.selectbox(
            "Bookmaker",
            [
                "SportyBet",
                "SunBet",
                "Virgin Bet"
            ],
            key="bm_bookmaker"
        )

        match_name = st.text_input(
            "Match",
            key="bm_match"
        )

        match_date = st.date_input(
            "Match date",
            key="bm_date"
        )

        capture_time = st.time_input(
            "Capture time",
            key="bm_time"
        )

        monitor_team = st.selectbox(
            "Team",
            team_list(data),
            key="bm_team"
        )

        monitor_market = st.selectbox(
            "Market",
            MARKETS,
            key="bm_market"
        )

    with col2:

        monitor_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="bm_direction"
        )

        monitor_line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="bm_line"
        )

        monitor_odds = st.number_input(
            "Odds",
            min_value=1.01,
            max_value=100.0,
            value=1.30,
            step=0.01,
            key="bm_odds"
        )

        monitor_sample = st.selectbox(
            "Historical sample",
            [5, 10, 15],
            index=1,
            key="bm_sample"
        )

        monitor_season = st.selectbox(
            "Season",
            season_list(data),
            key="bm_season"
        )

        monitor_competition = st.selectbox(
            "Competition",
            competition_list(data),
            key="bm_competition"
        )

    if st.button(
        "Analyse & Add Market",
        key="bm_add"
    ):

        bm_df = filtered_team_matches(
            data,
            monitor_team,
            monitor_season,
            monitor_competition,
            "All",
            monitor_sample
        )

        result = analyse_market(
            bm_df,
            monitor_market,
            monitor_direction,
            monitor_line
        )

        rate = result["hit_rate"]

        breakeven = 1 / monitor_odds

        entry = {
            "Bookmaker": bookmaker,
            "Match": match_name,
            "Match Date": str(match_date),
            "Capture Time": str(capture_time),
            "Team": monitor_team,
            "Season": monitor_season,
            "Competition": monitor_competition,
            "Market": monitor_market,
            "Direction": monitor_direction,
            "Line": monitor_line,
            "Odds": monitor_odds,
            "Historical Hit Rate": (
                f"{rate * 100:.1f}%"
                if rate is not None
                else "—"
            ),
            "Break-even": f"{breakeven * 100:.1f}%",
            "Status": "Watching",
            "Actual Result": ""
        }

        st.session_state.market_watchlist.append(
            entry
        )

        pd.DataFrame(
            st.session_state.market_watchlist
        ).to_csv(
            WATCHLIST_FILE,
            index=False
        )

        st.success(
            "Market added."
        )

    st.divider()

    if st.session_state.market_watchlist:

        watch_df = pd.DataFrame(
            st.session_state.market_watchlist
        )

        total = len(
            watch_df
        )

        won = int(
            (
                watch_df["Status"]
                == "Won"
            ).sum()
        )

        lost = int(
            (
                watch_df["Status"]
                == "Lost"
            ).sum()
        )

        watching = int(
            (
                watch_df["Status"]
                == "Watching"
            ).sum()
        )

        settled = won + lost

        hit_rate_display = (
            f"{won / settled * 100:.1f}%"
            if settled
            else "—"
        )

        a, b, c, d, e = st.columns(5)

        a.metric(
            "Total",
            total
        )

        b.metric(
            "Watching",
            watching
        )

        c.metric(
            "Won",
            won
        )

        d.metric(
            "Lost",
            lost
        )

        e.metric(
            "Tracked Hit Rate",
            hit_rate_display
        )

        st.dataframe(
            watch_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = (
            watch_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Watchlist",
            csv_data,
            "market_watchlist.csv",
            "text/csv",
            key="bm_download"
        )

    else:

        st.info(
            "No markets are currently being tracked."
        )


# ============================================================
# TAB 4
# DATA FORMAT
# ============================================================

with tab4:

    st.subheader(
        "📥 Data Format & Dataset Health"
    )

    st.write(
        "Required columns:"
    )

    st.code(
        ",".join(
            REQUIRED_COLUMNS
        )
    )

    st.divider()

    st.subheader(
        "Step 55: Data Integrity Audit"
    )

    st.dataframe(
        data_integrity_report(data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 56: Match Completeness"
    )

    st.dataframe(
        completeness_report(data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Optional Data Detected"
    )

    optional_rows = []

    for label, columns in OPTIONAL_COLUMN_GROUPS.items():

        present = [
            c
            for c in columns
            if c in data.columns
        ]

        optional_rows.append(
            {
                "Feature": label,
                "Columns Found": (
                    ", ".join(present)
                    if present
                    else "None"
                ),
                "Available": (
                    "Yes"
                    if present
                    else "No"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            optional_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )

    demo_bytes = (
        demo
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download Demo CSV",
        demo_bytes,
        "football_demo.csv",
        "text/csv",
        key="data_demo_download"
    )


# ============================================================
# TAB 5
# MATCH RESEARCH
# ============================================================

with tab5:

    st.subheader(
        "🔬 Match Research"
    )

    teams = team_list(data)

    r1, r2 = st.columns(2)

    with r1:

        home_team = st.selectbox(
            "Home Team",
            teams,
            key="mr_home"
        )

    with r2:

        away_options = [
            t
            for t in teams
            if t != home_team
        ]

        away_team = st.selectbox(
            "Away Team",
            away_options,
            key="mr_away"
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        research_season = st.selectbox(
            "Season",
            season_list(data),
            key="mr_season"
        )

    with c2:

        research_competition = st.selectbox(
            "Competition",
            competition_list(data),
            key="mr_competition"
        )

    with c3:

        research_sample = st.selectbox(
            "Recent matches",
            [5, 10, 15],
            index=1,
            key="mr_sample"
        )

    home_matches = filtered_team_matches(
        data,
        home_team,
        research_season,
        research_competition,
        "Home",
        research_sample
    )

    away_matches = filtered_team_matches(
        data,
        away_team,
        research_season,
        research_competition,
        "Away",
        research_sample
    )

    home_all = filtered_team_matches(
        data,
        home_team,
        research_season,
        research_competition,
        "All",
        research_sample
    )

    away_all = filtered_team_matches(
        data,
        away_team,
        research_season,
        research_competition,
        "All",
        research_sample
    )

    st.divider()

    st.subheader(
        "Home vs Away Team Comparison"
    )

    comparison_rows = []

    for label, column in [
        ("Shots", "team_shots"),
        ("SOT", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:

        comparison_rows.append(
            {
                "Metric": label,
                home_team: fmt_num(
                    average_value(
                        home_matches,
                        column
                    )
                ),
                away_team: fmt_num(
                    average_value(
                        away_matches,
                        column
                    )
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            comparison_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 73: Defensive Vulnerability"
    )

    st.dataframe(
        defensive_vulnerability(
            home_matches,
            away_matches
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Step 74: Attacking Opportunity"
    )

    st.dataframe(
        attacking_opportunity(
            home_matches,
            away_matches
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 61: First / Second Half"
    )

    half_df = half_analysis(
        pd.concat(
            [
                home_matches,
                away_matches
            ],
            ignore_index=True
        )
    )

    if half_df.empty:

        st.info(
            "No first-half/second-half columns were detected."
        )

    else:

        st.dataframe(
            half_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Steps 62–63: Game State"
    )

    home_state = game_state_analysis(
        home_matches
    )

    away_state = game_state_analysis(
        away_matches
    )

    if (
        home_state is None
        and away_state is None
    ):

        st.info(
            "No game-state data is available."
        )

    else:

        if home_state is not None:

            st.write(
                f"{home_team}"
            )

            st.dataframe(
                home_state,
                use_container_width=True,
                hide_index=True
            )

        if away_state is not None:

            st.write(
                f"{away_team}"
            )

            st.dataframe(
                away_state,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.subheader(
        "Head-to-Head"
    )

    h2h = data[
        (
            (
                data["home_team"]
                == home_team
            )
            &
            (
                data["away_team"]
                == away_team
            )
        )
        |
        (
            (
                data["home_team"]
                == away_team
            )
            &
            (
                data["away_team"]
                == home_team
            )
        )
    ].copy()

    if research_season != "All":

        h2h = h2h[
            h2h["season"]
            == research_season
        ]

    if research_competition != "All":

        h2h = h2h[
            h2h["competition"]
            == research_competition
        ]

    if h2h.empty:

        st.info(
            "No H2H matches found."
        )

    else:

        st.dataframe(
            h2h.head(10),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Research Notes"
    )

    notes_section(
        "match_research_notes_82"
    )


# ============================================================
# TAB 6
# DECISION WORKSPACE
# STEPS 67–82
# ============================================================

with tab6:

    st.subheader(
        "🧠 Pre-Match Decision Workspace"
    )

    st.caption(
        "This workspace organises evidence and exposes contradictions. "
        "It deliberately does not assign a 'safe bet' label."
    )

    teams = team_list(data)

    d1, d2 = st.columns(2)

    with d1:

        ws_home = st.selectbox(
            "Home Team",
            teams,
            key="ws_home"
        )

    with d2:

        ws_away_options = [
            t
            for t in teams
            if t != ws_home
        ]

        ws_away = st.selectbox(
            "Away Team",
            ws_away_options,
            key="ws_away"
        )

    w1, w2, w3 = st.columns(3)

    with w1:

        ws_season = st.selectbox(
            "Season",
            season_list(data),
            key="ws_season"
        )

    with w2:

        ws_competition = st.selectbox(
            "Competition",
            competition_list(data),
            key="ws_competition"
        )

    with w3:

        ws_sample = st.selectbox(
            "Primary sample",
            [5, 10, 15],
            index=1,
            key="ws_sample"
        )

    home_ws = filtered_team_matches(
        data,
        ws_home,
        ws_season,
        ws_competition,
        "Home",
        ws_sample
    )

    away_ws = filtered_team_matches(
        data,
        ws_away,
        ws_season,
        ws_competition,
        "Away",
        ws_sample
    )

    st.divider()

    st.subheader(
        "Primary Market"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        ws_market = st.selectbox(
            "Market",
            MARKETS,
            key="ws_market"
        )

    with p2:

        ws_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="ws_direction"
        )

    with p3:

        ws_line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="ws_line"
        )

    with p4:

        ws_odds = st.number_input(
            "Odds",
            min_value=1.01,
            max_value=100.0,
            value=1.30,
            step=0.01,
            key="ws_odds"
        )

    home_result = analyse_market(
        home_ws,
        ws_market,
        ws_direction,
        ws_line
    )

    away_result = analyse_market(
        away_ws,
        ws_market,
        ws_direction,
        ws_line
    )

    home_rate = home_result[
        "hit_rate"
    ]

    away_rate = away_result[
        "hit_rate"
    ]

    st.divider()

    st.subheader(
        "Step 69: Multi-Market Comparison"
    )

    combined = pd.concat(
        [
            home_ws.assign(
                Side="Home"
            ),
            away_ws.assign(
                Side="Away"
            )
        ],
        ignore_index=True
    )

    multi_df = multi_market_analysis(
        combined,
        ws_direction,
        [
            1.5,
            2.5,
            3.5,
            4.5,
            5.5
        ]
    )

    st.dataframe(
        multi_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 70: Market Matrix"
    )

    st.dataframe(
        market_matrix(
            combined,
            ws_direction
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 71–72: Contradictions & Evidence Agreement"
    )

    home_recent = analyse_market(
        home_ws.head(5),
        ws_market,
        ws_direction,
        ws_line
    )

    home_longer = analyse_market(
        home_ws,
        ws_market,
        ws_direction,
        ws_line
    )

    away_recent = analyse_market(
        away_ws.head(5),
        ws_market,
        ws_direction,
        ws_line
    )

    away_longer = analyse_market(
        away_ws,
        ws_market,
        ws_direction,
        ws_line
    )

    contradiction_list = contradiction_detector(
        home_rate,
        away_rate,
        home_recent["hit_rate"],
        home_longer["hit_rate"],
        home_rate
    )

    for item in contradiction_list:

        st.warning(
            item
        )

    agreement_label, agreement_message = evidence_agreement(
        [
            home_rate,
            away_rate,
            home_recent["hit_rate"],
            away_recent["hit_rate"]
        ]
    )

    st.info(
        f"**{agreement_label}:** {agreement_message}"
    )

    st.divider()

    st.subheader(
        "Step 67: Evidence Strength"
    )

    evidence = evidence_strength(
        max(
            home_result["sample_size"],
            away_result["sample_size"]
        ),
        home_recent["hit_rate"],
        home_longer["hit_rate"],
        home_rate,
        away_rate
    )

    for item in evidence:

        st.write(
            f"• {item}"
        )

    st.divider()

    st.subheader(
        "Step 75: Market Dependency"
    )

    dependency = market_dependency(
        combined,
        ws_market,
        ws_direction,
        ws_line
    )

    if dependency is None:

        st.info(
            "Not enough data to calculate market relationships."
        )

    else:

        st.dataframe(
            dependency,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Step 68: Research Checklist"
    )

    checklist_results = render_checklist(
        "decision_check"
    )

    completed_count = sum(
        checklist_results
    )

    st.progress(
        completed_count
        /
        len(CHECKLIST_ITEMS)
    )

    st.write(
        f"{completed_count} / {len(CHECKLIST_ITEMS)} "
        "research checks completed."
    )

    st.divider()

    st.subheader(
        "Step 78: Research Notes"
    )

    ws_notes = notes_section(
        "decision_workspace_notes"
    )

    st.divider()

    st.subheader(
        "Step 79: Final Research Worksheet"
    )

    st.write(
        f"**Fixture:** {ws_home} vs {ws_away}"
    )

    st.write(
        f"**Market:** {ws_market} {ws_direction} {ws_line}"
    )

    st.write(
        f"**Recorded odds:** {ws_odds:.2f}"
    )

    final_table = pd.DataFrame(
        [
            {
                "Evidence": f"{ws_home} home sample",
                "Matches": home_result["sample_size"],
                "Historical Rate": fmt_pct(
                    home_rate
                )
            },
            {
                "Evidence": f"{ws_away} away sample",
                "Matches": away_result["sample_size"],
                "Historical Rate": fmt_pct(
                    away_rate
                )
            },
            {
                "Evidence": f"{ws_home} recent 5",
                "Matches": home_recent["sample_size"],
                "Historical Rate": fmt_pct(
                    home_recent["hit_rate"]
                )
            },
            {
                "Evidence": f"{ws_away} recent 5",
                "Matches": away_recent["sample_size"],
                "Historical Rate": fmt_pct(
                    away_recent["hit_rate"]
                )
            }
        ]
    )

    st.dataframe(
        final_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 80: Generate Research Report"
    )

    if st.button(
        "Generate Final Research Report",
        key="generate_report"
    ):

        report = build_report(
            ws_home,
            ws_away,
            ws_market,
            ws_direction,
            ws_line,
            ws_odds,
            home_rate,
            away_rate,
            home_result["sample_size"],
            away_result["sample_size"],
            contradiction_list,
            checklist_results,
            ws_notes
        )

        st.text_area(
            "Research Report",
            report,
            height=500,
            key="generated_report"
        )

        st.download_button(
            "Download Research Report",
            report.encode("utf-8"),
            "match_research_report.txt",
            "text/plain",
            key="download_report"
        )

    st.divider()

    st.subheader(
        "Step 81: Save Research Session"
    )

    if st.button(
        "Save Research Session",
        key="save_research"
    ):

        entry = {
            "Timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Home Team": ws_home,
            "Away Team": ws_away,
            "Market": ws_market,
            "Direction": ws_direction,
            "Line": ws_line,
            "Odds": ws_odds,
            "Home Sample": home_result[
                "sample_size"
            ],
            "Home Historical Rate": (
                home_rate
                if home_rate is not None
                else ""
            ),
            "Away Sample": away_result[
                "sample_size"
            ],
            "Away Historical Rate": (
                away_rate
                if away_rate is not None
                else ""
            ),
            "Checklist Completed": completed_count,
            "Checklist Total": len(
                CHECKLIST_ITEMS
            ),
            "Notes": ws_notes
        }

        save_research_history(
            entry
        )

        st.success(
            "Research session saved."
        )


# ============================================================
# TAB 7
# DATA & AUDIT
# ============================================================

with tab7:

    st.subheader(
        "🧪 Research & Data Audit Centre"
    )

    st.caption(
        "Use this area to inspect data quality and review previous "
        "research sessions rather than relying only on a single market rate."
    )

    st.subheader(
        "Dataset Integrity"
    )

    integrity = data_integrity_report(
        data
    )

    problems = integrity[
        integrity["Status"] != "OK"
    ]

    if problems.empty:

        st.success(
            "No integrity problems were detected by the current checks."
        )

    else:

        st.warning(
            f"{len(problems)} integrity item(s) require review."
        )

    st.dataframe(
        integrity,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Match Completeness"
    )

    completeness = completeness_report(
        data
    )

    st.dataframe(
        completeness,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Step 81: Research History"
    )

    history = load_research_history()

    if history:

        history_df = pd.DataFrame(
            history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        history_csv = (
            history_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Research History",
            history_csv,
            "research_history.csv",
            "text/csv",
            key="history_download"
        )

    else:

        st.info(
            "No saved research sessions yet."
        )

    st.divider()

    st.subheader(
        "Step 82: Post-Match Audit"
    )

    st.caption(
        "Enter the actual market result to compare the completed "
        "match with the pre-match research record."
    )

    audit_market = st.selectbox(
        "Market",
        MARKETS,
        key="audit_market"
    )

    audit_direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="audit_direction"
    )

    audit_line = st.number_input(
        "Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5,
        key="audit_line"
    )

    audit_actual = st.number_input(
        "Actual result",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
        key="audit_actual"
    )

    if st.button(
        "Settle Audit",
        key="settle_audit"
    ):

        outcome = settle_research_record(
            audit_actual,
            audit_direction,
            audit_line
        )

        if outcome == "Won":

            st.success(
                f"Historical market result: {outcome}"
            )

        else:

            st.error(
                f"Historical market result: {outcome}"
            )

        st.info(
            f"Recorded market: {audit_market} "
            f"{audit_direction} {audit_line}. "
            f"Actual result: {audit_actual}."
        )

    st.divider()

    st.subheader(
        "Research Audit Questions"
    )

    st.checkbox(
        "Did I use the correct home/away sample?",
        key="audit_question_1"
    )

    st.checkbox(
        "Did I check recent form rather than relying only on an average?",
        key="audit_question_2"
    )

    st.checkbox(
        "Did I examine comparable opponents?",
        key="audit_question_3"
    )

    st.checkbox(
        "Did I check the opponent's defensive numbers?",
        key="audit_question_4"
    )

    st.checkbox(
        "Did I test nearby market lines?",
        key="audit_question_5"
    )

    st.checkbox(
        "Did I check whether the market was dependent on game state?",
        key="audit_question_6"
    )

    st.checkbox(
        "Did I record any tactical or lineup contradiction?",
        key="audit_question_7"
    )

    st.checkbox(
        "Did I avoid ignoring evidence simply because the team is a big name?",
        key="audit_question_8"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Football Betting Research Hub — Steps 55–82. "
    "Historical statistics are evidence for research, not guarantees. "
    "No automatic safe-bet, winner, ranking or prediction label is generated."
)
