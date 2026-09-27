import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================
# FOOTBALL BETTING RESEARCH HUB
# STEPS 41–54 INTEGRATED
#
# Research assistant, NOT a prediction engine.
#
# Primary dataset:
# football_master_2024_27_v1.csv
#
# Existing functionality retained:
# - Team Research
# - Market Tester
# - Bookmaker Monitor
# - Data Format
# - Match Research
#
# New:
# 41 Comparable-opponent research
# 42 Opponent-strength filtering
# 43 Recent-form trend analysis
# 44 Market consistency analysis
# 45 Hit/miss streak detection
# 46 Line ladder testing
# 47 Multi-line market matrix
# 48 Attack vs defensive matchup index
# 49 Produced vs conceded comparison
# 50 Sample stability check
# 51 Recent-vs-longer-sample comparison
# 52 Research confidence/data-quality panel
# 53 Fixture research report/export
# 54 Research decision checklist/audit
# ============================================================

st.set_page_config(
    page_title="Football Betting Research Hub",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Betting Research Hub")
st.caption(
    "Research assistant, not a prediction engine. "
    "Use match-by-match evidence to test a market before betting."
)

# ============================================================
# CONFIGURATION
# ============================================================

MASTER_FILE = "football_master_2024_27_v1.csv"
WATCHLIST_FILE = "market_watchlist.csv"

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

OPPONENT_COLUMN_MAP = {
    "Shots": "opp_shots",
    "Shots on Target": "opp_sot",
    "Corners": "opp_corners",
    "Goals": "opp_goals"
}

NUMERIC_COLUMNS = [
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_corners",
    "away_corners"
]


# ============================================================
# BASIC HELPERS
# ============================================================

def fmt_pct(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%"


def fmt_num(value, decimals=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def safe_float(value, default=None):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def market_column(market):
    return MARKET_COLUMN_MAP.get(market)


def opponent_column(market):
    return OPPONENT_COLUMN_MAP.get(market)


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
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

    for c in [
        "season",
        "competition",
        "home_team",
        "away_team"
    ]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace("nan", "")
            )

    for c in NUMERIC_COLUMNS:
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


# ============================================================
# TEAM PERSPECTIVE
# ============================================================

def team_matches(df, team):

    home = df[
        df["home_team"].eq(team)
    ].copy()

    home["team_goals"] = home["home_goals"]
    home["opp_goals"] = home["away_goals"]

    home["team_shots"] = home["home_shots"]
    home["opp_shots"] = home["away_shots"]

    home["team_sot"] = home["home_sot"]
    home["opp_sot"] = home["away_sot"]

    home["team_corners"] = home["home_corners"]
    home["opp_corners"] = home["away_corners"]

    home["venue"] = "Home"

    away = df[
        df["away_team"].eq(team)
    ].copy()

    away["team_goals"] = away["away_goals"]
    away["opp_goals"] = away["home_goals"]

    away["team_shots"] = away["away_shots"]
    away["opp_shots"] = away["home_shots"]

    away["team_sot"] = away["away_sot"]
    away["opp_sot"] = away["home_sot"]

    away["team_corners"] = away["away_corners"]
    away["opp_corners"] = away["home_corners"]

    away["venue"] = "Away"

    result = pd.concat(
        [home, away],
        ignore_index=True
    )

    if "date" in result.columns:
        result = result.sort_values(
            "date",
            ascending=False
        )

    return result


def filtered_team_matches(
    data,
    team,
    season="All",
    competition="All",
    venue="All",
    sample=15
):

    result = team_matches(
        data,
        team
    ).copy()

    if season != "All":
        result = result[
            result["season"] == season
        ]

    if competition != "All":
        result = result[
            result["competition"] == competition
        ]

    if venue != "All":
        result = result[
            result["venue"] == venue
        ]

    result = result.sort_values(
        "date",
        ascending=False
    )

    return result.head(sample)


# ============================================================
# LIST HELPERS
# ============================================================

def team_list(data):

    return sorted(
        set(data["home_team"].dropna())
        |
        set(data["away_team"].dropna())
    )


def season_list(data):

    values = (
        data["season"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return ["All"] + sorted(
        values,
        reverse=True
    )


def competition_list(data):

    values = (
        data["competition"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return ["All"] + sorted(values)


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

    column = market_column(market)

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

    column = market_column(market)

    if column not in df.columns:
        return pd.DataFrame()

    cols = [
        "date",
        "home_team",
        "away_team",
        "venue",
        column
    ]

    cols = [
        c for c in cols
        if c in df.columns
    ]

    result = df[cols].copy()

    values = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    if direction == "Over":
        result["Hit"] = np.where(
            values > line,
            "✓",
            "✗"
        )
    else:
        result["Hit"] = np.where(
            values < line,
            "✓",
            "✗"
        )

    return result


# ============================================================
# SUMMARY FUNCTIONS
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

    return values.mean()


def team_summary(df):

    result = {
        "Matches": len(df)
    }

    mapping = {
        "Shots": "team_shots",
        "Shots on Target": "team_sot",
        "Corners": "team_corners",
        "Goals": "team_goals"
    }

    for label, column in mapping.items():

        result[label] = average_value(
            df,
            column
        )

    return result


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
            "Limited historical evidence."
        )

    if n < 15:
        return (
            "Reasonable sample",
            "A useful sample is available."
        )

    return (
        "Strong sample",
        "15 or more matches are available."
    )


# ============================================================
# H2H
# ============================================================

def get_match_history(
    data,
    home_team,
    away_team
):

    result = data[
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

    return result.sort_values(
        "date",
        ascending=False
    )


# ============================================================
# STEP 41
# COMPARABLE OPPONENT RESEARCH
# ============================================================

def comparable_opponents(
    data,
    team,
    venue="All",
    sample=15
):

    matches = team_matches(
        data,
        team
    )

    if venue != "All":
        matches = matches[
            matches["venue"] == venue
        ]

    matches = matches.sort_values(
        "date",
        ascending=False
    ).head(sample)

    if matches.empty:
        return matches

    return matches


# ============================================================
# STEP 42
# OPPONENT STRENGTH FILTERING
# ============================================================

def opponent_strength_table(
    data,
    team,
    venue="All",
    sample=15
):

    matches = comparable_opponents(
        data,
        team,
        venue,
        sample
    )

    if matches.empty:
        return pd.DataFrame()

    rows = []

    for _, row in matches.iterrows():

        opponent = (
            row["away_team"]
            if row["venue"] == "Home"
            else row["home_team"]
        )

        opponent_matches = team_matches(
            data,
            opponent
        )

        opponent_avg_sot = average_value(
            opponent_matches.head(15),
            "team_sot"
        )

        opponent_avg_shots = average_value(
            opponent_matches.head(15),
            "team_shots"
        )

        rows.append({
            "Date": row["date"],
            "Opponent": opponent,
            "Venue": row["venue"],
            "Opponent Avg Shots": opponent_avg_shots,
            "Opponent Avg SOT": opponent_avg_sot,
            "Team Shots": row["team_shots"],
            "Team SOT": row["team_sot"],
            "Team Corners": row["team_corners"]
        })

    return pd.DataFrame(rows)


# ============================================================
# STEP 43
# RECENT FORM TREND
# ============================================================

def trend_analysis(
    df,
    column
):

    if df is None or df.empty:
        return {
            "recent_avg": None,
            "older_avg": None,
            "difference": None,
            "direction": "No data"
        }

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) < 4:
        return {
            "recent_avg": values.mean()
            if len(values)
            else None,
            "older_avg": None,
            "difference": None,
            "direction": "Insufficient sample"
        }

    half = max(
        2,
        len(values) // 2
    )

    recent = values.iloc[:half]
    older = values.iloc[half:]

    recent_avg = recent.mean()
    older_avg = older.mean()
    difference = recent_avg - older_avg

    if difference > 0.25:
        direction = "Increasing"
    elif difference < -0.25:
        direction = "Decreasing"
    else:
        direction = "Stable"

    return {
        "recent_avg": recent_avg,
        "older_avg": older_avg,
        "difference": difference,
        "direction": direction
    }


# ============================================================
# STEP 44
# MARKET CONSISTENCY
# ============================================================

def market_consistency(
    df,
    market,
    direction,
    line
):

    result = analyse_market(
        df,
        market,
        direction,
        line
    )

    n = result["sample_size"]

    if n == 0:
        return {
            "hit_rate": None,
            "consistency": None,
            "longest_hit_streak": 0,
            "longest_miss_streak": 0
        }

    column = market_column(market)

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    hits = []

    for value in values:

        if direction == "Over":
            hits.append(value > line)
        else:
            hits.append(value < line)

    longest_hit = 0
    longest_miss = 0

    current_hit = 0
    current_miss = 0

    for hit in hits:

        if hit:
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

    return {
        "hit_rate": result["hit_rate"],
        "consistency": (
            result["hit_rate"]
            if result["hit_rate"] is not None
            else None
        ),
        "longest_hit_streak": longest_hit,
        "longest_miss_streak": longest_miss
    }


# ============================================================
# STEP 45
# HIT/MISS STREAK
# ============================================================

def current_streak(
    df,
    market,
    direction,
    line
):

    column = market_column(market)

    if (
        df is None
        or df.empty
        or column not in df.columns
    ):
        return {
            "type": "None",
            "length": 0
        }

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return {
            "type": "None",
            "length": 0
        }

    current_type = None
    length = 0

    for value in values:

        hit = (
            value > line
            if direction == "Over"
            else value < line
        )

        this_type = (
            "Hit"
            if hit
            else "Miss"
        )

        if current_type is None:
            current_type = this_type
            length = 1

        elif this_type == current_type:
            length += 1

        else:
            break

    return {
        "type": current_type,
        "length": length
    }


# ============================================================
# STEP 46
# LINE LADDER
# ============================================================

def line_ladder(
    df,
    market,
    direction,
    max_line=10.5
):

    column = market_column(market)

    if (
        df is None
        or df.empty
        or column not in df.columns
    ):
        return pd.DataFrame()

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return pd.DataFrame()

    rows = []

    for line in np.arange(
        0.5,
        max_line + 0.1,
        0.5
    ):

        if direction == "Over":
            hits = (values > line).sum()
        else:
            hits = (values < line).sum()

        n = len(values)

        rows.append({
            "Line": round(line, 1),
            "Hits": int(hits),
            "Misses": int(n - hits),
            "Sample": int(n),
            "Hit Rate": hits / n
        })

    result = pd.DataFrame(rows)

    result["Hit Rate"] = (
        result["Hit Rate"] * 100
    ).round(1)

    result["Hit Rate"] = (
        result["Hit Rate"].astype(str)
        + "%"
    )

    return result


# ============================================================
# STEP 47
# MARKET MATRIX
# ============================================================

def market_matrix(
    df,
    market,
    direction,
    lines=None
):

    if lines is None:
        lines = [
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            7.5
        ]

    rows = []

    for line in lines:

        result = analyse_market(
            df,
            market,
            direction,
            line
        )

        rows.append({
            "Line": line,
            "Hits": result["hits"],
            "Misses": result["misses"],
            "Sample": result["sample_size"],
            "Hit Rate": fmt_pct(
                result["hit_rate"]
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# STEP 48
# ATTACK VS DEFENCE MATCHUP INDEX
# ============================================================

def matchup_index(
    attacking_df,
    defending_df,
    market
):

    team_col = market_column(
        market
    )

    opp_col = opponent_column(
        market
    )

    attack = average_value(
        attacking_df,
        team_col
    )

    defence = average_value(
        defending_df,
        opp_col
    )

    if attack is None or defence is None:
        return None

    return {
        "attack": attack,
        "defence_conceded": defence,
        "combined": (
            attack + defence
        ) / 2
    }


# ============================================================
# STEP 49
# PRODUCED VS CONCEDED
# ============================================================

def produced_conceded_table(
    home_matches,
    away_matches
):

    rows = []

    for market in [
        "Shots",
        "Shots on Target",
        "Corners",
        "Goals"
    ]:

        col = market_column(market)
        opp = opponent_column(market)

        home_produced = average_value(
            home_matches,
            col
        )

        away_conceded = average_value(
            away_matches,
            opp
        )

        away_produced = average_value(
            away_matches,
            col
        )

        home_conceded = average_value(
            home_matches,
            opp
        )

        rows.append({
            "Market": market,
            "Home Produced": home_produced,
            "Away Conceded": away_conceded,
            "Away Produced": away_produced,
            "Home Conceded": home_conceded
        })

    return pd.DataFrame(rows)


# ============================================================
# STEP 50
# SAMPLE STABILITY
# ============================================================

def sample_stability(
    df,
    market,
    direction,
    line
):

    if df is None or df.empty:
        return pd.DataFrame()

    sizes = [
        5,
        10,
        15
    ]

    rows = []

    for n in sizes:

        sample = df.head(n)

        if sample.empty:
            continue

        result = analyse_market(
            sample,
            market,
            direction,
            line
        )

        rows.append({
            "Sample": n,
            "Available": result["sample_size"],
            "Hits": result["hits"],
            "Misses": result["misses"],
            "Hit Rate": fmt_pct(
                result["hit_rate"]
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# STEP 51
# RECENT VS LONGER SAMPLE
# ============================================================

def recent_vs_longer(
    df,
    market
):

    col = market_column(market)

    if (
        df is None
        or df.empty
        or col not in df.columns
    ):
        return None

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    if len(values) < 5:
        return {
            "Recent": values.mean()
            if len(values)
            else None,
            "Longer": None,
            "Difference": None
        }

    recent = values.head(
        min(5, len(values))
    )

    longer = values.head(
        min(15, len(values))
    )

    return {
        "Recent": recent.mean(),
        "Longer": longer.mean(),
        "Difference": (
            recent.mean()
            - longer.mean()
        )
    }


# ============================================================
# STEP 52
# DATA QUALITY / RESEARCH PANEL
# ============================================================

def data_quality_report(
    df,
    required_columns
):

    if df is None:
        return {
            "Rows": 0,
            "Missing required fields": len(
                required_columns
            ),
            "Date coverage": "—",
            "Complete metric rows": 0
        }

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    metric_columns = [
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "home_corners",
        "away_corners"
    ]

    available_metrics = [
        c
        for c in metric_columns
        if c in df.columns
    ]

    if available_metrics:
        complete_rows = int(
            df[available_metrics]
            .notna()
            .all(axis=1)
            .sum()
        )
    else:
        complete_rows = 0

    if "date" in df.columns:

        valid_dates = df["date"].dropna()

        if not valid_dates.empty:

            coverage = (
                f"{valid_dates.min().date()} "
                f"→ "
                f"{valid_dates.max().date()}"
            )
        else:
            coverage = "No valid dates"

    else:
        coverage = "No date column"

    return {
        "Rows": len(df),
        "Missing required fields": len(missing),
        "Date coverage": coverage,
        "Complete metric rows": complete_rows
    }


# ============================================================
# STEP 53
# REPORT CREATION
# ============================================================

def build_research_report(
    home_team,
    away_team,
    season,
    competition,
    market,
    direction,
    line,
    home_matches,
    away_matches,
    h2h
):

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

    home_avg = average_value(
        home_matches,
        market_column(market)
    )

    away_avg = average_value(
        away_matches,
        market_column(market)
    )

    report = []

    report.append(
        "FOOTBALL MATCH RESEARCH REPORT"
    )

    report.append(
        "=" * 50
    )

    report.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    report.append(
        f"Fixture: {home_team} vs {away_team}"
    )

    report.append(
        f"Season: {season}"
    )

    report.append(
        f"Competition: {competition}"
    )

    report.append("")
    report.append("MARKET")
    report.append("-" * 50)

    report.append(
        f"{market} {direction} {line}"
    )

    report.append(
        f"Home sample: {home_result['sample_size']} "
        f"| Hit rate: {fmt_pct(home_result['hit_rate'])}"
    )

    report.append(
        f"Away sample: {away_result['sample_size']} "
        f"| Hit rate: {fmt_pct(away_result['hit_rate'])}"
    )

    report.append("")
    report.append("AVERAGES")
    report.append("-" * 50)

    report.append(
        f"{home_team}: {fmt_num(home_avg)}"
    )

    report.append(
        f"{away_team}: {fmt_num(away_avg)}"
    )

    report.append("")
    report.append("HEAD-TO-HEAD")
    report.append("-" * 50)

    report.append(
        f"Historical meetings found: {len(h2h)}"
    )

    report.append("")
    report.append(
        "This report contains historical descriptive "
        "information and is not a prediction."
    )

    return "\n".join(report)


# ============================================================
# STEP 54
# RESEARCH CHECKLIST
# ============================================================

CHECKLIST_ITEMS = [
    "Correct fixture selected",
    "Correct competition/season selected",
    "Home/away sample checked",
    "Recent match-by-match data reviewed",
    "Opponent defensive context checked",
    "Comparable opponents considered",
    "Current line tested",
    "Alternative lines tested",
    "Recent-vs-longer sample checked",
    "Hit/miss streak reviewed",
    "Sample size considered",
    "Data quality checked",
    "Team news / lineup information checked manually",
    "Final market decision recorded separately"
]


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
            2, 0, 17, 7, 5, 2, 6, 3
        ],
        [
            "2026-09-14",
            "2026/27",
            "Premier League",
            "Everton",
            "Chelsea",
            0, 2, 9, 14, 3, 5, 4, 7
        ],
        [
            "2026-09-07",
            "2026/27",
            "Premier League",
            "Chelsea",
            "Fulham",
            3, 1, 19, 8, 7, 3, 8, 2
        ],
        [
            "2026-08-30",
            "2026/27",
            "Premier League",
            "Newcastle",
            "Chelsea",
            1, 1, 12, 11, 4, 4, 5, 5
        ],
        [
            "2026-08-24",
            "2026/27",
            "Premier League",
            "Chelsea",
            "Wolves",
            2, 1, 16, 10, 6, 3, 7, 4
        ],
        [
            "2026-09-18",
            "2026/27",
            "Bundesliga",
            "Bayern Munich",
            "Union Berlin",
            3, 1, 20, 7, 8, 2, 8, 2
        ],
        [
            "2026-09-13",
            "2026/27",
            "Bundesliga",
            "Mainz",
            "Bayern Munich",
            0, 3, 6, 18, 2, 7, 3, 8
        ],
        [
            "2026-09-06",
            "2026/27",
            "Bundesliga",
            "Bayern Munich",
            "Freiburg",
            2, 0, 17, 8, 6, 2, 9, 4
        ],
        [
            "2026-08-30",
            "2026/27",
            "Bundesliga",
            "Dortmund",
            "Bayern Munich",
            1, 2, 11, 13, 4, 5, 5, 6
        ],
        [
            "2026-08-23",
            "2026/27",
            "Bundesliga",
            "Bayern Munich",
            "Leipzig",
            4, 1, 22, 9, 9, 3, 10, 2
        ]
    ],
    columns=REQUIRED_COLUMNS
)


# ============================================================
# LOAD DATA
# ============================================================

st.sidebar.header("1. Data")

uploaded = st.sidebar.file_uploader(
    "Upload match CSV",
    type=["csv"]
)

if uploaded is not None:

    try:

        data = clean_data(
            pd.read_csv(uploaded)
        )

        st.sidebar.success(
            f"Loaded {len(data)} matches"
        )

    except Exception as exc:

        st.error(
            f"Could not read CSV: {exc}"
        )

        st.stop()

elif os.path.exists(MASTER_FILE):

    try:

        data = clean_data(
            pd.read_csv(MASTER_FILE)
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

    data = clean_data(demo)

    st.sidebar.info(
        f"Demo dataset — {len(data)} matches"
    )


missing = [
    c
    for c in REQUIRED_COLUMNS
    if c not in data.columns
]

if missing:

    st.error(
        "Your CSV is missing: "
        + ", ".join(missing)
    )

    st.stop()


# ============================================================
# SIDEBAR DATA QUALITY
# ============================================================

with st.sidebar.expander(
    "Dataset Quality"
):

    quality = data_quality_report(
        data,
        REQUIRED_COLUMNS
    )

    st.write(
        f"Rows: {quality['Rows']}"
    )

    st.write(
        f"Complete metric rows: "
        f"{quality['Complete metric rows']}"
    )

    st.write(
        f"Date coverage: "
        f"{quality['Date coverage']}"
    )

    st.write(
        f"Missing required fields: "
        f"{quality['Missing required fields']}"
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🔎 Team Research",
    "🎯 Market Tester",
    "📊 Bookmaker Monitor",
    "📥 Data Format",
    "🔬 Match Research",
    "🧠 Advanced Research 41–54"
])

tab1, tab2, tab3, tab4, tab5, tab6 = tabs


# ============================================================
# TAB 1 — TEAM RESEARCH
# ============================================================

with tab1:

    st.subheader("Team Research Dashboard")

    teams = team_list(data)

    selected_team = st.selectbox(
        "Team",
        teams,
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
        selected_team,
        season,
        competition,
        venue,
        sample
    )

    q_label, q_message = sample_quality(
        len(df)
    )

    if len(df) < 5:
        st.warning(
            f"{q_label}: {q_message}"
        )
    else:
        st.info(
            f"{q_label}: {q_message}"
        )

    summary = team_summary(df)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Matches",
        len(df)
    )

    c2.metric(
        "Avg Shots",
        fmt_num(summary["Shots"])
    )

    c3.metric(
        "Avg SOT",
        fmt_num(summary["Shots on Target"])
    )

    c4.metric(
        "Avg Corners",
        fmt_num(summary["Corners"])
    )

    c5.metric(
        "Avg Goals",
        fmt_num(summary["Goals"])
    )

    st.divider()

    st.subheader("Recent Match History")

    if not df.empty:

        cols = [
            "date",
            "home_team",
            "away_team",
            "venue",
            "team_shots",
            "team_sot",
            "team_corners",
            "team_goals"
        ]

        st.dataframe(
            df[
                [
                    c
                    for c in cols
                    if c in df.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No matches found.")


# ============================================================
# TAB 2 — MARKET TESTER
# ============================================================

with tab2:

    st.subheader("🎯 Market Tester")

    team = st.selectbox(
        "Team",
        team_list(data),
        key="tester_team"
    )

    season2 = st.selectbox(
        "Season",
        season_list(data),
        key="tester_season"
    )

    competition2 = st.selectbox(
        "Competition",
        competition_list(data),
        key="tester_competition"
    )

    venue2 = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="tester_venue"
    )

    market = st.selectbox(
        "Market",
        list(MARKET_COLUMN_MAP.keys()),
        key="tester_market"
    )

    direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="tester_direction"
    )

    line = st.number_input(
        "Line",
        0.0,
        30.0,
        3.5,
        0.5,
        key="tester_line"
    )

    odds = st.number_input(
        "Decimal odds",
        1.01,
        100.0,
        1.30,
        0.01,
        key="tester_odds"
    )

    sample2 = st.select_slider(
        "Sample",
        options=[5, 10, 15],
        value=10,
        key="tester_sample"
    )

    test_df = filtered_team_matches(
        data,
        team,
        season2,
        competition2,
        venue2,
        sample2
    )

    result = analyse_market(
        test_df,
        market,
        direction,
        line
    )

    rate = result["hit_rate"]

    breakeven = 1 / odds

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Historical Hit Rate",
        fmt_pct(rate)
    )

    c2.metric(
        "Break-even",
        f"{breakeven * 100:.1f}%"
    )

    c3.metric(
        "Hits",
        result["hits"]
    )

    c4.metric(
        "Misses",
        result["misses"]
    )

    if rate is not None:

        st.metric(
            "Historical difference vs break-even",
            f"{(rate - breakeven) * 100:+.1f} pp"
        )

    st.caption(
        "Historical results are descriptive and do not establish "
        "the probability of the next match."
    )

    st.divider()

    history = market_history(
        test_df,
        market,
        direction,
        line
    )

    if not history.empty:

        st.subheader("Match-by-Match History")

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 3 — BOOKMAKER MONITOR
# ============================================================

with tab3:

    st.subheader("📊 Bookmaker Market Monitor")

    if "market_watchlist" not in st.session_state:

        if os.path.exists(WATCHLIST_FILE):

            try:

                loaded = pd.read_csv(
                    WATCHLIST_FILE
                ).fillna("")

                if "Status" not in loaded.columns:
                    loaded["Status"] = "Watching"

                if "Actual Result" not in loaded.columns:
                    loaded["Actual Result"] = ""

                st.session_state.market_watchlist = (
                    loaded.to_dict("records")
                )

            except Exception:
                st.session_state.market_watchlist = []

        else:
            st.session_state.market_watchlist = []

    b1, b2 = st.columns(2)

    with b1:

        bookmaker = st.selectbox(
            "Bookmaker",
            [
                "SportyBet",
                "SunBet",
                "Virgin Bet"
            ],
            key="bookmaker"
        )

        match_name = st.text_input(
            "Match",
            key="book_match"
        )

        monitor_team = st.selectbox(
            "Team",
            team_list(data),
            key="book_team"
        )

        monitor_market = st.selectbox(
            "Market",
            list(MARKET_COLUMN_MAP.keys()),
            key="book_market"
        )

    with b2:

        monitor_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="book_direction"
        )

        monitor_line = st.number_input(
            "Line",
            0.0,
            30.0,
            3.5,
            0.5,
            key="book_line"
        )

        monitor_odds = st.number_input(
            "Odds",
            1.01,
            100.0,
            1.30,
            0.01,
            key="book_odds"
        )

    if st.button(
        "Analyse & Add Market",
        key="book_add"
    ):

        monitor_df = filtered_team_matches(
            data,
            monitor_team,
            "All",
            "All",
            "All",
            10
        )

        result = analyse_market(
            monitor_df,
            monitor_market,
            monitor_direction,
            monitor_line
        )

        entry = {
            "Bookmaker": bookmaker,
            "Match": match_name,
            "Team": monitor_team,
            "Market": monitor_market,
            "Direction": monitor_direction,
            "Line": monitor_line,
            "Odds": monitor_odds,
            "Historical Hit Rate":
                fmt_pct(result["hit_rate"]),
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

    if st.session_state.market_watchlist:

        monitor_df = pd.DataFrame(
            st.session_state.market_watchlist
        )

        st.dataframe(
            monitor_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = monitor_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Watchlist",
            csv_data,
            "market_watchlist.csv",
            "text/csv",
            key="download_monitor"
        )


# ============================================================
# TAB 4 — DATA FORMAT
# ============================================================

with tab4:

    st.subheader("📥 CSV Format")

    st.write(
        "Your real dataset should contain:"
    )

    st.code(
        ",".join(REQUIRED_COLUMNS)
    )

    st.dataframe(
        data.head(10),
        use_container_width=True,
        hide_index=True
    )

    demo_bytes = demo.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Demo CSV",
        demo_bytes,
        "football_demo.csv",
        "text/csv",
        key="download_demo"
    )


# ============================================================
# TAB 5 — MATCH RESEARCH
# ============================================================

with tab5:

    st.subheader("🔬 Match Research")

    teams = team_list(data)

    r1, r2 = st.columns(2)

    with r1:

        home_team = st.selectbox(
            "Home Team",
            teams,
            key="match_home"
        )

    with r2:

        away_options = [
            x for x in teams
            if x != home_team
        ]

        away_team = st.selectbox(
            "Away Team",
            away_options,
            key="match_away"
        )

    r3, r4, r5 = st.columns(3)

    with r3:

        research_season = st.selectbox(
            "Season",
            season_list(data),
            key="match_season"
        )

    with r4:

        research_comp = st.selectbox(
            "Competition",
            competition_list(data),
            key="match_comp"
        )

    with r5:

        research_sample = st.selectbox(
            "Sample",
            [5, 10, 15],
            index=1,
            key="match_sample"
        )

    home_matches = filtered_team_matches(
        data,
        home_team,
        research_season,
        research_comp,
        "Home",
        research_sample
    )

    away_matches = filtered_team_matches(
        data,
        away_team,
        research_season,
        research_comp,
        "Away",
        research_sample
    )

    st.divider()

    st.subheader("Home vs Away")

    rows = []

    for label, col in [
        ("Shots", "team_shots"),
        ("SOT", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:

        rows.append({
            "Metric": label,
            home_team:
                average_value(
                    home_matches,
                    col
                ),
            away_team:
                average_value(
                    away_matches,
                    col
                )
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Attack vs Defence")

    matchup_rows = []

    for market_name in MARKET_COLUMN_MAP:

        home_attack = average_value(
            home_matches,
            market_column(market_name)
        )

        away_defence = average_value(
            away_matches,
            opponent_column(market_name)
        )

        away_attack = average_value(
            away_matches,
            market_column(market_name)
        )

        home_defence = average_value(
            home_matches,
            opponent_column(market_name)
        )

        matchup_rows.append({
            "Market": market_name,
            "Home Produced": home_attack,
            "Away Conceded": away_defence,
            "Away Produced": away_attack,
            "Home Conceded": home_defence
        })

    st.dataframe(
        pd.DataFrame(matchup_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Head-to-Head")

    h2h = get_match_history(
        data,
        home_team,
        away_team
    )

    if not h2h.empty:

        st.dataframe(
            h2h.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No historical meetings found."
        )


# ============================================================
# TAB 6 — ADVANCED RESEARCH 41–54
# ============================================================

with tab6:

    st.subheader(
        "🧠 Advanced Research Engine — Steps 41–54"
    )

    st.caption(
        "This section adds deeper historical testing without "
        "turning the application into an automatic prediction engine."
    )

    # --------------------------------------------------------
    # FIXTURE
    # --------------------------------------------------------

    teams = team_list(data)

    a1, a2 = st.columns(2)

    with a1:

        adv_home = st.selectbox(
            "Home Team",
            teams,
            key="adv_home"
        )

    with a2:

        adv_away_options = [
            x for x in teams
            if x != adv_home
        ]

        adv_away = st.selectbox(
            "Away Team",
            adv_away_options,
            key="adv_away"
        )

    a3, a4, a5 = st.columns(3)

    with a3:

        adv_season = st.selectbox(
            "Season",
            season_list(data),
            key="adv_season"
        )

    with a4:

        adv_comp = st.selectbox(
            "Competition",
            competition_list(data),
            key="adv_comp"
        )

    with a5:

        adv_sample = st.selectbox(
            "Primary sample",
            [5, 10, 15],
            index=2,
            key="adv_sample"
        )

    adv_market = st.selectbox(
        "Research Market",
        list(MARKET_COLUMN_MAP.keys()),
        key="adv_market"
    )

    adv_direction = st.selectbox(
        "Market Direction",
        ["Over", "Under"],
        key="adv_direction"
    )

    adv_line = st.number_input(
        "Market Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5,
        key="adv_line"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    adv_home_matches = filtered_team_matches(
        data,
        adv_home,
        adv_season,
        adv_comp,
        "Home",
        adv_sample
    )

    adv_away_matches = filtered_team_matches(
        data,
        adv_away,
        adv_season,
        adv_comp,
        "Away",
        adv_sample
    )

    adv_home_all = filtered_team_matches(
        data,
        adv_home,
        adv_season,
        adv_comp,
        "All",
        15
    )

    adv_away_all = filtered_team_matches(
        data,
        adv_away,
        adv_season,
        adv_comp,
        "All",
        15
    )

    # ========================================================
    # STEP 41 — COMPARABLE OPPONENT RESEARCH
    # ========================================================

    st.divider()

    st.subheader(
        "41. Comparable Opponent Research"
    )

    home_comp = comparable_opponents(
        data,
        adv_home,
        "Home",
        adv_sample
    )

    away_comp = comparable_opponents(
        data,
        adv_away,
        "Away",
        adv_sample
    )

    c1, c2 = st.columns(2)

    with c1:

        st.caption(
            f"{adv_home} — comparable home sample"
        )

        if not home_comp.empty:

            st.dataframe(
                home_comp[
                    [
                        "date",
                        "home_team",
                        "away_team",
                        "team_shots",
                        "team_sot",
                        "team_corners"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No comparable sample.")

    with c2:

        st.caption(
            f"{adv_away} — comparable away sample"
        )

        if not away_comp.empty:

            st.dataframe(
                away_comp[
                    [
                        "date",
                        "home_team",
                        "away_team",
                        "team_shots",
                        "team_sot",
                        "team_corners"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No comparable sample.")

    # ========================================================
    # STEP 42 — OPPONENT STRENGTH
    # ========================================================

    st.subheader(
        "42. Opponent Strength Context"
    )

    home_strength = opponent_strength_table(
        data,
        adv_home,
        "Home",
        adv_sample
    )

    away_strength = opponent_strength_table(
        data,
        adv_away,
        "Away",
        adv_sample
    )

    if not home_strength.empty:

        st.caption(
            f"{adv_home} opponent context"
        )

        st.dataframe(
            home_strength,
            use_container_width=True,
            hide_index=True
        )

    if not away_strength.empty:

        st.caption(
            f"{adv_away} opponent context"
        )

        st.dataframe(
            away_strength,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 43 — TREND
    # ========================================================

    st.subheader(
        "43. Recent-Form Trend"
    )

    trend_col = market_column(
        adv_market
    )

    home_trend = trend_analysis(
        adv_home_matches,
        trend_col
    )

    away_trend = trend_analysis(
        adv_away_matches,
        trend_col
    )

    trend_df = pd.DataFrame([
        {
            "Team": adv_home,
            "Recent Avg":
                home_trend["recent_avg"],
            "Older Avg":
                home_trend["older_avg"],
            "Difference":
                home_trend["difference"],
            "Trend":
                home_trend["direction"]
        },
        {
            "Team": adv_away,
            "Recent Avg":
                away_trend["recent_avg"],
            "Older Avg":
                away_trend["older_avg"],
            "Difference":
                away_trend["difference"],
            "Trend":
                away_trend["direction"]
        }
    ])

    st.dataframe(
        trend_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 44 — CONSISTENCY
    # ========================================================

    st.subheader(
        "44. Market Consistency"
    )

    home_consistency = market_consistency(
        adv_home_matches,
        adv_market,
        adv_direction,
        adv_line
    )

    away_consistency = market_consistency(
        adv_away_matches,
        adv_market,
        adv_direction,
        adv_line
    )

    consistency_df = pd.DataFrame([
        {
            "Team": adv_home,
            "Hit Rate":
                fmt_pct(
                    home_consistency["hit_rate"]
                ),
            "Longest Hit Streak":
                home_consistency[
                    "longest_hit_streak"
                ],
            "Longest Miss Streak":
                home_consistency[
                    "longest_miss_streak"
                ]
        },
        {
            "Team": adv_away,
            "Hit Rate":
                fmt_pct(
                    away_consistency["hit_rate"]
                ),
            "Longest Hit Streak":
                away_consistency[
                    "longest_hit_streak"
                ],
            "Longest Miss Streak":
                away_consistency[
                    "longest_miss_streak"
                ]
        }
    ])

    st.dataframe(
        consistency_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 45 — CURRENT STREAK
    # ========================================================

    st.subheader(
        "45. Current Hit / Miss Streak"
    )

    home_streak = current_streak(
        adv_home_matches,
        adv_market,
        adv_direction,
        adv_line
    )

    away_streak = current_streak(
        adv_away_matches,
        adv_market,
        adv_direction,
        adv_line
    )

    streak_df = pd.DataFrame([
        {
            "Team": adv_home,
            "Current Streak":
                home_streak["type"],
            "Length":
                home_streak["length"]
        },
        {
            "Team": adv_away,
            "Current Streak":
                away_streak["type"],
            "Length":
                away_streak["length"]
        }
    ])

    st.dataframe(
        streak_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 46 — LINE LADDER
    # ========================================================

    st.subheader(
        "46. Line Ladder"
    )

    ladder_team = st.selectbox(
        "Team for line ladder",
        [adv_home, adv_away],
        key="ladder_team"
    )

    ladder_df_source = (
        adv_home_matches
        if ladder_team == adv_home
        else adv_away_matches
    )

    ladder = line_ladder(
        ladder_df_source,
        adv_market,
        adv_direction,
        max_line=10.5
    )

    if not ladder.empty:

        st.dataframe(
            ladder,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 47 — MARKET MATRIX
    # ========================================================

    st.subheader(
        "47. Multi-Line Market Matrix"
    )

    matrix_team = st.selectbox(
        "Team for market matrix",
        [adv_home, adv_away],
        key="matrix_team"
    )

    matrix_source = (
        adv_home_matches
        if matrix_team == adv_home
        else adv_away_matches
    )

    matrix = market_matrix(
        matrix_source,
        adv_market,
        adv_direction
    )

    st.dataframe(
        matrix,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 48 — MATCHUP INDEX
    # ========================================================

    st.subheader(
        "48. Attack vs Defensive Matchup"
    )

    home_matchup = matchup_index(
        adv_home_matches,
        adv_away_matches,
        adv_market
    )

    away_matchup = matchup_index(
        adv_away_matches,
        adv_home_matches,
        adv_market
    )

    matchup_rows = []

    if home_matchup:

        matchup_rows.append({
            "Team": adv_home,
            "Attack Produced":
                home_matchup["attack"],
            "Opponent Conceded":
                home_matchup[
                    "defence_conceded"
                ],
            "Combined Context":
                home_matchup["combined"]
        })

    if away_matchup:

        matchup_rows.append({
            "Team": adv_away,
            "Attack Produced":
                away_matchup["attack"],
            "Opponent Conceded":
                away_matchup[
                    "defence_conceded"
                ],
            "Combined Context":
                away_matchup["combined"]
        })

    if matchup_rows:

        st.dataframe(
            pd.DataFrame(matchup_rows),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 49 — PRODUCED VS CONCEDED
    # ========================================================

    st.subheader(
        "49. Produced vs Conceded"
    )

    pc_df = produced_conceded_table(
        adv_home_matches,
        adv_away_matches
    )

    st.dataframe(
        pc_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 50 — SAMPLE STABILITY
    # ========================================================

    st.subheader(
        "50. Sample Stability"
    )

    home_stability = sample_stability(
        adv_home_all,
        adv_market,
        adv_direction,
        adv_line
    )

    away_stability = sample_stability(
        adv_away_all,
        adv_market,
        adv_direction,
        adv_line
    )

    s1, s2 = st.columns(2)

    with s1:

        st.caption(
            f"{adv_home} stability"
        )

        st.dataframe(
            home_stability,
            use_container_width=True,
            hide_index=True
        )

    with s2:

        st.caption(
            f"{adv_away} stability"
        )

        st.dataframe(
            away_stability,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 51 — RECENT VS LONGER
    # ========================================================

    st.subheader(
        "51. Recent vs Longer Sample"
    )

    home_recent_longer = recent_vs_longer(
        adv_home_all,
        adv_market
    )

    away_recent_longer = recent_vs_longer(
        adv_away_all,
        adv_market
    )

    recent_longer_df = pd.DataFrame([
        {
            "Team": adv_home,
            "Recent Avg":
                (
                    home_recent_longer["Recent"]
                    if home_recent_longer
                    else None
                ),
            "15-Match Avg":
                (
                    home_recent_longer["Longer"]
                    if home_recent_longer
                    else None
                ),
            "Difference":
                (
                    home_recent_longer["Difference"]
                    if home_recent_longer
                    else None
                )
        },
        {
            "Team": adv_away,
            "Recent Avg":
                (
                    away_recent_longer["Recent"]
                    if away_recent_longer
                    else None
                ),
            "15-Match Avg":
                (
                    away_recent_longer["Longer"]
                    if away_recent_longer
                    else None
                ),
            "Difference":
                (
                    away_recent_longer["Difference"]
                    if away_recent_longer
                    else None
                )
        }
    ])

    st.dataframe(
        recent_longer_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 52 — DATA QUALITY
    # ========================================================

    st.subheader(
        "52. Research Confidence / Data Quality"
    )

    home_quality = data_quality_report(
        adv_home_matches,
        REQUIRED_COLUMNS
    )

    away_quality = data_quality_report(
        adv_away_matches,
        REQUIRED_COLUMNS
    )

    quality_df = pd.DataFrame([
        {
            "Team": adv_home,
            "Sample": len(adv_home_matches),
            "Complete Metric Rows":
                home_quality[
                    "Complete metric rows"
                ],
            "Date Coverage":
                home_quality[
                    "Date coverage"
                ]
        },
        {
            "Team": adv_away,
            "Sample": len(adv_away_matches),
            "Complete Metric Rows":
                away_quality[
                    "Complete metric rows"
                ],
            "Date Coverage":
                away_quality[
                    "Date coverage"
                ]
        }
    ])

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "A larger or cleaner historical sample improves "
        "research quality, but it does not guarantee a future result."
    )

    # ========================================================
    # STEP 53 — EXPORT REPORT
    # ========================================================

    st.subheader(
        "53. Fixture Research Report"
    )

    h2h_report = get_match_history(
        data,
        adv_home,
        adv_away
    )

    report = build_research_report(
        adv_home,
        adv_away,
        adv_season,
        adv_comp,
        adv_market,
        adv_direction,
        adv_line,
        adv_home_matches,
        adv_away_matches,
        h2h_report
    )

    st.text_area(
        "Generated research report",
        report,
        height=300,
        key="generated_report"
    )

    st.download_button(
        "Download Research Report",
        report.encode("utf-8"),
        f"{adv_home}_vs_{adv_away}_research.txt",
        "text/plain",
        key="download_research_report"
    )

    # ========================================================
    # STEP 54 — DECISION AUDIT
    # ========================================================

    st.subheader(
        "54. Research Decision Checklist"
    )

    st.caption(
        "Complete the checklist before treating a market "
        "as sufficiently researched."
    )

    checklist_results = {}

    for i, item in enumerate(
        CHECKLIST_ITEMS
    ):

        checklist_results[item] = st.checkbox(
            item,
            key=f"checklist_{i}"
        )

    completed = sum(
        checklist_results.values()
    )

    total_checks = len(
        CHECKLIST_ITEMS
    )

    st.progress(
        completed / total_checks
    )

    st.metric(
        "Research checklist completed",
        f"{completed}/{total_checks}"
    )

    notes = st.text_area(
        "Final research notes",
        placeholder=(
            "Record what the historical evidence shows, "
            "what remains uncertain, lineup/team-news information, "
            "and why you are or are not comfortable researching "
            "this market further."
        ),
        height=180,
        key="advanced_final_notes"
    )

    audit = pd.DataFrame([
        {
            "Checklist Item": item,
            "Completed":
                "Yes"
                if checklist_results[item]
                else "No"
        }
        for item in CHECKLIST_ITEMS
    ])

    audit_csv = audit.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Research Checklist",
        audit_csv,
        "research_checklist.csv",
        "text/csv",
        key="download_checklist"
    )

    st.caption(
        "The checklist is an audit tool. It does not produce "
        "a safe-bet label, prediction, probability, or automatic pick."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Football Betting Research Hub — historical research only. "
    "Match-by-match evidence should be combined with current "
    "team news, lineups, tactical context and bookmaker market "
    "information before any betting decision."
)
