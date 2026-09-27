
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================
# FOOTBALL BETTING RESEARCH HUB — V1
# STEPS 19–40 INTEGRATED
# Research assistant, not a prediction engine.
# ============================================================

st.set_page_config(
    page_title="Football Betting Research V1",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Betting Research Tool — V1")
st.caption(
    "Research assistant, not a prediction engine. "
    "Use match-by-match data to test markets and compare historical evidence."
)

# ============================================================
# FILE CONFIGURATION
# ============================================================

MASTER_FILE = "football_master_2024_27_v1.csv"
WATCHLIST_FILE = "market_watchlist.csv"
RESEARCH_FILE = "match_research_history.csv"

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

LINES_BY_MARKET = {
    "Shots": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
               10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
               18.5, 19.5, 20.5],
    "Shots on Target": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                        8.5, 9.5, 10.5],
    "Corners": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                10.5, 11.5, 12.5],
    "Goals": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
}


# ============================================================
# STEP 19 — DATA CLEANING
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
                .replace("nan", "")
            )

    numeric_columns = [
        c for c in REQUIRED_COLUMNS
        if c not in text_columns + ["date"]
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

    df = df.sort_values(
        "date",
        ascending=False
    ).reset_index(drop=True)

    return df


# ============================================================
# STEP 20 — TEAM PERSPECTIVE ENGINE
# ============================================================

@st.cache_data
def team_matches(df, team):

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

    return out.sort_values(
        "date",
        ascending=False
    ).reset_index(drop=True)


@st.cache_data
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
        m = m[m["season"] == season]

    if competition != "All":
        m = m[m["competition"] == competition]

    if venue != "All":
        m = m[m["venue"] == venue]

    return m.sort_values(
        "date",
        ascending=False
    ).head(sample).copy()


# ============================================================
# STEP 21 — MARKET ANALYSIS ENGINE
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
        "pushes": 0,
        "sample_size": 0
    }

    if df is None or df.empty:
        return empty

    column = MARKET_COLUMN_MAP.get(market)

    if column not in df.columns:
        return empty

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return empty

    if direction == "Over":
        hits = int((values > line).sum())
        pushes = int((values == line).sum())
    else:
        hits = int((values < line).sum())
        pushes = int((values == line).sum())

    sample_size = len(values)
    misses = sample_size - hits - pushes

    decisive = hits + misses

    rate = (
        hits / decisive
        if decisive > 0
        else None
    )

    return {
        "hit_rate": rate,
        "hits": hits,
        "misses": misses,
        "pushes": pushes,
        "sample_size": sample_size
    }


def market_history(
    df,
    market,
    direction,
    line
):

    if df is None or df.empty:
        return pd.DataFrame()

    column = MARKET_COLUMN_MAP.get(market)

    if column not in df.columns:
        return pd.DataFrame()

    result = df[
        [
            c for c in [
                "date",
                "home_team",
                "away_team",
                "venue",
                column
            ]
            if c in df.columns
        ]
    ].copy()

    result[column] = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    if direction == "Over":
        result["Result"] = np.where(
            result[column] > line,
            "✓ Hit",
            np.where(
                result[column] == line,
                "Push",
                "✗ Miss"
            )
        )
    else:
        result["Result"] = np.where(
            result[column] < line,
            "✓ Hit",
            np.where(
                result[column] == line,
                "Push",
                "✗ Miss"
            )
        )

    return result


# ============================================================
# STEP 22 — STATISTICS
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


def std_value(df, column):

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

    if len(values) < 2:
        return None

    return float(values.std())


def min_value(df, column):

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

    return float(values.min())


def max_value(df, column):

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

    return float(values.max())


def fmt(value, decimals=2):

    if value is None or pd.isna(value):
        return "—"

    return f"{value:.{decimals}f}"


def fmt_pct(value):

    if value is None or pd.isna(value):
        return "—"

    return f"{value * 100:.1f}%"


def sample_quality(n):

    if n == 0:
        return (
            "No data",
            "No historical matches are available."
        )

    if n < 5:
        return (
            "Very small sample",
            "Fewer than 5 matches are available."
        )

    if n < 10:
        return (
            "Small sample",
            "A limited historical sample is available."
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
# STEP 23 — LIST FUNCTIONS
# ============================================================

@st.cache_data
def team_list(data):

    return sorted(
        set(data["home_team"].dropna())
        |
        set(data["away_team"].dropna())
    )


@st.cache_data
def season_list(data):

    return (
        ["All"]
        +
        sorted(
            data["season"]
            .dropna()
            .unique()
            .tolist(),
            reverse=True
        )
    )


@st.cache_data
def competition_list(data):

    return (
        ["All"]
        +
        sorted(
            data["competition"]
            .dropna()
            .unique()
            .tolist()
        )
    )


# ============================================================
# STEP 24 — TEAM SUMMARY
# ============================================================

def team_summary(df):

    return {
        "Matches": len(df),
        "Shots": average_value(df, "team_shots"),
        "SOT": average_value(df, "team_sot"),
        "Corners": average_value(df, "team_corners"),
        "Goals": average_value(df, "team_goals"),
        "Shots Median": median_value(df, "team_shots"),
        "SOT Median": median_value(df, "team_sot"),
        "Corners Median": median_value(df, "team_corners"),
        "Goals Median": median_value(df, "team_goals")
    }


# ============================================================
# STEP 25 — H2H ENGINE
# ============================================================

@st.cache_data
def get_h2h(data, home_team, away_team):

    return data[
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
    ].sort_values(
        "date",
        ascending=False
    ).copy()


# ============================================================
# STEP 26 — MARKET LINE LADDER
# ============================================================

def market_line_ladder(
    df,
    market,
    direction
):

    rows = []

    for line in LINES_BY_MARKET.get(
        market,
        []
    ):

        result = analyse_market(
            df,
            market,
            direction,
            line
        )

        if result["sample_size"] > 0:

            rows.append(
                {
                    "Market": market,
                    "Direction": direction,
                    "Line": line,
                    "Hits": result["hits"],
                    "Misses": result["misses"],
                    "Pushes": result["pushes"],
                    "Sample": result["sample_size"],
                    "Hit Rate": fmt_pct(
                        result["hit_rate"]
                    )
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# STEP 27 — MULTI-MARKET COMPARISON
# ============================================================

def multi_market_table(df):

    rows = []

    for market in MARKETS:

        column = MARKET_COLUMN_MAP[market]

        avg = average_value(
            df,
            column
        )

        med = median_value(
            df,
            column
        )

        std = std_value(
            df,
            column
        )

        minimum = min_value(
            df,
            column
        )

        maximum = max_value(
            df,
            column
        )

        rows.append(
            {
                "Market": market,
                "Average": fmt(avg),
                "Median": fmt(med),
                "Std Dev": fmt(std),
                "Minimum": fmt(minimum),
                "Maximum": fmt(maximum)
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 28 — RECENT TREND ANALYSIS
# ============================================================

def trend_table(
    df,
    market,
    direction,
    line
):

    if df is None or df.empty:
        return pd.DataFrame()

    column = MARKET_COLUMN_MAP[market]

    working = df.copy()

    working[column] = pd.to_numeric(
        working[column],
        errors="coerce"
    )

    working = working.dropna(
        subset=[column]
    )

    if working.empty:
        return pd.DataFrame()

    rows = []

    for n in [5, 10, 15]:

        sample = working.head(n)

        if sample.empty:
            continue

        result = analyse_market(
            sample,
            market,
            direction,
            line
        )

        rows.append(
            {
                "Sample": f"Last {n}",
                "Matches": result["sample_size"],
                "Hits": result["hits"],
                "Misses": result["misses"],
                "Pushes": result["pushes"],
                "Hit Rate": fmt_pct(
                    result["hit_rate"]
                ),
                "Average": fmt(
                    average_value(
                        sample,
                        column
                    )
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 29 — CONSISTENCY PROFILE
# ============================================================

def consistency_profile(df, market):

    column = MARKET_COLUMN_MAP[market]

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return {}

    return {
        "Average": values.mean(),
        "Median": values.median(),
        "Minimum": values.min(),
        "Maximum": values.max(),
        "Std Dev": values.std(),
        "Range": values.max() - values.min()
    }


# ============================================================
# STEP 30 — ATTACK / DEFENCE CONTEXT
# ============================================================

def attack_defence_table(
    home_matches,
    away_matches
):

    rows = []

    mappings = [
        ("Shots", "team_shots", "opp_shots"),
        ("Shots on Target", "team_sot", "opp_sot"),
        ("Corners", "team_corners", "opp_corners"),
        ("Goals", "team_goals", "opp_goals")
    ]

    for label, team_col, opp_col in mappings:

        rows.append(
            {
                "Metric": label,
                "Home Attack": fmt(
                    average_value(
                        home_matches,
                        team_col
                    )
                ),
                "Away Defence Conceded": fmt(
                    average_value(
                        away_matches,
                        opp_col
                    )
                ),
                "Away Attack": fmt(
                    average_value(
                        away_matches,
                        team_col
                    )
                ),
                "Home Defence Conceded": fmt(
                    average_value(
                        home_matches,
                        opp_col
                    )
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 31 — CANDIDATE MARKET MATRIX
# ============================================================

def candidate_market_matrix(
    home_matches,
    away_matches
):

    rows = []

    for market in MARKETS:

        for direction in ["Over", "Under"]:

            for line in LINES_BY_MARKET[market]:

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

                if (
                    home_result["sample_size"] > 0
                    or
                    away_result["sample_size"] > 0
                ):

                    rows.append(
                        {
                            "Market": market,
                            "Direction": direction,
                            "Line": line,
                            "Home Hit Rate": fmt_pct(
                                home_result["hit_rate"]
                            ),
                            "Away Hit Rate": fmt_pct(
                                away_result["hit_rate"]
                            ),
                            "Home Sample": home_result[
                                "sample_size"
                            ],
                            "Away Sample": away_result[
                                "sample_size"
                            ]
                        }
                    )

    return pd.DataFrame(rows)


# ============================================================
# STEP 32 — ODDS / BREAK-EVEN TOOLS
# ============================================================

def break_even_probability(odds):

    if odds is None or odds <= 1:
        return None

    return 1 / odds


def implied_probability(odds):

    return break_even_probability(odds)


def historical_difference(
    hit_rate,
    odds
):

    if hit_rate is None:
        return None

    breakeven = break_even_probability(
        odds
    )

    if breakeven is None:
        return None

    return hit_rate - breakeven


# ============================================================
# STEP 33 — RESEARCH QUALITY INDICATOR
# ============================================================

def research_quality(
    home_n,
    away_n,
    h2h_n
):

    score = 0
    notes = []

    if home_n >= 15:
        score += 2
    elif home_n >= 10:
        score += 1
    else:
        notes.append("Limited home sample")

    if away_n >= 15:
        score += 2
    elif away_n >= 10:
        score += 1
    else:
        notes.append("Limited away sample")

    if h2h_n >= 5:
        score += 1
    else:
        notes.append("Limited H2H sample")

    if score >= 5:
        label = "Broad historical coverage"
    elif score >= 3:
        label = "Moderate historical coverage"
    else:
        label = "Limited historical coverage"

    return label, notes


# ============================================================
# STEP 34 — RESEARCH HISTORY STORAGE
# ============================================================

def load_research_history():

    if os.path.exists(RESEARCH_FILE):

        try:
            return pd.read_csv(
                RESEARCH_FILE
            ).fillna("")

        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def save_research_record(record):

    existing = load_research_history()

    new_row = pd.DataFrame([record])

    combined = pd.concat(
        [existing, new_row],
        ignore_index=True
    )

    combined.to_csv(
        RESEARCH_FILE,
        index=False
    )

    return combined


# ============================================================
# STEP 35 — WATCHLIST HELPERS
# ============================================================

def load_watchlist():

    if os.path.exists(WATCHLIST_FILE):

        try:
            return pd.read_csv(
                WATCHLIST_FILE
            ).fillna("").to_dict("records")

        except Exception:
            return []

    return []


def save_watchlist(items):

    pd.DataFrame(
        items
    ).to_csv(
        WATCHLIST_FILE,
        index=False
    )


# ============================================================
# STEP 36 — LOAD REAL DATA
# ============================================================

st.sidebar.header("1. Data")

uploaded = st.sidebar.file_uploader(
    "Upload match CSV",
    type=["csv"]
)

if uploaded is not None:

    try:

        raw_data = pd.read_csv(
            uploaded
        )

        data = clean_data(
            raw_data
        )

        st.sidebar.success(
            f"Uploaded dataset: {len(data):,} matches"
        )

    except Exception as exc:

        st.error(
            f"Could not read uploaded CSV: {exc}"
        )

        st.stop()

else:

    if os.path.exists(MASTER_FILE):

        try:

            raw_data = pd.read_csv(
                MASTER_FILE
            )

            data = clean_data(
                raw_data
            )

            st.sidebar.success(
                f"Real master dataset: {len(data):,} matches"
            )

        except Exception as exc:

            st.error(
                f"Could not read {MASTER_FILE}: {exc}"
            )

            st.stop()

    else:

        st.error(
            f"{MASTER_FILE} was not found."
        )

        st.info(
            "Place your real football_master_2024_27_v1.csv "
            "in the same folder as app.py."
        )

        st.stop()


# ============================================================
# STEP 37 — DATA VALIDATION
# ============================================================

missing = [
    c
    for c in REQUIRED_COLUMNS
    if c not in data.columns
]

if missing:

    st.error(
        "Your CSV is missing required columns:\n\n"
        + ", ".join(missing)
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.sidebar.expander(
    "Dataset information",
    expanded=False
):

    st.write(
        f"**Matches:** {len(data):,}"
    )

    st.write(
        f"**Teams:** {len(team_list(data)):,}"
    )

    st.write(
        f"**Seasons:** {data['season'].nunique():,}"
    )

    st.write(
        f"**Competitions:** {data['competition'].nunique():,}"
    )

    if not data.empty:

        st.write(
            f"**Latest date:** "
            f"{data['date'].max().date()}"
        )

        st.write(
            f"**Oldest date:** "
            f"{data['date'].min().date()}"
        )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🔎 Team Research",
        "🎯 Market Tester",
        "📊 Bookmaker Monitor",
        "📥 Data Format",
        "🔬 Match Research",
        "📚 Research History"
    ]
)


# ============================================================
# TAB 1 — TEAM RESEARCH
# ============================================================

with tab1:

    st.subheader(
        "Team Research Dashboard"
    )

    teams = team_list(data)

    dashboard_team = st.selectbox(
        "Team",
        teams,
        key="dashboard_team"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dashboard_season = st.selectbox(
            "Season",
            season_list(data),
            key="dashboard_season"
        )

    with c2:

        dashboard_competition = st.selectbox(
            "Competition",
            competition_list(data),
            key="dashboard_competition"
        )

    with c3:

        dashboard_venue = st.selectbox(
            "Venue",
            ["All", "Home", "Away"],
            key="dashboard_venue"
        )

    dashboard_sample = st.select_slider(
        "Recent sample",
        options=[5, 10, 15],
        value=10,
        key="dashboard_sample"
    )

    dashboard_df = filtered_team_matches(
        data,
        dashboard_team,
        dashboard_season,
        dashboard_competition,
        dashboard_venue,
        dashboard_sample
    )

    summary = team_summary(
        dashboard_df
    )

    quality_label, quality_message = sample_quality(
        len(dashboard_df)
    )

    if len(dashboard_df) < 5:
        st.warning(
            f"⚠️ {quality_label}: {quality_message}"
        )
    elif len(dashboard_df) < 10:
        st.info(
            f"ℹ️ {quality_label}: {quality_message}"
        )
    else:
        st.success(
            f"✓ {quality_label}: {quality_message}"
        )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Matches",
        summary["Matches"]
    )

    m2.metric(
        "Avg Shots",
        fmt(summary["Shots"])
    )

    m3.metric(
        "Avg SOT",
        fmt(summary["SOT"])
    )

    m4.metric(
        "Avg Corners",
        fmt(summary["Corners"])
    )

    m5.metric(
        "Avg Goals",
        fmt(summary["Goals"])
    )

    st.divider()

    st.subheader(
        "Market Hit-Rate Overview"
    )

    overview_rows = []

    for market_name in MARKETS:

        for direction_name in [
            "Over",
            "Under"
        ]:

            for line in [
                0.5,
                1.5,
                2.5,
                3.5,
                4.5
            ]:

                result = analyse_market(
                    dashboard_df,
                    market_name,
                    direction_name,
                    line
                )

                if result["sample_size"] > 0:

                    overview_rows.append(
                        {
                            "Market": market_name,
                            "Direction": direction_name,
                            "Line": line,
                            "Hits": result["hits"],
                            "Misses": result["misses"],
                            "Pushes": result["pushes"],
                            "Sample": result["sample_size"],
                            "Hit Rate": fmt_pct(
                                result["hit_rate"]
                            )
                        }
                    )

    st.dataframe(
        pd.DataFrame(overview_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Market Statistical Profile"
    )

    st.dataframe(
        multi_market_table(
            dashboard_df
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Home / Away Comparison"
    )

    split_rows = []

    for venue_label in [
        "Home",
        "Away"
    ]:

        split_df = filtered_team_matches(
            data,
            dashboard_team,
            dashboard_season,
            dashboard_competition,
            venue_label,
            dashboard_sample
        )

        split = team_summary(
            split_df
        )

        split_rows.append(
            {
                "Venue": venue_label,
                "Matches": split["Matches"],
                "Avg Shots": fmt(
                    split["Shots"]
                ),
                "Avg SOT": fmt(
                    split["SOT"]
                ),
                "Avg Corners": fmt(
                    split["Corners"]
                ),
                "Avg Goals": fmt(
                    split["Goals"]
                )
            }
        )

    st.dataframe(
        pd.DataFrame(split_rows),
        use_container_width=True,
        hide_index=True
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
        "opp_shots",
        "team_sot",
        "opp_sot",
        "team_corners",
        "opp_corners"
    ]

    st.dataframe(
        dashboard_df[
            [
                c
                for c in display_columns
                if c in dashboard_df.columns
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 — MARKET TESTER
# ============================================================

with tab2:

    st.subheader(
        "🎯 Market Tester"
    )

    st.caption(
        "Test one market against historical match-by-match data."
    )

    team = st.selectbox(
        "Team",
        team_list(data),
        key="market_team"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        season2 = st.selectbox(
            "Season",
            season_list(data),
            key="market_season"
        )

    with c2:

        competition2 = st.selectbox(
            "Competition",
            competition_list(data),
            key="market_competition"
        )

    with c3:

        venue2 = st.selectbox(
            "Venue",
            ["All", "Home", "Away"],
            key="market_venue"
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        market = st.selectbox(
            "Market",
            MARKETS,
            key="market_type"
        )

    with c5:

        direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="market_direction"
        )

    with c6:

        line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="market_line"
        )

    odds = st.number_input(
        "Decimal odds",
        min_value=1.01,
        max_value=100.0,
        value=1.30,
        step=0.01,
        key="market_odds"
    )

    sample = st.select_slider(
        "Recent sample",
        options=[5, 10, 15],
        value=10,
        key="market_sample"
    )

    market_df = filtered_team_matches(
        data,
        team,
        season2,
        competition2,
        venue2,
        sample
    )

    result = analyse_market(
        market_df,
        market,
        direction,
        line
    )

    rate = result["hit_rate"]
    breakeven = break_even_probability(
        odds
    )

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

    if rate is not None and breakeven is not None:

        difference = rate - breakeven

        st.metric(
            "Historical difference vs break-even",
            f"{difference * 100:+.1f} pp"
        )

        st.progress(
            min(
                max(rate, 0.0),
                1.0
            )
        )

    st.caption(
        "Historical hit rate is descriptive. "
        "It does not establish the probability of the next match."
    )

    st.divider()

    st.subheader(
        "Line Ladder"
    )

    ladder = market_line_ladder(
        market_df,
        market,
        direction
    )

    if not ladder.empty:

        st.dataframe(
            ladder,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No ladder data available."
        )

    st.divider()

    st.subheader(
        "Recent Trend"
    )

    trend = trend_table(
        market_df,
        market,
        direction,
        line
    )

    st.dataframe(
        trend,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Market History"
    )

    history = market_history(
        market_df,
        market,
        direction,
        line
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 — BOOKMAKER MONITOR
# ============================================================

with tab3:

    st.subheader(
        "📊 Bookmaker Market Monitor"
    )

    if "market_watchlist" not in st.session_state:

        st.session_state.market_watchlist = (
            load_watchlist()
        )

    st.caption(
        "Record bookmaker lines and odds manually. "
        "No bookmaker account is connected."
    )

    left, right = st.columns(2)

    with left:

        bookmaker = st.selectbox(
            "Bookmaker",
            [
                "SportyBet",
                "SunBet",
                "Virgin Bet"
            ],
            key="monitor_bookmaker"
        )

        match_name = st.text_input(
            "Match",
            placeholder="e.g. Chelsea vs Brentford",
            key="monitor_match"
        )

        match_date = st.date_input(
            "Match date",
            key="monitor_match_date"
        )

        capture_time = st.time_input(
            "Capture time",
            key="monitor_capture_time"
        )

        monitor_team = st.selectbox(
            "Team",
            team_list(data),
            key="monitor_team"
        )

        monitor_market = st.selectbox(
            "Market",
            MARKETS,
            key="monitor_market"
        )

    with right:

        monitor_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="monitor_direction"
        )

        monitor_line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="monitor_line"
        )

        monitor_odds = st.number_input(
            "Decimal odds",
            min_value=1.01,
            max_value=100.0,
            value=1.30,
            step=0.01,
            key="monitor_odds"
        )

        monitor_sample = st.selectbox(
            "Historical sample",
            [5, 10, 15],
            index=1,
            key="monitor_sample"
        )

        monitor_season = st.selectbox(
            "Season",
            season_list(data),
            key="monitor_season"
        )

        monitor_competition = st.selectbox(
            "Competition",
            competition_list(data),
            key="monitor_competition"
        )

        monitor_venue = st.selectbox(
            "Venue",
            ["All", "Home", "Away"],
            key="monitor_venue"
        )

    if st.button(
        "Analyse & Add Market",
        key="add_market"
    ):

        monitor_df = filtered_team_matches(
            data,
            monitor_team,
            monitor_season,
            monitor_competition,
            monitor_venue,
            monitor_sample
        )

        analysis = analyse_market(
            monitor_df,
            monitor_market,
            monitor_direction,
            monitor_line
        )

        historical_rate = analysis[
            "hit_rate"
        ]

        be = break_even_probability(
            monitor_odds
        )

        difference = historical_difference(
            historical_rate,
            monitor_odds
        )

        entry = {
            "Bookmaker": bookmaker,
            "Match": match_name,
            "Match Date": str(match_date),
            "Capture Time": str(capture_time),
            "Team": monitor_team,
            "Season": monitor_season,
            "Competition": monitor_competition,
            "Venue": monitor_venue,
            "Market": monitor_market,
            "Direction": monitor_direction,
            "Line": monitor_line,
            "Odds": monitor_odds,
            "Historical Hit Rate": fmt_pct(
                historical_rate
            ),
            "Break-even": fmt_pct(be),
            "Difference vs Break-even": (
                f"{difference * 100:+.1f} pp"
                if difference is not None
                else "—"
            ),
            "Status": "Watching",
            "Actual Result": ""
        }

        st.session_state.market_watchlist.append(
            entry
        )

        save_watchlist(
            st.session_state.market_watchlist
        )

        st.success(
            "Market added to watchlist."
        )

    st.divider()

    st.subheader(
        "Tracked Markets"
    )

    watchlist = st.session_state.market_watchlist

    if watchlist:

        watchlist_df = pd.DataFrame(
            watchlist
        ).fillna("")

        total = len(watchlist_df)

        watching = int(
            (watchlist_df["Status"] == "Watching").sum()
        )

        won = int(
            (watchlist_df["Status"] == "Won").sum()
        )

        lost = int(
            (watchlist_df["Status"] == "Lost").sum()
        )

        void = int(
            (watchlist_df["Status"] == "Void").sum()
        )

        settled = won + lost

        tracked_rate = (
            won / settled
            if settled > 0
            else None
        )

        a, b, c, d, e = st.columns(5)

        a.metric("Total", total)
        b.metric("Watching", watching)
        c.metric("Won", won)
        d.metric("Lost", lost)
        e.metric(
            "Tracked Hit Rate",
            fmt_pct(tracked_rate)
        )

        st.caption(
            f"Void markets: {void}"
        )

        st.dataframe(
            watchlist_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Edit / Settle"
        )

        for i, row in watchlist_df.iterrows():

            label = (
                f"{i + 1}. "
                f"{row.get('Match', '')} — "
                f"{row.get('Market', '')} "
                f"{row.get('Direction', '')} "
                f"{row.get('Line', '')}"
            )

            with st.expander(label):

                ec1, ec2 = st.columns(2)

                with ec1:

                    try:
                        current_odds = float(
                            row.get(
                                "Odds",
                                1.30
                            )
                        )
                    except Exception:
                        current_odds = 1.30

                    try:
                        current_line = float(
                            row.get(
                                "Line",
                                3.5
                            )
                        )
                    except Exception:
                        current_line = 3.5

                    new_odds = st.number_input(
                        "Odds",
                        min_value=1.01,
                        max_value=100.0,
                        value=current_odds,
                        step=0.01,
                        key=f"edit_odds_{i}"
                    )

                    new_line = st.number_input(
                        "Line",
                        min_value=0.0,
                        max_value=30.0,
                        value=current_line,
                        step=0.5,
                        key=f"edit_line_{i}"
                    )

                with ec2:

                    statuses = [
                        "Watching",
                        "Won",
                        "Lost",
                        "Void"
                    ]

                    current_status = row.get(
                        "Status",
                        "Watching"
                    )

                    if current_status not in statuses:
                        current_status = "Watching"

                    new_status = st.selectbox(
                        "Status",
                        statuses,
                        index=statuses.index(
                            current_status
                        ),
                        key=f"edit_status_{i}"
                    )

                    actual = st.number_input(
                        "Actual Result",
                        min_value=0.0,
                        max_value=100.0,
                        value=(
                            float(row["Actual Result"])
                            if str(
                                row.get(
                                    "Actual Result",
                                    ""
                                )
                            ).strip()
                            not in ["", "nan"]
                            else 0.0
                        ),
                        step=0.5,
                        key=f"actual_{i}"
                    )

                if st.button(
                    "Save Changes",
                    key=f"save_watch_{i}"
                ):

                    st.session_state.market_watchlist[i][
                        "Odds"
                    ] = new_odds

                    st.session_state.market_watchlist[i][
                        "Line"
                    ] = new_line

                    st.session_state.market_watchlist[i][
                        "Status"
                    ] = new_status

                    st.session_state.market_watchlist[i][
                        "Actual Result"
                    ] = actual

                    save_watchlist(
                        st.session_state.market_watchlist
                    )

                    st.success(
                        "Market updated."
                    )

                    st.rerun()

                if st.button(
                    "Delete Market",
                    key=f"delete_watch_{i}"
                ):

                    st.session_state.market_watchlist.pop(
                        i
                    )

                    save_watchlist(
                        st.session_state.market_watchlist
                    )

                    st.rerun()

        csv_watchlist = (
            pd.DataFrame(
                st.session_state.market_watchlist
            )
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Watchlist CSV",
            csv_watchlist,
            "market_watchlist.csv",
            "text/csv",
            key="download_watchlist"
        )

        if st.button(
            "Clear Monitor",
            key="clear_monitor"
        ):

            st.session_state.market_watchlist = []

            if os.path.exists(
                WATCHLIST_FILE
            ):
                os.remove(
                    WATCHLIST_FILE
                )

            st.rerun()

    else:

        st.info(
            "No markets added yet."
        )


# ============================================================
# TAB 4 — DATA FORMAT
# ============================================================

with tab4:

    st.subheader(
        "📥 Data Format"
    )

    st.write(
        "The real master database currently uses these "
        "13 core fields:"
    )

    st.code(
        ",".join(REQUIRED_COLUMNS)
    )

    st.success(
        f"Current real dataset loaded: {len(data):,} matches"
    )

    st.dataframe(
        data.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Data Coverage"
    )

    coverage_rows = []

    for column in REQUIRED_COLUMNS:

        non_null = data[column].notna().sum()

        coverage_rows.append(
            {
                "Column": column,
                "Rows": len(data),
                "Available": non_null,
                "Missing": len(data) - non_null,
                "Coverage": (
                    f"{non_null / len(data) * 100:.1f}%"
                    if len(data) > 0
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = (
        data
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download Current Dataset",
        csv_bytes,
        "football_master_2024_27_cleaned.csv",
        "text/csv",
        key="download_cleaned_data"
    )


# ============================================================
# TAB 5 — MATCH RESEARCH
# ============================================================

with tab5:

    st.subheader(
        "🔬 Match Research"
    )

    st.caption(
        "Research a specific fixture by comparing both teams "
        "through historical match-by-match evidence."
    )

    teams = team_list(data)

    r1, r2 = st.columns(2)

    with r1:

        home_team = st.selectbox(
            "Home Team",
            teams,
            key="research_home_team"
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
            key="research_away_team"
        )

    st.divider()

    st.subheader(
        "Match Context"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        research_season = st.selectbox(
            "Season",
            season_list(data),
            key="research_season"
        )

    with c2:

        research_competition = st.selectbox(
            "Competition",
            competition_list(data),
            key="research_competition"
        )

    with c3:

        research_sample = st.selectbox(
            "Recent matches",
            [5, 10, 15],
            index=1,
            key="research_sample"
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

    h2h = get_h2h(
        data,
        home_team,
        away_team
    )

    if research_season != "All":

        h2h = h2h[
            h2h["season"] == research_season
        ]

    if research_competition != "All":

        h2h = h2h[
            h2h["competition"] == research_competition
        ]

    # ========================================================
    # STEP 19–24 — CORE COMPARISON
    # ========================================================

    if len(home_matches) < 5:

        st.warning(
            f"{home_team}: only {len(home_matches)} "
            "home matches available."
        )

    if len(away_matches) < 5:

        st.warning(
            f"{away_team}: only {len(away_matches)} "
            "away matches available."
        )

    st.divider()

    st.subheader(
        "Home vs Away Team Comparison"
    )

    home_summary = team_summary(
        home_matches
    )

    away_summary = team_summary(
        away_matches
    )

    comparison_rows = []

    for label, key in [
        ("Matches", "Matches"),
        ("Avg Shots", "Shots"),
        ("Avg SOT", "SOT"),
        ("Avg Corners", "Corners"),
        ("Avg Goals", "Goals")
    ]:

        comparison_rows.append(
            {
                "Metric": label,
                "Home Team": (
                    home_summary[key]
                    if key == "Matches"
                    else fmt(home_summary[key])
                ),
                "Away Team": (
                    away_summary[key]
                    if key == "Matches"
                    else fmt(away_summary[key])
                )
            }
        )

    st.dataframe(
        pd.DataFrame(comparison_rows),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 25 — ATTACK VS DEFENCE
    # ========================================================

    st.divider()

    st.subheader(
        "Attack vs Defence"
    )

    st.dataframe(
        attack_defence_table(
            home_matches,
            away_matches
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 26 — STATISTICAL PROFILE
    # ========================================================

    st.divider()

    st.subheader(
        "Statistical Consistency Profile"
    )

    selected_profile_market = st.selectbox(
        "Profile Market",
        MARKETS,
        key="profile_market"
    )

    profile_df = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Venue": "Home",
                **{
                    k: fmt(v)
                    for k, v in consistency_profile(
                        home_matches,
                        selected_profile_market
                    ).items()
                }
            },
            {
                "Team": away_team,
                "Venue": "Away",
                **{
                    k: fmt(v)
                    for k, v in consistency_profile(
                        away_matches,
                        selected_profile_market
                    ).items()
                }
            }
        ]
    )

    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 27 — MARKET TEST
    # ========================================================

    st.divider()

    st.subheader(
        "Fixture Market Comparison"
    )

    mc1, mc2, mc3 = st.columns(3)

    with mc1:

        research_market = st.selectbox(
            "Market",
            MARKETS,
            key="research_market"
        )

    with mc2:

        research_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="research_direction"
        )

    with mc3:

        research_line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="research_line"
        )

    home_market = analyse_market(
        home_matches,
        research_market,
        research_direction,
        research_line
    )

    away_market = analyse_market(
        away_matches,
        research_market,
        research_direction,
        research_line
    )

    market_compare = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Venue": "Home",
                "Sample": home_market["sample_size"],
                "Hits": home_market["hits"],
                "Misses": home_market["misses"],
                "Pushes": home_market["pushes"],
                "Hit Rate": fmt_pct(
                    home_market["hit_rate"]
                )
            },
            {
                "Team": away_team,
                "Venue": "Away",
                "Sample": away_market["sample_size"],
                "Hits": away_market["hits"],
                "Misses": away_market["misses"],
                "Pushes": away_market["pushes"],
                "Hit Rate": fmt_pct(
                    away_market["hit_rate"]
                )
            }
        ]
    )

    st.dataframe(
        market_compare,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 28 — TREND SPLIT
    # ========================================================

    st.subheader(
        "Recent Trend Comparison"
    )

    home_trend = trend_table(
        home_matches,
        research_market,
        research_direction,
        research_line
    )

    away_trend = trend_table(
        away_matches,
        research_market,
        research_direction,
        research_line
    )

    tc1, tc2 = st.columns(2)

    with tc1:

        st.caption(
            f"{home_team} — Home"
        )

        st.dataframe(
            home_trend,
            use_container_width=True,
            hide_index=True
        )

    with tc2:

        st.caption(
            f"{away_team} — Away"
        )

        st.dataframe(
            away_trend,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 29 — OPPONENT CONTEXT
    # ========================================================

    st.divider()

    st.subheader(
        "Opponent Context"
    )

    market_column = MARKET_COLUMN_MAP[
        research_market
    ]

    opponent_column = {
        "Shots": "opp_shots",
        "Shots on Target": "opp_sot",
        "Corners": "opp_corners",
        "Goals": "opp_goals"
    }[research_market]

    home_opp_values = pd.to_numeric(
        home_matches[opponent_column],
        errors="coerce"
    ).dropna()

    away_opp_values = pd.to_numeric(
        away_matches[opponent_column],
        errors="coerce"
    ).dropna()

    if research_direction == "Over":

        home_opp_context = (
            home_opp_values < research_line
        )

        away_opp_context = (
            away_opp_values < research_line
        )

    else:

        home_opp_context = (
            home_opp_values > research_line
        )

        away_opp_context = (
            away_opp_values > research_line
        )

    home_context_rate = (
        home_opp_context.mean()
        if len(home_opp_context)
        else None
    )

    away_context_rate = (
        away_opp_context.mean()
        if len(away_opp_context)
        else None
    )

    opponent_df = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Team Market Hit Rate": fmt_pct(
                    home_market["hit_rate"]
                ),
                "Team Sample": home_market[
                    "sample_size"
                ],
                "Opponent Context Rate": fmt_pct(
                    away_context_rate
                ),
                "Opponent Sample": len(
                    away_opp_values
                )
            },
            {
                "Team": away_team,
                "Team Market Hit Rate": fmt_pct(
                    away_market["hit_rate"]
                ),
                "Team Sample": away_market[
                    "sample_size"
                ],
                "Opponent Context Rate": fmt_pct(
                    home_context_rate
                ),
                "Opponent Sample": len(
                    home_opp_values
                )
            }
        ]
    )

    st.dataframe(
        opponent_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 30 — HOME / AWAY FORM
    # ========================================================

    st.divider()

    st.subheader(
        "Recent Home / Away Form"
    )

    form_columns = [
        "date",
        "home_team",
        "away_team",
        "venue",
        "team_goals",
        "opp_goals",
        "team_shots",
        "opp_shots",
        "team_sot",
        "opp_sot",
        "team_corners",
        "opp_corners"
    ]

    fc1, fc2 = st.columns(2)

    with fc1:

        st.caption(
            f"{home_team} — Home"
        )

        st.dataframe(
            home_matches[
                [
                    c
                    for c in form_columns
                    if c in home_matches.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with fc2:

        st.caption(
            f"{away_team} — Away"
        )

        st.dataframe(
            away_matches[
                [
                    c
                    for c in form_columns
                    if c in away_matches.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 31 — OVERALL FORM
    # ========================================================

    st.divider()

    st.subheader(
        "Overall Recent Form"
    )

    oc1, oc2 = st.columns(2)

    with oc1:

        st.caption(home_team)

        st.dataframe(
            home_all[
                [
                    c
                    for c in form_columns
                    if c in home_all.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with oc2:

        st.caption(away_team)

        st.dataframe(
            away_all[
                [
                    c
                    for c in form_columns
                    if c in away_all.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 32 — H2H
    # ========================================================

    st.divider()

    st.subheader(
        "Head-to-Head"
    )

    if not h2h.empty:

        h2h_columns = [
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

        st.dataframe(
            h2h[
                [
                    c
                    for c in h2h_columns
                    if c in h2h.columns
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"{len(h2h)} H2H meeting(s) found."
        )

    else:

        st.info(
            "No H2H matches found for the selected filters."
        )

    # ========================================================
    # STEP 33 — H2H MARKET TEST
    # ========================================================

    if not h2h.empty:

        st.subheader(
            "H2H Market Context"
        )

        h2h_team_perspective = []

        for _, row in h2h.iterrows():

            if row["home_team"] == home_team:

                if research_market == "Shots":
                    value = row["home_shots"]
                elif research_market == "Shots on Target":
                    value = row["home_sot"]
                elif research_market == "Corners":
                    value = row["home_corners"]
                else:
                    value = row["home_goals"]

            else:

                if research_market == "Shots":
                    value = row["away_shots"]
                elif research_market == "Shots on Target":
                    value = row["away_sot"]
                elif research_market == "Corners":
                    value = row["away_corners"]
                else:
                    value = row["away_goals"]

            h2h_team_perspective.append(
                value
            )

        h2h_values = pd.Series(
            h2h_team_perspective
        )

        h2h_values = pd.to_numeric(
            h2h_values,
            errors="coerce"
        ).dropna()

        if not h2h_values.empty:

            if research_direction == "Over":

                h2h_hits = (
                    h2h_values > research_line
                ).sum()

            else:

                h2h_hits = (
                    h2h_values < research_line
                ).sum()

            st.metric(
                "H2H historical hit rate",
                f"{h2h_hits / len(h2h_values) * 100:.1f}%"
            )

            st.caption(
                "H2H is supplementary historical context; "
                "it should not be treated as a forecast."
            )

    # ========================================================
    # STEP 34 — CANDIDATE MARKET MATRIX
    # ========================================================

    st.divider()

    st.subheader(
        "Candidate Market Matrix"
    )

    st.caption(
        "This table exposes historical market behaviour "
        "without assigning a 'safe' label."
    )

    candidate_df = candidate_market_matrix(
        home_matches,
        away_matches
    )

    if not candidate_df.empty:

        candidate_filter_market = st.multiselect(
            "Show markets",
            MARKETS,
            default=MARKETS,
            key="candidate_market_filter"
        )

        candidate_display = candidate_df[
            candidate_df["Market"].isin(
                candidate_filter_market
            )
        ]

        st.dataframe(
            candidate_display,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 35 — RESEARCH QUALITY
    # ========================================================

    st.divider()

    st.subheader(
        "Research Coverage"
    )

    quality_label, quality_notes = research_quality(
        len(home_matches),
        len(away_matches),
        len(h2h)
    )

    st.info(
        f"Research coverage: **{quality_label}**"
    )

    if quality_notes:

        for note in quality_notes:
            st.write(
                f"• {note}"
            )

    # ========================================================
    # STEP 36 — RESEARCH CHECKLIST
    # ========================================================

    st.divider()

    st.subheader(
        "Research Checklist"
    )

    checklist_items = [
        "Historical home sample checked",
        "Historical away sample checked",
        "Recent form checked",
        "Attack production checked",
        "Defensive concession checked",
        "Opponent context checked",
        "Market line tested",
        "Recent trend checked",
        "H2H checked",
        "Bookmaker line/odds manually verified",
        "Lineups/team news checked externally",
        "Referee/context information checked externally"
    ]

    checked = []

    for item in checklist_items:

        checked.append(
            st.checkbox(
                item,
                key=f"check_{item}"
            )
        )

    completed = sum(checked)

    st.progress(
        completed / len(checklist_items)
    )

    st.caption(
        f"{completed}/{len(checklist_items)} research items completed."
    )

    # ========================================================
    # STEP 37 — BOOKMAKER ENTRY FROM MATCH RESEARCH
    # ========================================================

    st.divider()

    st.subheader(
        "Add Tested Market to Bookmaker Monitor"
    )

    save_bookmaker = st.selectbox(
        "Bookmaker",
        [
            "SportyBet",
            "SunBet",
            "Virgin Bet"
        ],
        key="research_bookmaker"
    )

    save_odds = st.number_input(
        "Bookmaker odds",
        min_value=1.01,
        max_value=100.0,
        value=1.30,
        step=0.01,
        key="research_odds"
    )

    if st.button(
        "Add This Market to Monitor",
        key="research_add_monitor"
    ):

        rate_for_monitor = home_market[
            "hit_rate"
        ]

        be_for_monitor = break_even_probability(
            save_odds
        )

        diff_for_monitor = historical_difference(
            rate_for_monitor,
            save_odds
        )

        entry = {
            "Bookmaker": save_bookmaker,
            "Match": f"{home_team} vs {away_team}",
            "Match Date": "",
            "Capture Time": "",
            "Team": home_team,
            "Season": research_season,
            "Competition": research_competition,
            "Venue": "Home",
            "Market": research_market,
            "Direction": research_direction,
            "Line": research_line,
            "Odds": save_odds,
            "Historical Hit Rate": fmt_pct(
                rate_for_monitor
            ),
            "Break-even": fmt_pct(
                be_for_monitor
            ),
            "Difference vs Break-even": (
                f"{diff_for_monitor * 100:+.1f} pp"
                if diff_for_monitor is not None
                else "—"
            ),
            "Status": "Watching",
            "Actual Result": ""
        }

        if "market_watchlist" not in st.session_state:

            st.session_state.market_watchlist = (
                load_watchlist()
            )

        st.session_state.market_watchlist.append(
            entry
        )

        save_watchlist(
            st.session_state.market_watchlist
        )

        st.success(
            "Market added to Bookmaker Monitor."
        )

    # ========================================================
    # STEP 38 — SAVE FIXTURE RESEARCH
    # ========================================================

    st.divider()

    st.subheader(
        "Save Match Research"
    )

    notes = st.text_area(
        "Research notes",
        placeholder=(
            "Record lineup information, injuries, tactical "
            "observations, referee notes, bookmaker line "
            "movement, or other observations."
        ),
        height=180,
        key="fixture_notes"
    )

    if st.button(
        "Save Fixture Research",
        key="save_fixture_research"
    ):

        record = {
            "Saved At": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Home Team": home_team,
            "Away Team": away_team,
            "Season": research_season,
            "Competition": research_competition,
            "Sample": research_sample,
            "Market": research_market,
            "Direction": research_direction,
            "Line": research_line,
            "Home Hit Rate": fmt_pct(
                home_market["hit_rate"]
            ),
            "Away Hit Rate": fmt_pct(
                away_market["hit_rate"]
            ),
            "H2H Matches": len(h2h),
            "Research Coverage": quality_label,
            "Notes": notes
        }

        save_research_record(
            record
        )

        st.success(
            "Fixture research saved."
        )

    # ========================================================
    # STEP 39 — MATCH RESEARCH SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "Match Research Summary"
    )

    summary_rows = [
        {
            "Area": "Home Attack — Shots",
            "Team": home_team,
            "Value": fmt(
                average_value(
                    home_matches,
                    "team_shots"
                )
            )
        },
        {
            "Area": "Home Attack — SOT",
            "Team": home_team,
            "Value": fmt(
                average_value(
                    home_matches,
                    "team_sot"
                )
            )
        },
        {
            "Area": "Away Defence — SOT Conceded",
            "Team": away_team,
            "Value": fmt(
                average_value(
                    away_matches,
                    "opp_sot"
                )
            )
        },
        {
            "Area": "Away Attack — Shots",
            "Team": away_team,
            "Value": fmt(
                average_value(
                    away_matches,
                    "team_shots"
                )
            )
        },
        {
            "Area": "Away Attack — SOT",
            "Team": away_team,
            "Value": fmt(
                average_value(
                    away_matches,
                    "team_sot"
                )
            )
        },
        {
            "Area": "Home Defence — SOT Conceded",
            "Team": home_team,
            "Value": fmt(
                average_value(
                    home_matches,
                    "opp_sot"
                )
            )
        }
    ]

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The summary is descriptive historical evidence. "
        "It does not predict the result of the fixture."
    )

    # ========================================================
    # STEP 40 — WORKFLOW STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "Research Workflow Status"
    )

    workflow_rows = [
        {
            "Research Area": "Home historical sample",
            "Status": (
                "Available"
                if len(home_matches) >= 5
                else "Limited"
            )
        },
        {
            "Research Area": "Away historical sample",
            "Status": (
                "Available"
                if len(away_matches) >= 5
                else "Limited"
            )
        },
        {
            "Research Area": "Attack statistics",
            "Status": "Available"
        },
        {
            "Research Area": "Defensive concession",
            "Status": "Available"
        },
        {
            "Research Area": "Market testing",
            "Status": "Available"
        },
        {
            "Research Area": "Recent trend",
            "Status": "Available"
        },
        {
            "Research Area": "Opponent context",
            "Status": "Available"
        },
        {
            "Research Area": "H2H",
            "Status": (
                "Available"
                if len(h2h) > 0
                else "No data"
            )
        },
        {
            "Research Area": "Bookmaker monitoring",
            "Status": "Available"
        },
        {
            "Research Area": "Research notes",
            "Status": "Available"
        },
        {
            "Research Area": "Lineups / injuries",
            "Status": "External information required"
        },
        {
            "Research Area": "Referee information",
            "Status": "External information required"
        },
        {
            "Research Area": "Possession / xG",
            "Status": "Not in current CSV"
        },
        {
            "Research Area": "Player markets",
            "Status": "Not in current CSV"
        }
    ]

    st.dataframe(
        pd.DataFrame(workflow_rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 6 — RESEARCH HISTORY
# ============================================================

with tab6:

    st.subheader(
        "📚 Research History"
    )

    st.caption(
        "Previously saved fixture research records."
    )

    research_history = load_research_history()

    if research_history.empty:

        st.info(
            "No saved fixture research yet."
        )

    else:

        st.dataframe(
            research_history,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        csv_history = (
            research_history
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Research History",
            csv_history,
            "match_research_history.csv",
            "text/csv",
            key="download_research_history"
        )

        if st.button(
            "Clear Research History",
            key="clear_research_history"
        ):

            if os.path.exists(
                RESEARCH_FILE
            ):

                os.remove(
                    RESEARCH_FILE
                )

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Football Betting Research Tool V1 — Steps 1–40 integrated."
)

st.caption(
    "Research assistant, not a prediction engine. "
    "Historical statistics describe previous matches and "
    "do not establish the probability of the next match."
)

st.caption(
    "Current database: football_master_2024_27_v1.csv"
)
