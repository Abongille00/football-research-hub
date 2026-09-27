import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, date, time

# ============================================================
# FOOTBALL BETTING RESEARCH TOOL
# STEPS 1–100
#
# Steps 1–90:
#   Core dataset loading
#   Team research
#   Market testing
#   Bookmaker monitor
#   Match research
#   Historical/context analysis
#
# Steps 91–100:
#   91. Data Quality Dashboard
#   92. Team Data Coverage
#   93. Recent Form Engine
#   94. Market Consistency Analysis
#   95. Opponent Strength / Context
#   96. Line Sensitivity
#   97. Market Stability Panel
#   98. Match Research Scorecard
#   99. Research Audit Trail
#   100. Research Performance Dashboard
#
# IMPORTANT:
# This application is a research/audit tool.
# Historical statistics are descriptive and are not forecasts.
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Betting Research V1",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Betting Research Tool — V1")
st.caption(
    "Research assistant, not a prediction engine. "
    "Use match-by-match data to test markets, examine context, "
    "and audit research decisions."
)


# ============================================================
# CONFIGURATION
# ============================================================

MASTER_FILE = "football_master_2024_27_v1.csv"

WATCHLIST_FILE = "market_watchlist.csv"
AUDIT_FILE = "research_audit_log.csv"

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

BASE_MARKET_COLUMNS = [
    "Shots",
    "Shots on Target",
    "Corners",
    "Goals"
]


# ============================================================
# STEP 1–10
# DATA CLEANING / NORMALISATION
# ============================================================

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

    return df


# ============================================================
# TEAM PERSPECTIVE ENGINE
# ============================================================

def team_matches(df, team):
    """Convert raw match data into selected team's perspective."""

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

    return out


def filtered_team_matches(
    data,
    team,
    season="All",
    competition="All",
    venue="All",
    sample=15
):
    """Consistent team filtering engine."""

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

    return m.head(sample)


# ============================================================
# MARKET ENGINE
# ============================================================

def analyse_market(
    df,
    market,
    direction,
    line
):
    """Analyse historical market performance."""

    empty_result = {
        "hit_rate": None,
        "hits": 0,
        "misses": 0,
        "sample_size": 0
    }

    if df is None or df.empty:
        return empty_result

    column = MARKET_COLUMN_MAP.get(
        market
    )

    if (
        column is None
        or column not in df.columns
    ):
        return empty_result

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return empty_result

    if direction == "Over":
        hits = int(
            (values > line).sum()
        )
    else:
        hits = int(
            (values < line).sum()
        )

    sample_size = int(
        len(values)
    )

    misses = sample_size - hits

    return {
        "hit_rate": (
            hits / sample_size
        ),
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
    """Create transparent match-by-match history."""

    if df is None or df.empty:
        return pd.DataFrame()

    column = MARKET_COLUMN_MAP.get(
        market
    )

    if (
        column is None
        or column not in df.columns
    ):
        return pd.DataFrame()

    result_columns = [
        "date",
        "home_team",
        "away_team",
        "venue",
        column
    ]

    available = [
        c for c in result_columns
        if c in df.columns
    ]

    result = df[
        available
    ].copy()

    if column in result.columns:

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


def hit_rate(
    series,
    line,
    over=True
):
    """Calculate simple historical hit rate."""

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(s) == 0:
        return None

    if over:
        hits = (
            s > line
        ).sum()
    else:
        hits = (
            s < line
        ).sum()

    return hits / len(s)


# ============================================================
# GENERAL HELPERS
# ============================================================

def fmt_pct(value):

    if (
        value is None
        or pd.isna(value)
    ):
        return "—"

    return f"{value * 100:.1f}%"


def fmt_num(value):

    if (
        value is None
        or pd.isna(value)
    ):
        return "—"

    return f"{value:.2f}"


def sample_quality(sample_size):

    if sample_size == 0:
        return (
            "No data",
            "No matches are available."
        )

    if sample_size < 5:
        return (
            "Very small sample",
            "Fewer than 5 matches are available."
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


def average_value(
    df,
    column
):

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


def median_value(
    df,
    column
):

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

    return values.median()


def std_value(
    df,
    column
):

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

    return values.std()


def team_summary(df):

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

            metrics[
                f"{label} Avg"
            ] = (
                values.mean()
                if len(values)
                else None
            )

    return metrics


def match_team_summary(matches):

    if (
        matches is None
        or matches.empty
    ):
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


def team_list(data):

    return sorted(
        set(
            data["home_team"]
            .dropna()
        )
        |
        set(
            data["away_team"]
            .dropna()
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


def get_match_history(
    data,
    home_team,
    away_team
):

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
            "Matches": home_result[
                "sample_size"
            ],
            "Hits": home_result[
                "hits"
            ],
            "Misses": home_result[
                "misses"
            ],
            "Hit Rate": (
                fmt_pct(
                    home_result[
                        "hit_rate"
                    ]
                )
            )
        },
        {
            "Team": "Away",
            "Matches": away_result[
                "sample_size"
            ],
            "Hits": away_result[
                "hits"
            ],
            "Misses": away_result[
                "misses"
            ],
            "Hit Rate": (
                fmt_pct(
                    away_result[
                        "hit_rate"
                    ]
                )
            )
        }
    ])


# ============================================================
# DATA LOAD
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
            f"Uploaded dataset: {len(data):,} matches"
        )

    except Exception as exc:

        st.error(
            f"Could not read uploaded CSV: {exc}"
        )

        st.stop()

else:

    if os.path.exists(
        MASTER_FILE
    ):

        try:

            data = clean_data(
                pd.read_csv(
                    MASTER_FILE
                )
            )

            st.sidebar.success(
                f"Primary dataset: "
                f"{len(data):,} matches"
            )

        except Exception as exc:

            st.error(
                f"Could not read {MASTER_FILE}: {exc}"
            )

            st.stop()

    else:

        st.error(
            f"{MASTER_FILE} was not found. "
            "Place the real CSV in the same folder "
            "as app.py."
        )

        st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

missing = [
    c for c in REQUIRED_COLUMNS
    if c not in data.columns
]

if missing:

    st.error(
        "Your CSV is missing these columns: "
        + ", ".join(missing)
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "market_watchlist" not in st.session_state:

    if os.path.exists(
        WATCHLIST_FILE
    ):

        try:

            loaded = pd.read_csv(
                WATCHLIST_FILE
            )

            if "Status" not in loaded.columns:
                loaded["Status"] = "Watching"

            if "Actual Result" not in loaded.columns:
                loaded["Actual Result"] = ""

            st.session_state.market_watchlist = (
                loaded
                .fillna("")
                .to_dict("records")
            )

        except Exception:

            st.session_state.market_watchlist = []

    else:

        st.session_state.market_watchlist = []


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🔎 Team Research",
    "🎯 Market Tester",
    "📊 Bookmaker Monitor",
    "📥 Data Format",
    "🔬 Match Research",
    "🧪 Data Quality",
    "📈 Coverage",
    "📋 Recent Form",
    "📊 Consistency",
    "🎚️ Line Sensitivity",
    "🧭 Stability",
    "📝 Scorecard",
    "🗂️ Audit Trail",
    "🏁 Performance"
])

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
    tab9,
    tab10,
    tab11,
    tab12,
    tab13,
    tab14
) = tabs


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

    dashboard_season = st.selectbox(
        "Season",
        season_list(data),
        key="dashboard_season"
    )

    dashboard_competition = st.selectbox(
        "Competition",
        competition_list(data),
        key="dashboard_competition"
    )

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

    dashboard_metrics = team_summary(
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

    d1, d2, d3, d4, d5 = st.columns(5)

    d1.metric(
        "Matches",
        dashboard_metrics.get(
            "Matches",
            0
        )
    )

    d2.metric(
        "Avg Shots",
        fmt_num(
            dashboard_metrics.get(
                "Shots Avg"
            )
        )
    )

    d3.metric(
        "Avg SOT",
        fmt_num(
            dashboard_metrics.get(
                "Shots on Target Avg"
            )
        )
    )

    d4.metric(
        "Avg Corners",
        fmt_num(
            dashboard_metrics.get(
                "Corners Avg"
            )
        )
    )

    d5.metric(
        "Avg Goals",
        fmt_num(
            dashboard_metrics.get(
                "Goals Avg"
            )
        )
    )

    st.divider()

    st.subheader(
        "Market Hit-Rate Overview"
    )

    overview_rows = []

    for market_name in BASE_MARKET_COLUMNS:

        for direction_name in [
            "Over",
            "Under"
        ]:

            for test_line in [
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
                    test_line
                )

                if result[
                    "sample_size"
                ] > 0:

                    overview_rows.append({
                        "Market": market_name,
                        "Direction": direction_name,
                        "Line": test_line,
                        "Hits": result[
                            "hits"
                        ],
                        "Sample": result[
                            "sample_size"
                        ],
                        "Hit Rate": fmt_pct(
                            result[
                                "hit_rate"
                            ]
                        )
                    })

    if overview_rows:

        st.dataframe(
            pd.DataFrame(
                overview_rows
            ),
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
        "team_shots",
        "team_sot",
        "team_corners",
        "team_goals"
    ]

    available_columns = [
        c for c in display_columns
        if c in dashboard_df.columns
    ]

    st.dataframe(
        dashboard_df[
            available_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 — MARKET TESTER
# ============================================================

with tab2:

    st.subheader(
        "🎯 Test a Market"
    )

    team = st.selectbox(
        "Team to test",
        team_list(data),
        key="market_team"
    )

    season2 = st.selectbox(
        "Season",
        season_list(data),
        key="market_season"
    )

    competition2 = st.selectbox(
        "Competition",
        competition_list(data),
        key="market_competition"
    )

    venue2 = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="market_venue"
    )

    market = st.selectbox(
        "Market",
        BASE_MARKET_COLUMNS,
        key="market_type"
    )

    direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="market_direction"
    )

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

    n2 = st.select_slider(
        "Recent sample",
        options=[5, 10, 15],
        value=10,
        key="market_sample"
    )

    m = filtered_team_matches(
        data,
        team,
        season2,
        competition2,
        venue2,
        n2
    )

    analysis = analyse_market(
        m,
        market,
        direction,
        line
    )

    rate = analysis["hit_rate"]
    hits = analysis["hits"]
    misses = analysis["misses"]
    sample_size = analysis["sample_size"]

    breakeven = 1 / odds

    a, b, c, d = st.columns(4)

    a.metric(
        "Historical hit rate",
        fmt_pct(rate)
    )

    b.metric(
        "Break-even probability",
        f"{breakeven * 100:.1f}%"
    )

    c.metric(
        "Sample size",
        sample_size
    )

    d.metric(
        "Historical vs break-even",
        (
            "Above"
            if rate is not None
            and rate >= breakeven
            else (
                "Below"
                if rate is not None
                else "—"
            )
        )
    )

    if rate is not None:

        st.progress(
            min(
                max(rate, 0),
                1
            )
        )

        st.metric(
            "Historical difference vs break-even",
            f"{(rate - breakeven) * 100:+.1f} pp"
        )

    st.caption(
        "Historical hit rate is descriptive only. "
        "It does not establish the probability of the next match."
    )

    st.divider()

    st.subheader(
        "Market History"
    )

    history_df = market_history(
        m,
        market,
        direction,
        line
    )

    if not history_df.empty:

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Hits",
        hits
    )

    h2.metric(
        "Misses",
        misses
    )

    h3.metric(
        "Matches analysed",
        sample_size
    )


# ============================================================
# TAB 3 — BOOKMAKER MONITOR
# ============================================================

with tab3:

    st.subheader(
        "📊 Bookmaker Market Monitor"
    )

    col1, col2 = st.columns(2)

    with col1:

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
            "",
            placeholder="Chelsea vs Brentford",
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
            BASE_MARKET_COLUMNS,
            key="monitor_market"
        )

    with col2:

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

        rate = analysis["hit_rate"]
        breakeven = 1 / monitor_odds

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
            "Historical Hit Rate": (
                fmt_pct(rate)
            ),
            "Break-even": (
                f"{breakeven * 100:.1f}%"
            ),
            "Edge vs Break-even": (
                f"{(rate - breakeven) * 100:+.1f} pp"
                if rate is not None
                else "—"
            ),
            "Historical vs Break-even": (
                "Above"
                if rate is not None
                and rate >= breakeven
                else (
                    "Below"
                    if rate is not None
                    else "—"
                )
            ),
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
            "Market added to watchlist."
        )

    st.divider()

    if st.session_state.market_watchlist:

        # Automatic settlement
        for item in st.session_state.market_watchlist:

            actual_text = str(
                item.get(
                    "Actual Result",
                    ""
                )
            ).strip()

            if (
                actual_text != ""
                and item.get(
                    "Status"
                ) == "Watching"
            ):

                try:

                    actual = float(
                        actual_text
                    )

                    market_line = float(
                        item["Line"]
                    )

                    if item[
                        "Direction"
                    ] == "Over":

                        won_result = (
                            actual
                            > market_line
                        )

                    else:

                        won_result = (
                            actual
                            < market_line
                        )

                    item["Status"] = (
                        "Won"
                        if won_result
                        else "Lost"
                    )

                except (
                    ValueError,
                    TypeError,
                    KeyError
                ):
                    pass

        pd.DataFrame(
            st.session_state.market_watchlist
        ).to_csv(
            WATCHLIST_FILE,
            index=False
        )

        watchlist_df = pd.DataFrame(
            st.session_state.market_watchlist
        ).fillna("")

        total = len(
            watchlist_df
        )

        watching = (
            watchlist_df[
                "Status"
            ].eq("Watching").sum()
        )

        won = (
            watchlist_df[
                "Status"
            ].eq("Won").sum()
        )

        lost = (
            watchlist_df[
                "Status"
            ].eq("Lost").sum()
        )

        void = (
            watchlist_df[
                "Status"
            ].eq("Void").sum()
        )

        settled = won + lost

        tracked_hit_rate = (
            won / settled
            if settled > 0
            else None
        )

        s1, s2, s3, s4, s5 = st.columns(5)

        s1.metric(
            "Total",
            total
        )

        s2.metric(
            "Watching",
            watching
        )

        s3.metric(
            "Won",
            won
        )

        s4.metric(
            "Lost",
            lost
        )

        s5.metric(
            "Tracked Hit Rate",
            fmt_pct(
                tracked_hit_rate
            )
        )

        st.caption(
            f"Void markets: {void}"
        )

        st.dataframe(
            watchlist_df,
            use_container_width=True,
            hide_index=True
        )

        csv_watchlist = (
            watchlist_df
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
        "📥 CSV Format"
    )

    st.write(
        "The primary dataset is "
        f"`{MASTER_FILE}`."
    )

    st.write(
        "Required columns:"
    )

    st.code(
        ",".join(
            REQUIRED_COLUMNS
        )
    )

    st.dataframe(
        data.head(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5 — MATCH RESEARCH
# ============================================================

with tab5:

    st.subheader(
        "🔬 Match Research"
    )

    st.caption(
        "Compare two teams using historical "
        "home/away and match-by-match data."
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
            x for x in teams
            if x != home_team
        ]

        away_team = st.selectbox(
            "Away Team",
            away_options,
            key="research_away_team"
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

    st.divider()

    st.subheader(
        "Home vs Away Comparison"
    )

    hs = match_team_summary(
        home_matches
    )

    aws = match_team_summary(
        away_matches
    )

    comparison_rows = []

    for metric, key in [
        ("Matches", "Matches"),
        ("Avg Shots", "Shots"),
        ("Avg SOT", "SOT"),
        ("Avg Corners", "Corners"),
        ("Avg Goals", "Goals")
    ]:

        comparison_rows.append({
            "Metric": metric,
            "Home Team": (
                hs[key]
                if key == "Matches"
                else fmt_num(hs[key])
            ),
            "Away Team": (
                aws[key]
                if key == "Matches"
                else fmt_num(aws[key])
            )
        })

    st.dataframe(
        pd.DataFrame(
            comparison_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Attack vs Defence"
    )

    attack_defence_rows = []

    pairs = [
        (
            "Shots",
            "team_shots",
            "opp_shots"
        ),
        (
            "Shots on Target",
            "team_sot",
            "opp_sot"
        ),
        (
            "Corners",
            "team_corners",
            "opp_corners"
        ),
        (
            "Goals",
            "team_goals",
            "opp_goals"
        )
    ]

    for label, team_col, opp_col in pairs:

        attack_defence_rows.append({
            "Metric": label,
            "Home Attack": fmt_num(
                average_value(
                    home_matches,
                    team_col
                )
            ),
            "Away Defence Conceded": fmt_num(
                average_value(
                    away_matches,
                    opp_col
                )
            ),
            "Away Attack": fmt_num(
                average_value(
                    away_matches,
                    team_col
                )
            ),
            "Home Defence Conceded": fmt_num(
                average_value(
                    home_matches,
                    opp_col
                )
            )
        })

    st.dataframe(
        pd.DataFrame(
            attack_defence_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Recent Home / Away Form"
    )

    left, right = st.columns(2)

    with left:

        st.caption(
            f"{home_team} — Home"
        )

        st.dataframe(
            home_matches[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "team_goals",
                    "opp_goals",
                    "team_shots",
                    "team_sot",
                    "team_corners"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.caption(
            f"{away_team} — Away"
        )

        st.dataframe(
            away_matches[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "team_goals",
                    "opp_goals",
                    "team_shots",
                    "team_sot",
                    "team_corners"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Market Comparison"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        research_market = st.selectbox(
            "Market",
            BASE_MARKET_COLUMNS,
            key="research_market"
        )

    with m2:

        research_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="research_direction"
        )

    with m3:

        research_line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="research_line"
        )

    st.dataframe(
        build_market_comparison(
            home_matches,
            away_matches,
            research_market,
            research_direction,
            research_line
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Head-to-Head"
    )

    h2h = get_match_history(
        data,
        home_team,
        away_team
    )

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

    if not h2h.empty:

        st.dataframe(
            h2h.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No previous meetings found."
        )

    st.divider()

    st.subheader(
        "Research Notes"
    )

    st.text_area(
        "Notes",
        placeholder=(
            "Lineups, injuries, tactical observations, "
            "team news, referee information, etc."
        ),
        height=160,
        key="match_research_notes"
    )


# ============================================================
# STEP 91
# DATA QUALITY DASHBOARD
# ============================================================

with tab6:

    st.subheader(
        "🧪 Step 91 — Data Quality Dashboard"
    )

    st.caption(
        "Checks the structural quality of the active dataset "
        "before using it for research."
    )

    total_rows = len(data)

    duplicate_count = int(
        data.duplicated().sum()
    )

    missing_by_column = (
        data.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    invalid_dates = int(
        data["date"].isna().sum()
    )

    blank_home = int(
        data["home_team"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    blank_away = int(
        data["away_team"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Total Matches",
        f"{total_rows:,}"
    )

    q2.metric(
        "Duplicate Rows",
        f"{duplicate_count:,}"
    )

    q3.metric(
        "Invalid Dates",
        f"{invalid_dates:,}"
    )

    q4.metric(
        "Missing Cells",
        f"{int(data.isna().sum().sum()):,}"
    )

    st.divider()

    st.subheader(
        "Column Completeness"
    )

    quality_df = pd.DataFrame({
        "Column": data.columns,
        "Missing": [
            int(data[c].isna().sum())
            for c in data.columns
        ],
        "Missing %": [
            f"{data[c].isna().mean() * 100:.2f}%"
            for c in data.columns
        ],
        "Unique Values": [
            data[c].nunique(dropna=True)
            for c in data.columns
        ]
    })

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Numeric Range Checks"
    )

    range_rows = []

    numeric_cols = [
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "home_corners",
        "away_corners"
    ]

    for c in numeric_cols:

        if c in data.columns:

            s = pd.to_numeric(
                data[c],
                errors="coerce"
            ).dropna()

            range_rows.append({
                "Column": c,
                "Minimum": (
                    s.min()
                    if not s.empty
                    else None
                ),
                "Maximum": (
                    s.max()
                    if not s.empty
                    else None
                ),
                "Average": (
                    s.mean()
                    if not s.empty
                    else None
                ),
                "Missing": int(
                    data[c].isna().sum()
                )
            })

    st.dataframe(
        pd.DataFrame(
            range_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    if duplicate_count == 0:

        st.success(
            "No exact duplicate rows detected."
        )

    else:

        st.warning(
            f"{duplicate_count:,} exact duplicate rows detected."
        )


# ============================================================
# STEP 92
# TEAM DATA COVERAGE
# ============================================================

with tab7:

    st.subheader(
        "📈 Step 92 — Team Data Coverage"
    )

    st.caption(
        "Shows how much historical data is available "
        "for each team."
    )

    all_teams = team_list(data)

    coverage_rows = []

    for t in all_teams:

        tm = team_matches(
            data,
            t
        )

        home_count = int(
            (
                tm["venue"]
                == "Home"
            ).sum()
        )

        away_count = int(
            (
                tm["venue"]
                == "Away"
            ).sum()
        )

        dates = pd.to_datetime(
            tm["date"],
            errors="coerce"
        ).dropna()

        coverage_rows.append({
            "Team": t,
            "Total Matches": len(tm),
            "Home Matches": home_count,
            "Away Matches": away_count,
            "Competitions": tm[
                "competition"
            ].nunique(),
            "Seasons": tm[
                "season"
            ].nunique(),
            "First Match": (
                dates.min()
                if not dates.empty
                else None
            ),
            "Latest Match": (
                dates.max()
                if not dates.empty
                else None
            )
        })

    coverage_df = pd.DataFrame(
        coverage_rows
    ).sort_values(
        "Total Matches",
        ascending=False
    )

    st.dataframe(
        coverage_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Team Coverage CSV",
        coverage_df.to_csv(
            index=False
        ).encode("utf-8"),
        "team_data_coverage.csv",
        "text/csv"
    )


# ============================================================
# STEP 93
# RECENT FORM ENGINE
# ============================================================

with tab8:

    st.subheader(
        "📋 Step 93 — Recent Form Engine"
    )

    form_team = st.selectbox(
        "Team",
        team_list(data),
        key="form_team"
    )

    form_season = st.selectbox(
        "Season",
        season_list(data),
        key="form_season"
    )

    form_competition = st.selectbox(
        "Competition",
        competition_list(data),
        key="form_competition"
    )

    form_sample = st.selectbox(
        "Sample",
        [5, 10, 15],
        index=1,
        key="form_sample"
    )

    form_venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="form_venue"
    )

    form_df = filtered_team_matches(
        data,
        form_team,
        form_season,
        form_competition,
        form_venue,
        form_sample
    )

    if not form_df.empty:

        form_display = form_df[
            [
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
        ].copy()

        form_display["Result"] = np.select(
            [
                form_display[
                    "team_goals"
                ]
                >
                form_display[
                    "opp_goals"
                ],

                form_display[
                    "team_goals"
                ]
                <
                form_display[
                    "opp_goals"
                ]
            ],
            [
                "W",
                "L"
            ],
            default="D"
        )

        st.dataframe(
            form_display,
            use_container_width=True,
            hide_index=True
        )

        wins = int(
            (form_display["Result"] == "W")
            .sum()
        )

        draws = int(
            (form_display["Result"] == "D")
            .sum()
        )

        losses = int(
            (form_display["Result"] == "L")
            .sum()
        )

        f1, f2, f3, f4 = st.columns(4)

        f1.metric(
            "Matches",
            len(form_display)
        )

        f2.metric(
            "Wins",
            wins
        )

        f3.metric(
            "Draws",
            draws
        )

        f4.metric(
            "Losses",
            losses
        )

    else:

        st.info(
            "No matches available."
        )


# ============================================================
# STEP 94
# MARKET CONSISTENCY ANALYSIS
# ============================================================

with tab9:

    st.subheader(
        "📊 Step 94 — Market Consistency Analysis"
    )

    consistency_team = st.selectbox(
        "Team",
        team_list(data),
        key="consistency_team"
    )

    consistency_market = st.selectbox(
        "Market",
        BASE_MARKET_COLUMNS,
        key="consistency_market"
    )

    consistency_direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="consistency_direction"
    )

    consistency_line = st.number_input(
        "Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5,
        key="consistency_line"
    )

    consistency_venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="consistency_venue"
    )

    consistency_sample = st.selectbox(
        "Sample",
        [5, 10, 15],
        index=2,
        key="consistency_sample"
    )

    consistency_df = filtered_team_matches(
        data,
        consistency_team,
        "All",
        "All",
        consistency_venue,
        consistency_sample
    )

    column = MARKET_COLUMN_MAP[
        consistency_market
    ]

    values = pd.to_numeric(
        consistency_df[column],
        errors="coerce"
    ).dropna()

    result = analyse_market(
        consistency_df,
        consistency_market,
        consistency_direction,
        consistency_line
    )

    if not values.empty:

        if consistency_direction == "Over":
            hits_series = values > consistency_line
        else:
            hits_series = values < consistency_line

        consecutive_current = 0

        for hit in hits_series.tolist():

            if hit:
                consecutive_current += 1
            else:
                break

        consecutive_longest = 0
        running = 0

        for hit in hits_series.tolist():

            if hit:
                running += 1
                consecutive_longest = max(
                    consecutive_longest,
                    running
                )
            else:
                running = 0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Hit Rate",
            fmt_pct(
                result["hit_rate"]
            )
        )

        c2.metric(
            "Current Consecutive Hits",
            consecutive_current
        )

        c3.metric(
            "Longest Hit Run",
            consecutive_longest
        )

        c4.metric(
            "Sample",
            result["sample_size"]
        )

        consistency_stats = pd.DataFrame([
            {
                "Statistic": "Minimum",
                "Value": values.min()
            },
            {
                "Statistic": "Maximum",
                "Value": values.max()
            },
            {
                "Statistic": "Mean",
                "Value": values.mean()
            },
            {
                "Statistic": "Median",
                "Value": values.median()
            },
            {
                "Statistic": "Standard Deviation",
                "Value": values.std()
            }
        ])

        st.dataframe(
            consistency_stats,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No usable market data."
        )


# ============================================================
# STEP 95
# OPPONENT CONTEXT
# ============================================================

with tab10:

    st.subheader(
        "🧭 Step 95 — Opponent Strength / Context"
    )

    st.caption(
        "This section uses historical opponent production "
        "and concession context. It does not assign an "
        "automatic strength rating."
    )

    context_team = st.selectbox(
        "Team",
        team_list(data),
        key="context_team"
    )

    context_market = st.selectbox(
        "Market",
        BASE_MARKET_COLUMNS,
        key="context_market"
    )

    context_venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="context_venue"
    )

    context_sample = st.selectbox(
        "Sample",
        [5, 10, 15],
        index=2,
        key="context_sample"
    )

    context_df = filtered_team_matches(
        data,
        context_team,
        "All",
        "All",
        context_venue,
        context_sample
    )

    metric_map = {
        "Shots": (
            "team_shots",
            "opp_shots"
        ),
        "Shots on Target": (
            "team_sot",
            "opp_sot"
        ),
        "Corners": (
            "team_corners",
            "opp_corners"
        ),
        "Goals": (
            "team_goals",
            "opp_goals"
        )
    }

    production_col, concession_col = metric_map[
        context_market
    ]

    context_rows = []

    for opponent in sorted(
        set(
            context_df[
                "home_team"
            ].dropna()
        )
        |
        set(
            context_df[
                "away_team"
            ].dropna()
        )
    ):

        if opponent == context_team:
            continue

        opp_rows = context_df[
            (
                (
                    context_df["home_team"]
                    == opponent
                )
                |
                (
                    context_df["away_team"]
                    == opponent
                )
            )
        ]

        if opp_rows.empty:
            continue

        context_rows.append({
            "Opponent": opponent,
            "Matches": len(opp_rows),
            f"{context_team} {context_market}":
                average_value(
                    opp_rows,
                    production_col
                ),
            f"{context_team} Conceded":
                average_value(
                    opp_rows,
                    concession_col
                )
        })

    if context_rows:

        context_display = pd.DataFrame(
            context_rows
        )

        st.dataframe(
            context_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No opponent context available."
        )

    st.info(
        "Opponent-level rows are descriptive. "
        "They should be interpreted alongside sample size, "
        "venue and competition."
    )


# ============================================================
# STEP 96
# LINE SENSITIVITY
# ============================================================

with tab11:

    st.subheader(
        "🎚️ Step 96 — Line Sensitivity"
    )

    st.caption(
        "Compare nearby lines to see how historical hit rates "
        "change as the market line changes."
    )

    sensitivity_team = st.selectbox(
        "Team",
        team_list(data),
        key="sensitivity_team"
    )

    sensitivity_market = st.selectbox(
        "Market",
        BASE_MARKET_COLUMNS,
        key="sensitivity_market"
    )

    sensitivity_direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="sensitivity_direction"
    )

    sensitivity_venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="sensitivity_venue"
    )

    sensitivity_sample = st.selectbox(
        "Historical sample",
        [5, 10, 15],
        index=2,
        key="sensitivity_sample"
    )

    start_line = st.number_input(
        "Starting line",
        min_value=0.0,
        max_value=25.0,
        value=1.5,
        step=0.5,
        key="sensitivity_start"
    )

    number_lines = st.selectbox(
        "Number of lines",
        [3, 5, 7],
        index=2,
        key="sensitivity_count"
    )

    sensitivity_df = filtered_team_matches(
        data,
        sensitivity_team,
        "All",
        "All",
        sensitivity_venue,
        sensitivity_sample
    )

    sensitivity_rows = []

    for i in range(
        number_lines
    ):

        test_line = (
            start_line
            + i * 0.5
        )

        result = analyse_market(
            sensitivity_df,
            sensitivity_market,
            sensitivity_direction,
            test_line
        )

        sensitivity_rows.append({
            "Line": test_line,
            "Hits": result["hits"],
            "Misses": result["misses"],
            "Sample": result["sample_size"],
            "Historical Hit Rate":
                fmt_pct(
                    result["hit_rate"]
                )
        })

    sensitivity_display = pd.DataFrame(
        sensitivity_rows
    )

    st.dataframe(
        sensitivity_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This table describes historical sensitivity to the "
        "line. It does not identify a preferred line."
    )


# ============================================================
# STEP 97
# MARKET STABILITY PANEL
# ============================================================

with tab12:

    st.subheader(
        "🧭 Step 97 — Market Stability Panel"
    )

    stability_team = st.selectbox(
        "Team",
        team_list(data),
        key="stability_team"
    )

    stability_market = st.selectbox(
        "Market",
        BASE_MARKET_COLUMNS,
        key="stability_market"
    )

    stability_direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="stability_direction"
    )

    stability_line = st.number_input(
        "Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5,
        key="stability_line"
    )

    stability_venue = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="stability_venue"
    )

    stability_df = filtered_team_matches(
        data,
        stability_team,
        "All",
        "All",
        stability_venue,
        15
    )

    stability_result = analyse_market(
        stability_df,
        stability_market,
        stability_direction,
        stability_line
    )

    stability_column = MARKET_COLUMN_MAP[
        stability_market
    ]

    stability_values = pd.to_numeric(
        stability_df[
            stability_column
        ],
        errors="coerce"
    ).dropna()

    if not stability_values.empty:

        st1, st2, st3, st4 = st.columns(4)

        st1.metric(
            "Historical Hit Rate",
            fmt_pct(
                stability_result[
                    "hit_rate"
                ]
            )
        )

        st2.metric(
            "Sample",
            stability_result[
                "sample_size"
            ]
        )

        st3.metric(
            "Median",
            fmt_num(
                stability_values.median()
            )
        )

        st4.metric(
            "Std Dev",
            fmt_num(
                stability_values.std()
            )
        )

        cv = (
            stability_values.std()
            /
            stability_values.mean()
            if (
                stability_values.mean()
                != 0
            )
            else None
        )

        stability_rows = [
            {
                "Measure": "Mean",
                "Value":
                    stability_values.mean()
            },
            {
                "Measure": "Median",
                "Value":
                    stability_values.median()
            },
            {
                "Measure": "Minimum",
                "Value":
                    stability_values.min()
            },
            {
                "Measure": "Maximum",
                "Value":
                    stability_values.max()
            },
            {
                "Measure": "Standard Deviation",
                "Value":
                    stability_values.std()
            },
            {
                "Measure": "Coefficient of Variation",
                "Value": cv
            }
        ]

        st.dataframe(
            pd.DataFrame(
                stability_rows
            ),
            use_container_width=True,
            hide_index=True
        )

        quality_label, quality_message = sample_quality(
            stability_result[
                "sample_size"
            ]
        )

        st.info(
            f"{quality_label}: {quality_message}"
        )

    else:

        st.info(
            "No stability data available."
        )


# ============================================================
# STEP 98
# MATCH RESEARCH SCORECARD
# ============================================================

with tab13:

    st.subheader(
        "📝 Step 98 — Match Research Scorecard"
    )

    st.caption(
        "A structured checklist for documenting research. "
        "It intentionally does not generate a betting score."
    )

    score_home = st.selectbox(
        "Home Team",
        team_list(data),
        key="score_home"
    )

    score_away_options = [
        t for t in team_list(data)
        if t != score_home
    ]

    score_away = st.selectbox(
        "Away Team",
        score_away_options,
        key="score_away"
    )

    score_market = st.selectbox(
        "Market researched",
        BASE_MARKET_COLUMNS,
        key="score_market"
    )

    score_direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="score_direction"
    )

    score_line = st.number_input(
        "Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5,
        key="score_line"
    )

    score_odds = st.number_input(
        "Odds observed",
        min_value=1.01,
        max_value=100.0,
        value=1.30,
        step=0.01,
        key="score_odds"
    )

    st.divider()

    st.subheader(
        "Research Checklist"
    )

    checks = {}

    checklist = [
        "Historical team production reviewed",
        "Home/away split reviewed",
        "Opponent defensive context reviewed",
        "Recent match-by-match results reviewed",
        "Market line tested",
        "Nearby lines compared",
        "Sample size checked",
        "Head-to-head reviewed where available",
        "Team news / lineup information checked",
        "Tactical context considered",
        "Market price recorded"
    ]

    for item in checklist:

        checks[item] = st.checkbox(
            item,
            key=f"check_{item}"
        )

    completed = sum(
        checks.values()
    )

    total_checks = len(
        checklist
    )

    st.metric(
        "Research items completed",
        f"{completed}/{total_checks}"
    )

    notes = st.text_area(
        "Research notes",
        height=200,
        key="scorecard_notes"
    )

    if st.button(
        "Save Scorecard to Audit",
        key="save_scorecard"
    ):

        audit_entry = {
            "Timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "Home Team": score_home,
            "Away Team": score_away,
            "Market": score_market,
            "Direction": score_direction,
            "Line": score_line,
            "Odds": score_odds,
            "Checklist Completed": completed,
            "Checklist Total": total_checks,
            "Checklist %": (
                completed / total_checks
                if total_checks
                else 0
            ),
            "Notes": notes,
            "Outcome": "",
            "Actual Result": ""
        }

        if os.path.exists(
            AUDIT_FILE
        ):

            existing = pd.read_csv(
                AUDIT_FILE
            )

            updated = pd.concat(
                [
                    existing,
                    pd.DataFrame([
                        audit_entry
                    ])
                ],
                ignore_index=True
            )

        else:

            updated = pd.DataFrame([
                audit_entry
            ])

        updated.to_csv(
            AUDIT_FILE,
            index=False
        )

        st.success(
            "Research scorecard saved to audit trail."
        )


# ============================================================
# STEP 99
# RESEARCH AUDIT TRAIL
# ============================================================

with tab14:

    st.subheader(
        "🗂️ Step 99 — Research Audit Trail"
    )

    st.caption(
        "Records what was researched and allows outcomes "
        "to be entered later."
    )

    if os.path.exists(
        AUDIT_FILE
    ):

        try:

            audit_df = pd.read_csv(
                AUDIT_FILE
            ).fillna("")

        except Exception:

            audit_df = pd.DataFrame()

    else:

        audit_df = pd.DataFrame()

    if not audit_df.empty:

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader(
            "Add Outcome"
        )

        audit_index = st.number_input(
            "Audit row number",
            min_value=1,
            max_value=len(audit_df),
            value=1,
            step=1,
            key="audit_index"
        )

        selected_idx = (
            int(audit_index) - 1
        )

        actual_result = st.number_input(
            "Actual market result",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            key="audit_actual"
        )

        if st.button(
            "Settle Audit Record",
            key="settle_audit"
        ):

            row = audit_df.iloc[
                selected_idx
            ]

            try:

                selected_line = float(
                    row["Line"]
                )

                if row["Direction"] == "Over":

                    won = (
                        actual_result
                        > selected_line
                    )

                else:

                    won = (
                        actual_result
                        < selected_line
                    )

                audit_df.loc[
                    selected_idx,
                    "Actual Result"
                ] = actual_result

                audit_df.loc[
                    selected_idx,
                    "Outcome"
                ] = (
                    "Won"
                    if won
                    else "Lost"
                )

                audit_df.to_csv(
                    AUDIT_FILE,
                    index=False
                )

                st.success(
                    "Audit record settled."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not settle record: {exc}"
                )

        st.download_button(
            "Download Research Audit CSV",
            audit_df.to_csv(
                index=False
            ).encode("utf-8"),
            "research_audit_log.csv",
            "text/csv",
            key="download_audit"
        )

    else:

        st.info(
            "No research records have been saved yet. "
            "Use the Match Research Scorecard to create one."
        )


# ============================================================
# STEP 100
# RESEARCH PERFORMANCE DASHBOARD
# ============================================================

st.divider()

# This section is displayed in the final tab through a
# separate conditional because the tabs are already created.

with tab14:

    st.divider()

    st.subheader(
        "🏁 Step 100 — Research Performance Dashboard"
    )

    if os.path.exists(
        AUDIT_FILE
    ):

        try:

            performance_df = pd.read_csv(
                AUDIT_FILE
            ).fillna("")

        except Exception:

            performance_df = pd.DataFrame()

    else:

        performance_df = pd.DataFrame()

    if not performance_df.empty:

        outcome_series = (
            performance_df[
                "Outcome"
            ]
            .astype(str)
            .str.strip()
        )

        total_researched = len(
            performance_df
        )

        settled_df = performance_df[
            outcome_series.isin(
                [
                    "Won",
                    "Lost"
                ]
            )
        ]

        won_count = int(
            (
                settled_df[
                    "Outcome"
                ]
                == "Won"
            ).sum()
        )

        lost_count = int(
            (
                settled_df[
                    "Outcome"
                ]
                == "Lost"
            ).sum()
        )

        settled_count = (
            won_count
            + lost_count
        )

        actual_hit_rate = (
            won_count / settled_count
            if settled_count > 0
            else None
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Research Records",
            total_researched
        )

        p2.metric(
            "Settled",
            settled_count
        )

        p3.metric(
            "Won",
            won_count
        )

        p4.metric(
            "Actual Settled Hit Rate",
            fmt_pct(
                actual_hit_rate
            )
        )

        st.caption(
            "Actual settled hit rate describes the records "
            "you have entered into the audit system. It is "
            "not a forecast of future performance."
        )

        st.divider()

        st.subheader(
            "Market-by-Market Audit"
        )

        market_perf_rows = []

        for market_name in sorted(
            performance_df[
                "Market"
            ]
            .dropna()
            .unique()
        ):

            subset = performance_df[
                performance_df[
                    "Market"
                ] == market_name
            ]

            subset_settled = subset[
                subset[
                    "Outcome"
                ].isin(
                    [
                        "Won",
                        "Lost"
                    ]
                )
            ]

            market_wins = int(
                (
                    subset_settled[
                        "Outcome"
                    ]
                    == "Won"
                ).sum()
            )

            market_settled = len(
                subset_settled
            )

            market_perf_rows.append({
                "Market": market_name,
                "Research Records": len(
                    subset
                ),
                "Settled": market_settled,
                "Won": market_wins,
                "Lost": (
                    market_settled
                    - market_wins
                ),
                "Actual Hit Rate": (
                    f"{market_wins / market_settled * 100:.1f}%"
                    if market_settled
                    else "—"
                )
            })

        if market_perf_rows:

            st.dataframe(
                pd.DataFrame(
                    market_perf_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader(
            "Team-by-Team Audit"
        )

        team_perf_rows = []

        audit_teams = sorted(
            set(
                performance_df[
                    "Home Team"
                ].dropna()
            )
            |
            set(
                performance_df[
                    "Away Team"
                ].dropna()
            )
        )

        for t in audit_teams:

            subset = performance_df[
                (
                    performance_df[
                        "Home Team"
                    ] == t
                )
                |
                (
                    performance_df[
                        "Away Team"
                    ] == t
                )
            ]

            settled_subset = subset[
                subset[
                    "Outcome"
                ].isin(
                    [
                        "Won",
                        "Lost"
                    ]
                )
            ]

            wins_t = int(
                (
                    settled_subset[
                        "Outcome"
                    ]
                    == "Won"
                ).sum()
            )

            settled_t = len(
                settled_subset
            )

            team_perf_rows.append({
                "Team": t,
                "Research Records": len(
                    subset
                ),
                "Settled": settled_t,
                "Won": wins_t,
                "Lost": (
                    settled_t
                    - wins_t
                ),
                "Actual Hit Rate": (
                    f"{wins_t / settled_t * 100:.1f}%"
                    if settled_t
                    else "—"
                )
            })

        if team_perf_rows:

            st.dataframe(
                pd.DataFrame(
                    team_perf_rows
                ).sort_values(
                    "Research Records",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader(
            "Research Completeness"
        )

        if "Checklist %" in performance_df.columns:

            checklist_values = pd.to_numeric(
                performance_df[
                    "Checklist %"
                ],
                errors="coerce"
            ).dropna()

            if not checklist_values.empty:

                st.metric(
                    "Average Research Completeness",
                    f"{checklist_values.mean() * 100:.1f}%"
                )

                completeness_df = performance_df[
                    [
                        "Timestamp",
                        "Home Team",
                        "Away Team",
                        "Market",
                        "Checklist Completed",
                        "Checklist Total",
                        "Checklist %"
                    ]
                ].copy()

                completeness_df[
                    "Checklist %"
                ] = (
                    pd.to_numeric(
                        completeness_df[
                            "Checklist %"
                        ],
                        errors="coerce"
                    )
                    * 100
                ).round(1)

                st.dataframe(
                    completeness_df,
                    use_container_width=True,
                    hide_index=True
                )

    else:

        st.info(
            "The performance dashboard will populate after "
            "research records are saved and settled."
        )


# ============================================================
# FINAL DATASET STATUS
# ============================================================

st.divider()

st.caption(
    f"Primary dataset: {MASTER_FILE} | "
    f"{len(data):,} matches loaded | "
    f"{data['season'].nunique()} seasons | "
    f"{data['competition'].nunique()} competitions | "
    f"{len(team_list(data)):,} teams"
)

st.caption(
    "Steps 91–100 add data-quality checks, coverage analysis, "
    "recent-form research, market consistency, line sensitivity, "
    "context analysis, a research scorecard, an audit trail and "
    "performance tracking. These features describe historical "
    "evidence and recorded research outcomes; they do not "
    "produce automatic 'safe bet' labels or future probabilities."
)
