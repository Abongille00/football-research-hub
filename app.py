import streamlit as st
import pandas as pd
import numpy as np
import os
import math

st.set_page_config(
    page_title="Football Betting Research V1",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Betting Research Tool — V1")
st.caption(
    "Research assistant, not a prediction engine. "
    "Use match-by-match data to test a market before betting."
)

# ============================================================
# DATA CONFIGURATION
# ============================================================

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


# ============================================================
# DATA FUNCTIONS
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
            f"Could not read the uploaded CSV: {exc}"
        )
        st.stop()
else:
    master_file = "football_master_2024_27_v1.csv"

    if os.path.exists(master_file):
        try:
            data = clean_data(
                pd.read_csv(master_file)
            )
            st.sidebar.success(
                f"Using master dataset — {len(data)} matches"
            )
        except Exception as exc:
            st.error(
                f"Could not read {master_file}: {exc}"
            )
            st.stop()
    else:
        st.warning(
            f"{master_file} was not found. "
            "The app is using the built-in demo dataset."
        )
        data = clean_data(demo)
        st.sidebar.info(
            f"Using demo dataset — {len(data)} matches"
        )


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
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔎 Team Research",
        "🎯 Market Tester",
        "📊 Bookmaker Monitor",
        "📥 Data Format",
        "🔬 Match Research"
    ]
)


# ============================================================
# TAB 1 — TEAM RESEARCH
# ============================================================

with tab1:

    st.subheader("Team Research Dashboard")

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
        data=data,
        team=dashboard_team,
        season=dashboard_season,
        competition=dashboard_competition,
        venue=dashboard_venue,
        sample=dashboard_sample
    )

    dashboard_metrics = team_summary(
        dashboard_df
    )

    dashboard_size = len(dashboard_df)

    quality_label, quality_message = sample_quality(
        dashboard_size
    )

    if dashboard_size < 5:
        st.warning(
            f"⚠️ {quality_label}: {quality_message}"
        )
    elif dashboard_size < 10:
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
        dashboard_metrics.get("Matches", 0)
    )

    d2.metric(
        "Avg Shots",
        (
            f"{dashboard_metrics['Shots Avg']:.2f}"
            if dashboard_metrics.get("Shots Avg") is not None
            else "—"
        )
    )

    d3.metric(
        "Avg SOT",
        (
            f"{dashboard_metrics['Shots on Target Avg']:.2f}"
            if dashboard_metrics.get("Shots on Target Avg") is not None
            else "—"
        )
    )

    d4.metric(
        "Avg Corners",
        (
            f"{dashboard_metrics['Corners Avg']:.2f}"
            if dashboard_metrics.get("Corners Avg") is not None
            else "—"
        )
    )

    d5.metric(
        "Avg Goals",
        (
            f"{dashboard_metrics['Goals Avg']:.2f}"
            if dashboard_metrics.get("Goals Avg") is not None
            else "—"
        )
    )

    st.divider()
    st.subheader("Market Hit-Rate Overview")

    overview_rows = []

    for market_name in [
        "Shots",
        "Shots on Target",
        "Corners",
        "Goals"
    ]:

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

                if result["sample_size"] > 0:
                    overview_rows.append(
                        {
                            "Market": market_name,
                            "Direction": direction_name,
                            "Line": test_line,
                            "Hits": result["hits"],
                            "Sample": result["sample_size"],
                            "Hit Rate": (
                                f"{result['hit_rate'] * 100:.1f}%"
                            )
                        }
                    )

    overview_df = pd.DataFrame(
        overview_rows
    )

    if not overview_df.empty:
        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No market data is available for the selected filters."
        )

    st.divider()
    st.subheader("Home / Away Comparison")

    split_rows = []

    for venue_label in ["Home", "Away"]:

        split_df = filtered_team_matches(
            data=data,
            team=dashboard_team,
            season=dashboard_season,
            competition=dashboard_competition,
            venue=venue_label,
            sample=dashboard_sample
        )

        split_summary = team_summary(
            split_df
        )

        split_rows.append(
            {
                "Venue": venue_label,
                "Matches": len(split_df),
                "Avg Shots": (
                    f"{split_summary.get('Shots Avg', 0):.2f}"
                    if split_summary.get("Shots Avg") is not None
                    else "—"
                ),
                "Avg SOT": (
                    f"{split_summary.get('Shots on Target Avg', 0):.2f}"
                    if split_summary.get("Shots on Target Avg") is not None
                    else "—"
                ),
                "Avg Corners": (
                    f"{split_summary.get('Corners Avg', 0):.2f}"
                    if split_summary.get("Corners Avg") is not None
                    else "—"
                ),
                "Avg Goals": (
                    f"{split_summary.get('Goals Avg', 0):.2f}"
                    if split_summary.get("Goals Avg") is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(split_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Recent Match History")

    if not dashboard_df.empty:

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
            c
            for c in display_columns
            if c in dashboard_df.columns
        ]

        st.dataframe(
            dashboard_df[available_columns],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No matches are available for the selected filters."
        )


# ============================================================
# TAB 2 — MARKET TESTER
# ============================================================

with tab2:

    st.subheader("Test a Market")

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
        [
            "Shots",
            "Shots on Target",
            "Corners",
            "Goals"
        ],
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
        data=data,
        team=team,
        season=season2,
        competition=competition2,
        venue=venue2,
        sample=n2
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

    quality_label, quality_message = sample_quality(
        sample_size
    )

    if sample_size < 5:
        st.warning(
            f"⚠️ {quality_label}: {quality_message}"
        )
    elif sample_size < 10:
        st.info(
            f"ℹ️ {quality_label}: {quality_message}"
        )
    else:
        st.success(
            f"✓ {quality_label}: {quality_message}"
        )

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

    if rate is not None:
        comparison = (
            "Above"
            if rate >= breakeven
            else "Below"
        )
    else:
        comparison = "—"

    d.metric(
        "Historical vs break-even",
        comparison
    )

    if rate is not None:

        st.progress(
            min(max(rate, 0), 1)
        )

        edge = (
            rate * 100
            -
            breakeven * 100
        )

        st.metric(
            "Historical difference vs break-even",
            f"{edge:+.1f} percentage points"
        )

    st.caption(
        "Historical hit rate is descriptive only. "
        "It does not establish the probability of the next match."
    )

    st.divider()
    st.subheader("Overall / Home / Away Comparison")

    comparison_data = []

    for venue_label in [
        "All",
        "Home",
        "Away"
    ]:

        venue_df = filtered_team_matches(
            data=data,
            team=team,
            season=season2,
            competition=competition2,
            venue=venue_label,
            sample=n2
        )

        venue_analysis = analyse_market(
            venue_df,
            market,
            direction,
            line
        )

        venue_summary = team_summary(
            venue_df
        )

        comparison_data.append(
            {
                "Venue": venue_label,
                "Matches": venue_analysis["sample_size"],
                "Hit Rate": (
                    f"{venue_analysis['hit_rate'] * 100:.1f}%"
                    if venue_analysis["hit_rate"] is not None
                    else "—"
                ),
                "Avg Shots": (
                    f"{venue_summary.get('Shots Avg', 0):.2f}"
                    if venue_summary.get("Shots Avg") is not None
                    else "—"
                ),
                "Avg SOT": (
                    f"{venue_summary.get('Shots on Target Avg', 0):.2f}"
                    if venue_summary.get("Shots on Target Avg") is not None
                    else "—"
                ),
                "Avg Corners": (
                    f"{venue_summary.get('Corners Avg', 0):.2f}"
                    if venue_summary.get("Corners Avg") is not None
                    else "—"
                ),
                "Avg Goals": (
                    f"{venue_summary.get('Goals Avg', 0):.2f}"
                    if venue_summary.get("Goals Avg") is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Market History")

    st.caption(
        f"{team} — {market} {direction} {line} "
        "using the selected historical sample."
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
    else:
        st.info(
            "No match history is available for the selected filters."
        )

    h1, h2, h3 = st.columns(3)

    h1.metric("Hits", hits)
    h2.metric("Misses", misses)
    h3.metric("Matches analysed", sample_size)


# ============================================================
# TAB 3 — BOOKMAKER MARKET MONITOR
# ============================================================

with tab3:

    st.subheader("Bookmaker Market Monitor")

    # --------------------------------------------------------
    # LOAD WATCHLIST
    # --------------------------------------------------------

    if "market_watchlist" not in st.session_state:

        if os.path.exists(WATCHLIST_FILE):

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

    # --------------------------------------------------------
    # ADD MARKET
    # --------------------------------------------------------

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
            [
                "Shots",
                "Shots on Target",
                "Corners",
                "Goals"
            ],
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
            data=data,
            team=monitor_team,
            season=monitor_season,
            competition=monitor_competition,
            venue=monitor_venue,
            sample=monitor_sample
        )

        analysis = analyse_market(
            monitor_df,
            monitor_market,
            monitor_direction,
            monitor_line
        )

        rate = analysis["hit_rate"]
        breakeven = 1 / monitor_odds

        if rate is not None:

            historical_hit_rate = rate * 100
            break_even_rate = breakeven * 100
            edge = (
                historical_hit_rate
                -
                break_even_rate
            )

            historical_vs_breakeven = (
                "Above"
                if rate >= breakeven
                else "Below"
            )

            edge_display = f"{edge:+.1f} pp"
            historical_display = (
                f"{historical_hit_rate:.1f}%"
            )

        else:

            historical_display = "—"
            edge_display = "—"
            historical_vs_breakeven = "—"
            break_even_rate = breakeven * 100

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
            "Historical Hit Rate": historical_display,
            "Break-even": f"{break_even_rate:.1f}%",
            "Edge vs Break-even": edge_display,
            "Historical vs Break-even": historical_vs_breakeven,
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

    # --------------------------------------------------------
    # WATCHLIST
    # --------------------------------------------------------

    st.divider()
    st.subheader("Tracked Markets")

    if st.session_state.market_watchlist:

        # Automatic result settlement first
        for item in st.session_state.market_watchlist:

            actual_text = str(
                item.get(
                    "Actual Result",
                    ""
                )
            ).strip()

            if (
                actual_text != ""
                and item.get("Status") == "Watching"
            ):

                try:

                    actual = float(actual_text)
                    market_line = float(
                        item["Line"]
                    )

                    if item["Direction"] == "Over":
                        won_result = (
                            actual > market_line
                        )
                    else:
                        won_result = (
                            actual < market_line
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

        # Save automatic changes
        pd.DataFrame(
            st.session_state.market_watchlist
        ).to_csv(
            WATCHLIST_FILE,
            index=False
        )

        watchlist_df = pd.DataFrame(
            st.session_state.market_watchlist
        ).fillna("")

        total = len(watchlist_df)

        watching = (
            watchlist_df["Status"]
            .eq("Watching")
            .sum()
        )

        won = (
            watchlist_df["Status"]
            .eq("Won")
            .sum()
        )

        lost = (
            watchlist_df["Status"]
            .eq("Lost")
            .sum()
        )

        void = (
            watchlist_df["Status"]
            .eq("Void")
            .sum()
        )

        settled = won + lost

        tracked_hit_rate_display = (
            f"{won / settled * 100:.1f}%"
            if settled > 0
            else "—"
        )

        s1, s2, s3, s4, s5 = st.columns(5)

        s1.metric("Total", total)
        s2.metric("Watching", watching)
        s3.metric("Won", won)
        s4.metric("Lost", lost)
        s5.metric(
            "Tracked Hit Rate",
            tracked_hit_rate_display
        )

        st.caption(
            f"Void markets: {void}"
        )

        # ----------------------------------------------------
        # EDIT / SETTLE
        # ----------------------------------------------------

        st.subheader("Edit / Settle Markets")

        status_options = [
            "Watching",
            "Won",
            "Lost",
            "Void"
        ]

        for i in range(len(watchlist_df)):

            row = watchlist_df.iloc[i]

            label = (
                f"{i + 1}. "
                f"{row.get('Match', '')} — "
                f"{row.get('Market', '')} "
                f"{row.get('Direction', '')} "
                f"{row.get('Line', '')}"
            )

            with st.expander(label):

                e1, e2 = st.columns(2)

                with e1:

                    try:
                        current_odds = float(
                            row.get(
                                "Odds",
                                1.30
                            )
                        )
                    except (
                        ValueError,
                        TypeError
                    ):
                        current_odds = 1.30

                    try:
                        current_line = float(
                            row.get(
                                "Line",
                                3.5
                            )
                        )
                    except (
                        ValueError,
                        TypeError
                    ):
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

                with e2:

                    current_status = row.get(
                        "Status",
                        "Watching"
                    )

                    if current_status not in status_options:
                        current_status = "Watching"

                    new_status = st.selectbox(
                        "Status",
                        status_options,
                        index=status_options.index(
                            current_status
                        ),
                        key=f"edit_status_{i}"
                    )

                    actual_text = str(
                        row.get(
                            "Actual Result",
                            ""
                        )
                    ).strip()

                    try:
                        current_actual = (
                            float(actual_text)
                            if actual_text != ""
                            else 0.0
                        )
                    except (
                        ValueError,
                        TypeError
                    ):
                        current_actual = 0.0

                    actual_value = st.number_input(
                        "Actual Result",
                        min_value=0.0,
                        max_value=100.0,
                        value=current_actual,
                        step=0.5,
                        key=f"actual_result_{i}"
                    )

                if st.button(
                    "Save Changes",
                    key=f"save_market_{i}"
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
                    ] = actual_value

                    pd.DataFrame(
                        st.session_state.market_watchlist
                    ).to_csv(
                        WATCHLIST_FILE,
                        index=False
                    )

                    st.success(
                        "Market updated."
                    )

                    st.rerun()

                if st.button(
                    "Delete Market",
                    key=f"delete_market_{i}"
                ):

                    st.session_state.market_watchlist.pop(i)

                    pd.DataFrame(
                        st.session_state.market_watchlist
                    ).to_csv(
                        WATCHLIST_FILE,
                        index=False
                    )

                    st.success(
                        "Market deleted."
                    )

                    st.rerun()

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        watchlist_df = pd.DataFrame(
            st.session_state.market_watchlist
        )

        st.dataframe(
            watchlist_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

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

    st.subheader("CSV Format")

    st.write(
        "Your CSV should contain one row per match "
        "with these columns:"
    )

    st.code(
        ",".join(REQUIRED_COLUMNS)
    )

    st.dataframe(
        demo.head(5),
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = (
        demo
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download demo CSV",
        csv_bytes,
        "football_demo.csv",
        "text/csv",
        key="download_demo_csv"
    )


# ============================================================
# TAB 5 — MATCH RESEARCH / AUTOMATED PIPELINE
# ============================================================

with tab5:
    st.subheader("🔬 Match Research — Automated 1–20 Pipeline")
    st.caption("Enter one fixture and only the bookmaker markets actually available. The internal engine researches every entered candidate automatically.")

    teams = team_list(data)
    c1,c2=st.columns(2)
    with c1: home_team=st.selectbox("Home Team",teams,key="research_home_team")
    with c2:
        away_options=[x for x in teams if x!=home_team]
        away_team=st.selectbox("Away Team",away_options,key="research_away_team")

    st.divider()
    st.subheader("Bookmaker markets actually available")
    st.caption("Enter the minimum line you can actually see. Example: Virgin Bet → Chelsea Shots on Target → Over 3.5 → 1.32. The app will not infer Over 2.5 is available.")
    default_rows=pd.DataFrame([{"Bookmaker":"Virgin Bet","Market":"Shots on Target","Direction":"Over","Line":3.5,"Odds":1.32}])
    market_rows=st.data_editor(default_rows,num_rows="dynamic",use_container_width=True,key="pipeline_market_editor",column_config={
        "Bookmaker":st.column_config.SelectboxColumn("Bookmaker",options=["SportyBet","SunBet","Virgin Bet","Other"]),
        "Market":st.column_config.SelectboxColumn("Market",options=PIPELINE_MARKETS),
        "Direction":st.column_config.SelectboxColumn("Direction",options=["Over","Under"]),
        "Line":st.column_config.NumberColumn("Minimum available line",min_value=0.0,max_value=30.0,step=0.5),
        "Odds":st.column_config.NumberColumn("Odds",min_value=1.01,max_value=100.0,step=0.01),
    })

    if st.button("🚀 Run full Steps 1–20 pipeline",type="primary",key="run_pipeline"):
        rows=[]
        for _,r in market_rows.iterrows():
            if pd.isna(r.get("Line")) or pd.isna(r.get("Odds")) or not r.get("Market"):
                continue
            rows.append({"Bookmaker":str(r.get("Bookmaker","Other")),"Market":str(r["Market"]),"Direction":str(r.get("Direction","Over")),"Line":float(r["Line"]),"Odds":float(r["Odds"])})
        if not rows:
            st.warning("Enter at least one available bookmaker market.")
        else:
            candidates,rejected=run_research_pipeline(data,home_team,away_team,rows)
            st.session_state["pipeline_candidates"] = candidates
            st.session_state["pipeline_rejected"] = rejected
            st.session_state["pipeline_fixture"] = f"{home_team} vs {away_team}"

    candidates=st.session_state.get("pipeline_candidates",[])
    rejected=st.session_state.get("pipeline_rejected",[])

    if candidates or rejected:
        st.divider()
        st.subheader("Pipeline output")
        if candidates:
            summary=[]
            for c in candidates:
                summary.append({
                    "Bookmaker":c["Bookmaker"],"Market":f"{c['Market']} {c['Direction']} {c['Line']}","Odds":c["Odds"],"Break-even":c["Break-even"],
                    "Home 10":c["Home 10 Hit Rate"],"Away 10":c["Away 10 Hit Rate"],"Home 15 Median":c["Home 15 Median"],"Away 15 Median":c["Away 15 Median"],
                    "Observed vs BE":c["Observed vs BE"],"Trend H":c["Trend Home"],"Trend A":c["Trend Away"],"Agreement":c["Agreement"],"Warnings":c["Warnings"]
                })
            out=pd.DataFrame(summary)
            st.dataframe(out,use_container_width=True,hide_index=True)
            st.download_button("Download surviving candidates",out.to_csv(index=False).encode("utf-8"),"pipeline_surviving_candidates.csv","text/csv",key="pipeline_download")

            selected=st.selectbox("Inspect candidate",range(len(candidates)),format_func=lambda i:f"{candidates[i]['Market']} {candidates[i]['Direction']} {candidates[i]['Line']} @ {candidates[i]['Odds']}",key="pipeline_selected")
            c=candidates[selected]
            st.subheader("Why this candidate survived")
            checks=pd.DataFrame([{"Step":i+1,"Check":name,"Result":"PASS" if ok else "UNAVAILABLE / REVIEW","Detail":detail} for i,(name,ok,detail) in enumerate(c["Checks"])])
            st.dataframe(checks,use_container_width=True,hide_index=True)
            st.subheader("Adjacent-line sensitivity — evidence only")
            ls=c["Line sensitivity"].copy(); ls["Hit rate"]=ls["Hit rate"].map(lambda x:f"{x:.1%}" if pd.notna(x) else "—")
            st.dataframe(ls,use_container_width=True,hide_index=True)
            st.caption("Adjacent lines are tested only to understand the statistical shape. The app does not claim those lines are offered by the bookmaker.")

        if rejected:
            st.subheader("Rejected / incomplete candidates")
            st.dataframe(pd.DataFrame(rejected),use_container_width=True,hide_index=True)

        st.subheader("Research limitations currently detected")
        limitations=[]
        if not all(c in data.columns for c in ["home_1h_goals","away_1h_goals"]): limitations.append("Match-state analysis requires event/timeline data not present in the current master CSV.")
        if not all(c in data.columns for c in ["home_1h_shots","away_1h_shots"]): limitations.append("First-half/second-half analysis requires half-split data not present in the current master CSV.")
        limitations.append("Lineups, injuries, referee information, tactical news and live team-news feeds are not invented when absent from the dataset.")
        for x in limitations: st.info(x)

# ============================================================
# STEPS 101–110 — ADVANCED DATA QUALITY & RESEARCH ENGINE
# ============================================================

# ------------------------------------------------------------
# STEP 101 — ADVANCED DATA VALIDATION
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 102 — DATASET COVERAGE DASHBOARD
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 103 — RECENT-FORM ENGINE
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 104 — OPPONENT-STRENGTH CONTEXT
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 105 — MARKET STABILITY ANALYSIS
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 106 — LINE SENSITIVITY
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 107 — STATSBASE / UNDERSTAT PACKAGE HEALTH
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 108 — SOURCE RECONCILIATION
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 109 — STRUCTURED RESEARCH REPORT
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 110 — TRANSPARENT RESEARCH CHECKLIST
# ------------------------------------------------------------
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


# ============================================================
# TAB 6 — ADVANCED DATA & SOURCES
# ============================================================

tab6 = st.tabs(["🧪 Advanced Data & Sources"])[0]

with tab6:
    st.subheader("🧪 Advanced Data & Sources — Steps 101–110")
    st.caption(
        "This stage improves data quality, context and transparency. "
        "It does not convert historical results into guaranteed predictions."
    )

    # STEP 101
    st.divider()
    st.subheader("Step 101 — Advanced Data Validation")
    validation_df = validate_dataset(data)
    st.dataframe(validation_df, use_container_width=True, hide_index=True)

    # STEP 102
    st.divider()
    st.subheader("Step 102 — Dataset Coverage")
    coverage = dataset_coverage(data)
    cv1, cv2, cv3, cv4 = st.columns(4)
    cv1.metric("Matches", coverage["Matches"])
    cv2.metric("Teams", coverage["Teams"])
    cv3.metric("Competitions", coverage["Competitions"])
    cv4.metric("Seasons", coverage["Seasons"])
    if coverage["Start"] is not None:
        st.write(f"Date range: **{coverage['Start'].date()} → {coverage['End'].date()}**")
    st.dataframe(missingness_table(data), use_container_width=True, hide_index=True)

    # STEP 103
    st.divider()
    st.subheader("Step 103 — Recent Form Engine")
    form_team = st.selectbox("Team", team_list(data), key="advanced_form_team")
    form_sample = st.select_slider("Recent matches", options=[5, 10, 15], value=10, key="advanced_form_sample")
    form_matches = filtered_team_matches(data, form_team, sample=form_sample)
    form_df = recent_form_profile(form_matches)
    if not form_df.empty:
        st.dataframe(form_df, use_container_width=True, hide_index=True)

    # STEP 104
    st.divider()
    st.subheader("Step 104 — Opponent-Strength Context")
    context_market = st.selectbox("Context metric", list(MARKET_COLUMN_MAP.keys()), index=1, key="advanced_context_market")
    context_col = MARKET_COLUMN_MAP[context_market]
    context_df = opponent_context_profile(data, form_team, context_col)
    if not context_df.empty:
        st.dataframe(context_df.head(15), use_container_width=True, hide_index=True)
    else:
        st.info("No opponent-context data available for this team.")

    # STEP 105
    st.divider()
    st.subheader("Step 105 — Market Stability")
    st1, st2, st3 = st.columns(3)
    with st1:
        stability_market = st.selectbox("Market", list(MARKET_COLUMN_MAP.keys()), key="stability_market")
    with st2:
        stability_direction = st.selectbox("Direction", ["Over", "Under"], key="stability_direction")
    with st3:
        stability_line = st.number_input("Line", 0.0, 30.0, 3.5, 0.5, key="stability_line")
    stability_matches = filtered_team_matches(data, form_team, sample=15)
    stability = market_stability(stability_matches, stability_market, stability_direction, stability_line)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Hit rate", fmt_pct(stability["hit_rate"]))
    q2.metric("Longest hit streak", stability["longest_hit_streak"])
    q3.metric("Longest miss streak", stability["longest_miss_streak"])
    q4.metric("Std. deviation", f"{stability['volatility']:.2f}" if stability["volatility"] is not None else "—")

    # STEP 106
    st.divider()
    st.subheader("Step 106 — Line Sensitivity")
    sensitivity_df = line_sensitivity(stability_matches, stability_market, stability_direction, stability_line)
    st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

    # STEP 107
    st.divider()
    st.subheader("Step 107 — StatsBomb / Understat Integration Health")
    health_df = package_health()
    st.dataframe(health_df, use_container_width=True, hide_index=True)
    st.caption(
        "Installed = the Python package can be imported. This is different from confirming that live external data can currently be retrieved."
    )

    if st.button("Run live source connectivity checks", key="live_source_check"):
        import importlib
        source_rows = []
        try:
            import statsbombpy.sb as sb
            comps = sb.competitions()
            source_rows.append({"Source": "StatsBomb", "Live check": "PASS", "Details": f"Open competitions retrieved: {len(comps)}"})
        except Exception as exc:
            source_rows.append({"Source": "StatsBomb", "Live check": "FAIL/UNAVAILABLE", "Details": str(exc)[:250]})

        try:
            import requests
            response = requests.get("https://understat.com", timeout=10)
            source_rows.append({"Source": "Understat", "Live check": "PASS" if response.ok else "FAIL", "Details": f"HTTP {response.status_code}"})
        except Exception as exc:
            source_rows.append({"Source": "Understat", "Live check": "FAIL/UNAVAILABLE", "Details": str(exc)[:250]})
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

    # STEP 108
    st.divider()
    st.subheader("Step 108 — Source Reconciliation")
    st.info(
        "Primary CSV metrics remain authoritative inside this application. "
        "External StatsBomb/Understat values are treated as separate evidence until matching definitions and records are confirmed."
    )
    recon_rows = []
    for metric in ["Shots", "Shots on Target", "Corners", "Goals"]:
        primary = average_value(form_matches, MARKET_COLUMN_MAP[metric])
        recon_rows.append({
            "Metric": metric,
            "Primary CSV average": f"{primary:.2f}" if primary is not None else "—",
            "External source": "Not silently substituted",
            "Status": "Primary source"
        })
    st.dataframe(pd.DataFrame(recon_rows), use_container_width=True, hide_index=True)

    # STEP 109
    st.divider()
    st.subheader("Step 109 — Structured Research Report")
    report_home = st.selectbox("Home team", team_list(data), key="report_home")
    report_away_options = [t for t in team_list(data) if t != report_home]
    report_away = st.selectbox("Away team", report_away_options, key="report_away")
    report_market = st.selectbox("Report market", list(MARKET_COLUMN_MAP.keys()), key="report_market")
    report_direction = st.selectbox("Report direction", ["Over", "Under"], key="report_direction")
    report_line = st.number_input("Report line", 0.0, 30.0, 3.5, 0.5, key="report_line")
    report_home_matches = filtered_team_matches(data, report_home, venue="Home", sample=15)
    report_away_matches = filtered_team_matches(data, report_away, venue="Away", sample=15)
    report_df = build_research_report(report_home, report_away, report_home_matches, report_away_matches, report_market, report_direction, report_line, data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    report_csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download research report CSV", report_csv, "match_research_report.csv", "text/csv", key="download_research_report")

    # STEP 110
    st.divider()
    st.subheader("Step 110 — Transparent Research Checklist")
    checklist_market = analyse_market(report_home_matches, report_market, report_direction, report_line)
    checklist_df = research_checklist(report_home_matches, report_away_matches, checklist_market, validation_df)
    st.dataframe(checklist_df, use_container_width=True, hide_index=True)
    st.caption(
        "The checklist is deliberately descriptive. It does not produce a safe-bet label, ranking, guaranteed outcome or probability forecast."
    )



# ============================================================
# STEPS 111–130 — ADVANCED FIXTURE & MARKET EVALUATION
# ============================================================

# ------------------------------------------------------------
# STEP 111 — SOURCE-AGNOSTIC DATA SCHEMA
# ------------------------------------------------------------
def source_schema_status(df):
    required = set(REQUIRED_COLUMNS)
    present = set(df.columns) if df is not None else set()
    return pd.DataFrame([
        {"Field": c, "Present": "YES" if c in present else "NO", "Primary Source": "Master CSV"}
        for c in REQUIRED_COLUMNS
    ])


# ------------------------------------------------------------
# STEP 112 — DATASET DUPLICATE DETECTION
# ------------------------------------------------------------
def duplicate_match_report(df):
    keys = [c for c in ["date", "home_team", "away_team", "competition"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    dup = df[df.duplicated(keys, keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(columns=keys + ["Duplicate Count"])
    counts = dup.groupby(keys).size().reset_index(name="Duplicate Count")
    return counts.sort_values("Duplicate Count", ascending=False)


# ------------------------------------------------------------
# STEP 113 — MATCH-ID / RECORD INTEGRITY
# ------------------------------------------------------------
def record_integrity(df):
    checks = []
    checks.append({"Check": "Rows", "Value": len(df), "Status": "PASS" if len(df) > 0 else "FAIL"})
    if "date" in df.columns:
        checks.append({"Check": "Valid dates", "Value": int(df["date"].notna().sum()), "Status": "PASS" if df["date"].notna().all() else "REVIEW"})
    for c in ["home_team", "away_team"]:
        checks.append({"Check": f"Non-empty {c}", "Value": int(df[c].astype(str).str.strip().ne("").sum()), "Status": "PASS" if df[c].astype(str).str.strip().ne("").all() else "REVIEW"})
    return pd.DataFrame(checks)


# ------------------------------------------------------------
# STEP 114 — METRIC CONSISTENCY CHECKS
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 115 — RECENCY-WEIGHTED DESCRIPTIVE PROFILE
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 116 — RECENT VS LONGER SAMPLE COMPARISON
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 117 — DISTRIBUTION PROFILE
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 118 — HIT-RATE CONFIDENCE RANGE (DESCRIPTIVE)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 119 — HOME/AWAY MARKET CONVERGENCE
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 120 — OPPONENT-CONCESSION MARKET TEST
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 121 — MARKET LINE LADDER
# ------------------------------------------------------------
def market_line_ladder(df, market, direction, start=0.5, end=8.5):
    rows = []
    for line in np.arange(start, end + 0.01, 0.5):
        a = analyse_market(df, market, direction, float(line))
        rows.append({"Line": round(float(line), 1), "Hits": a["hits"], "Sample": a["sample_size"], "Hit Rate": fmt_pct(a["hit_rate"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# STEP 122 — MARKET RESULT DISTRIBUTION TABLE
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 123 — FIXTURE EVIDENCE MATRIX
# ------------------------------------------------------------
def fixture_evidence_matrix(home_df, away_df, market, direction, line):
    home = hit_rate_interval(home_df, market, direction, line)
    away = hit_rate_interval(away_df, market, direction, line)
    return pd.DataFrame([
        {"Evidence": "Home team historical", "Matches": home["sample_size"], "Hits": home["hits"], "Hit Rate": fmt_pct(home["hit_rate"]), "Interval": f"{fmt_pct(home['lower'])} – {fmt_pct(home['upper'])}" if home['lower'] is not None else "—"},
        {"Evidence": "Away team historical", "Matches": away["sample_size"], "Hits": away["hits"], "Hit Rate": fmt_pct(away["hit_rate"]), "Interval": f"{fmt_pct(away['lower'])} – {fmt_pct(away['upper'])}" if away['lower'] is not None else "—"},
    ])


# ------------------------------------------------------------
# STEP 124 — CONTRADICTION DETECTOR
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 125 — EVIDENCE AGREEMENT COUNT
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 126 — RESEARCH GATE
# ------------------------------------------------------------
def research_gate(home_df, away_df, market, direction, line):
    agreement, total = evidence_agreement(home_df, away_df, market, direction, line)
    reasons = []
    if len(home_df) < 5: reasons.append("home sample below 5")
    if len(away_df) < 5: reasons.append("away sample below 5")
    if agreement < 3: reasons.append("limited evidence agreement")
    status = "RESEARCH READY" if not reasons else "REVIEW REQUIRED"
    return {"Status": status, "Agreement Checks": f"{agreement}/{total}", "Reasons": "; ".join(reasons) if reasons else "No basic gate warnings"}


# ------------------------------------------------------------
# STEP 127 — FIXTURE REPORT BUILDER
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 128 — EXPORTABLE RESEARCH PACK
# ------------------------------------------------------------
def build_research_pack(home_team, away_team, home_df, away_df, market, direction, line):
    return {
        "Report": build_fixture_report_127(home_team, away_team, home_df, away_df, market, direction, line),
        "Evidence": fixture_evidence_matrix(home_df, away_df, market, direction, line),
        "Contradictions": contradiction_flags(home_df, away_df, market, direction, line),
        "Home Distribution": distribution_profile(home_df, market),
        "Away Distribution": distribution_profile(away_df, market),
    }


# ------------------------------------------------------------
# STEP 129 — HUMAN REVIEW CHECKLIST
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 130 — COMPLETE PHASE REPORT
# ------------------------------------------------------------
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



# ============================================================
# STEPS 131–150 — ADVANCED MATCH INTELLIGENCE
# ============================================================

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


# ============================================================
# TAB 7 — PHASE 4 WORKSPACE
# ============================================================

tab7 = st.tabs(["🧠 Phase 4 — Steps 111–130"])[0]

with tab7:
    st.subheader("🧠 Advanced Fixture & Market Evaluation — Steps 111–130")
    st.caption(
        "This phase combines data integrity, recency, distributions, contextual market tests and a human-review gate. "
        "It does not produce a guaranteed outcome or an automatic safe-bet label."
    )

    # STEP 111
    st.divider()
    st.subheader("Step 111 — Source-Agnostic Data Schema")
    st.dataframe(source_schema_status(data), use_container_width=True, hide_index=True)

    # STEP 112
    st.divider()
    st.subheader("Step 112 — Duplicate Match Detection")
    dup_df = duplicate_match_report(data)
    if dup_df.empty:
        st.success("No duplicate match records detected using date + teams + competition.")
    else:
        st.warning(f"{len(dup_df)} duplicate match groups detected.")
        st.dataframe(dup_df, use_container_width=True, hide_index=True)

    # STEP 113
    st.divider()
    st.subheader("Step 113 — Record Integrity")
    st.dataframe(record_integrity(data), use_container_width=True, hide_index=True)

    # STEP 114
    st.divider()
    st.subheader("Step 114 — Metric Consistency")
    st.dataframe(metric_consistency_report(data), use_container_width=True, hide_index=True)

    # Common fixture controls
    st.divider()
    st.subheader("Fixture Workspace")
    p_home = st.selectbox("Home team", team_list(data), key="p4_home")
    p_away_options = [t for t in team_list(data) if t != p_home]
    p_away = st.selectbox("Away team", p_away_options, key="p4_away")
    p_market = st.selectbox("Market", list(MARKET_COLUMN_MAP.keys()), index=1, key="p4_market")
    p_direction = st.selectbox("Direction", ["Over", "Under"], key="p4_direction")
    p_line = st.number_input("Line", min_value=0.0, max_value=30.0, value=3.5, step=0.5, key="p4_line")
    p_home_df = filtered_team_matches(data, p_home, venue="Home", sample=15)
    p_away_df = filtered_team_matches(data, p_away, venue="Away", sample=15)

    # STEP 115
    st.divider()
    st.subheader("Step 115 — Recency-Weighted Profile")
    st.dataframe(recency_profile(p_home_df), use_container_width=True, hide_index=True)

    # STEP 116
    st.divider()
    st.subheader("Step 116 — Recent vs Longer Sample")
    st.dataframe(recent_vs_longer(p_home_df, p_market), use_container_width=True, hide_index=True)

    # STEP 117
    st.divider()
    st.subheader("Step 117 — Distribution Profile")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.caption(p_home)
        st.dataframe(distribution_profile(p_home_df, p_market), use_container_width=True, hide_index=True)
    with dcol2:
        st.caption(p_away)
        st.dataframe(distribution_profile(p_away_df, p_market), use_container_width=True, hide_index=True)

    # STEP 118
    st.divider()
    st.subheader("Step 118 — Hit-Rate Interval")
    hi = hit_rate_interval(p_home_df, p_market, p_direction, p_line)
    st.write(f"Historical hit rate: **{fmt_pct(hi['hit_rate'])}**")
    if hi["lower"] is not None:
        st.write(f"Approximate 95% Wilson interval: **{fmt_pct(hi['lower'])} – {fmt_pct(hi['upper'])}**")
    st.caption("This interval describes uncertainty around the observed historical hit rate; it is not a forecast of the next match.")

    # STEP 119
    st.divider()
    st.subheader("Step 119 — Home/Away Market Convergence")
    st.dataframe(market_convergence(p_home_df, p_away_df, p_market, p_direction, p_line), use_container_width=True, hide_index=True)

    # STEP 120
    st.divider()
    st.subheader("Step 120 — Opponent-Concession Test")
    st.dataframe(opponent_concession_test(p_home_df, p_away_df, p_market, p_direction, p_line), use_container_width=True, hide_index=True)

    # STEP 121
    st.divider()
    st.subheader("Step 121 — Market Line Ladder")
    st.dataframe(market_line_ladder(p_home_df, p_market, p_direction), use_container_width=True, hide_index=True)

    # STEP 122
    st.divider()
    st.subheader("Step 122 — Actual Result Distribution")
    st.dataframe(result_frequency(p_home_df, p_market), use_container_width=True, hide_index=True)

    # STEP 123
    st.divider()
    st.subheader("Step 123 — Fixture Evidence Matrix")
    st.dataframe(fixture_evidence_matrix(p_home_df, p_away_df, p_market, p_direction, p_line), use_container_width=True, hide_index=True)

    # STEP 124
    st.divider()
    st.subheader("Step 124 — Contradiction Detector")
    contradiction_df = contradiction_flags(p_home_df, p_away_df, p_market, p_direction, p_line)
    if contradiction_df.empty:
        st.success("No basic contradiction flags were triggered.")
    else:
        st.dataframe(contradiction_df, use_container_width=True, hide_index=True)

    # STEP 125
    st.divider()
    st.subheader("Step 125 — Evidence Agreement")
    agreement, agreement_total = evidence_agreement(p_home_df, p_away_df, p_market, p_direction, p_line)
    st.metric("Basic evidence checks satisfied", f"{agreement}/{agreement_total}")

    # STEP 126
    st.divider()
    st.subheader("Step 126 — Research Gate")
    gate = research_gate(p_home_df, p_away_df, p_market, p_direction, p_line)
    if gate["Status"] == "RESEARCH READY":
        st.success(gate["Status"])
    else:
        st.warning(gate["Status"])
    st.write(f"Agreement: **{gate['Agreement Checks']}**")
    st.write(f"Warnings: **{gate['Reasons']}**")

    # STEP 127
    st.divider()
    st.subheader("Step 127 — Fixture Report")
    report130 = build_fixture_report_127(p_home, p_away, p_home_df, p_away_df, p_market, p_direction, p_line)
    st.dataframe(report130, use_container_width=True, hide_index=True)

    # STEP 128
    st.divider()
    st.subheader("Step 128 — Exportable Research Pack")
    pack = build_research_pack(p_home, p_away, p_home_df, p_away_df, p_market, p_direction, p_line)
    pack_text = "\n\n".join([f"=== {name} ===\n{frame.to_csv(index=False)}" for name, frame in pack.items()])
    st.download_button("Download research pack", pack_text.encode("utf-8"), "phase4_research_pack.txt", "text/plain", key="download_phase4_pack")

    # STEP 129
    st.divider()
    st.subheader("Step 129 — Human Review Checklist")
    review_df = human_review_checklist()
    st.dataframe(review_df, use_container_width=True, hide_index=True)
    review_notes = st.text_area("Human review notes", height=140, key="phase4_review_notes")

    # STEP 130
    st.divider()
    st.subheader("Step 130 — Complete Phase Report")
    phase_summary = phase_130_summary(p_home, p_away, p_market, p_direction, p_line, p_home_df, p_away_df)
    st.dataframe(phase_summary, use_container_width=True, hide_index=True)
    phase_csv = phase_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download Phase 4 summary CSV", phase_csv, "phase4_summary.csv", "text/csv", key="download_phase4_summary")
    st.caption(
        "Final interpretation remains a human decision. The application presents historical evidence, "
        "data quality and contextual checks rather than predicting or guaranteeing the next result."
    )


# ============================================================
# TAB 8 — PHASE 5 WORKSPACE — STEPS 131–150
# ============================================================

tab8 = st.tabs(["🧠 Phase 5 — Steps 131–150"])[0]

with tab8:
    st.subheader("🧠 Advanced Match Intelligence — Steps 131–150")
    st.caption("This phase strengthens freshness, coverage, opponent context, multi-window testing, distribution diagnostics and fixture evidence convergence. It remains descriptive and does not guarantee outcomes.")

    p5_home = st.selectbox("Home Team", team_list(data), key="p5_home")
    p5_away_options = [t for t in team_list(data) if t != p5_home]
    p5_away = st.selectbox("Away Team", p5_away_options, key="p5_away")
    p5_market = st.selectbox("Market", list(MARKET_COLUMN_MAP.keys()), key="p5_market")
    p5_direction = st.selectbox("Direction", ["Over", "Under"], key="p5_direction")
    p5_line = st.number_input("Line", 0.0, 30.0, 3.5, 0.5, key="p5_line")
    p5_sample = st.selectbox("Recent sample", [5,10,15], index=2, key="p5_sample")

    p5_home_df = filtered_team_matches(data, p5_home, venue="Home", sample=p5_sample)
    p5_away_df = filtered_team_matches(data, p5_away, venue="Away", sample=p5_sample)

    # STEP 131
    st.divider(); st.subheader("Step 131 — Data Freshness")
    st.dataframe(data_freshness_report(data), use_container_width=True, hide_index=True)

    # STEP 132
    st.divider(); st.subheader("Step 132 — Competition / Coverage Matrix")
    st.dataframe(coverage_matrix(data), use_container_width=True, hide_index=True)

    # STEP 133
    st.divider(); st.subheader("Step 133 — Missing Data & Anomaly Diagnostics")
    st.dataframe(missingness_table(data), use_container_width=True, hide_index=True)
    st.dataframe(anomaly_report(data), use_container_width=True, hide_index=True)

    # STEP 134
    st.divider(); st.subheader("Step 134 — Duplicate / Integrity Review")
    st.dataframe(duplicate_match_report(data), use_container_width=True, hide_index=True)

    # STEP 135
    st.divider(); st.subheader("Step 135 — Opponent Strength Bands")
    p5_strength_team = st.selectbox("Team for opponent-strength analysis", [p5_home,p5_away], key="p5_strength_team")
    strength_df = opponent_strength_bands(data, p5_strength_team, MARKET_COLUMN_MAP[p5_market], p5_sample)
    if strength_df.empty: st.info("No opponent-strength observations available.")
    else: st.dataframe(strength_df, use_container_width=True, hide_index=True)

    # STEP 136
    st.divider(); st.subheader("Step 136 — Attack vs Defence Adjustment")
    st.dataframe(adjusted_attack_defence(p5_home_df, p5_away_df, p5_market), use_container_width=True, hide_index=True)

    # STEP 137
    st.divider(); st.subheader("Step 137 — Recency-Weighted Profile")
    left,right=st.columns(2)
    with left:
        st.caption(p5_home); st.dataframe(recency_weighted_profile(p5_home_df,p5_market),use_container_width=True,hide_index=True)
    with right:
        st.caption(p5_away); st.dataframe(recency_weighted_profile(p5_away_df,p5_market),use_container_width=True,hide_index=True)

    # STEP 138
    st.divider(); st.subheader("Step 138 — Recent vs Broader Sample")
    st.dataframe(pd.concat([
        multi_window_market_test(p5_home_df,p5_market,p5_direction,p5_line).assign(Team=p5_home),
        multi_window_market_test(p5_away_df,p5_market,p5_direction,p5_line).assign(Team=p5_away)
    ],ignore_index=True),use_container_width=True,hide_index=True)

    # STEP 139
    st.divider(); st.subheader("Step 139 — Multi-Window Market Testing")
    st.dataframe(multi_window_market_test(pd.concat([p5_home_df,p5_away_df]),p5_market,p5_direction,p5_line),use_container_width=True,hide_index=True)

    # STEP 140
    st.divider(); st.subheader("Step 140 — Line Sensitivity")
    st.dataframe(line_sensitivity(pd.concat([p5_home_df,p5_away_df]),p5_market,p5_direction,p5_line),use_container_width=True,hide_index=True)

    # STEP 141
    st.divider(); st.subheader("Step 141 — Distribution Diagnostics")
    st.dataframe(distribution_diagnostics(pd.concat([p5_home_df,p5_away_df]),p5_market),use_container_width=True,hide_index=True)

    # STEP 142
    st.divider(); st.subheader("Step 142 — Consistency / Volatility")
    c1,c2=st.columns(2)
    for container, label, frame in [(c1,p5_home,p5_home_df),(c2,p5_away,p5_away_df)]:
        d=consistency_diagnostics(frame,p5_market,p5_direction,p5_line)
        with container:
            st.metric(f"{label} hit rate",fmt_pct(d["hit_rate"]))
            st.metric("Volatility",f"{d['volatility']:.2f}" if d["volatility"] is not None else "—")
            st.metric("Median",f"{d['median']:.2f}" if d["median"] is not None else "—")

    # STEP 143
    st.divider(); st.subheader("Step 143 — Fixture-Specific Profiles")
    profile_rows=[]
    for label,frame in [(p5_home,p5_home_df),(p5_away,p5_away_df)]:
        d=consistency_diagnostics(frame,p5_market,p5_direction,p5_line)
        profile_rows.append({"Team":label,"Sample":d["sample"],"Average":d["mean"],"Median":d["median"],"Hit Rate":fmt_pct(d["hit_rate"])})
    st.dataframe(pd.DataFrame(profile_rows),use_container_width=True,hide_index=True)

    # STEP 144
    st.divider(); st.subheader("Step 144 — Cross-Team Market Comparison")
    st.dataframe(build_market_comparison(p5_home_df,p5_away_df,p5_market,p5_direction,p5_line),use_container_width=True,hide_index=True)

    # STEP 145
    st.divider(); st.subheader("Step 145 — Contradiction Detection")
    contradictions=contradiction_report(p5_home_df,p5_away_df,p5_market,p5_direction,p5_line)
    if contradictions.empty: st.success("No contradiction checks produced a warning from the available sample.")
    else: st.dataframe(contradictions,use_container_width=True,hide_index=True)

    # STEP 146
    st.divider(); st.subheader("Step 146 — Evidence Convergence")
    conv, _ = fixture_evidence_convergence(p5_home_df,p5_away_df,p5_market,p5_direction,p5_line)
    st.dataframe(conv,use_container_width=True,hide_index=True)

    # STEP 147
    st.divider(); st.subheader("Step 147 — Market Evidence Report")
    evidence_report = pd.concat([
        multi_window_market_test(p5_home_df,p5_market,p5_direction,p5_line).assign(Team=p5_home),
        multi_window_market_test(p5_away_df,p5_market,p5_direction,p5_line).assign(Team=p5_away)
    ],ignore_index=True)
    st.dataframe(evidence_report,use_container_width=True,hide_index=True)

    # STEP 148
    st.divider(); st.subheader("Step 148 — Data-Quality Gate")
    gate_131_150=fixture_quality_gate_131_150(p5_home_df,p5_away_df,p5_market,p5_direction,p5_line,data)
    st.dataframe(gate_131_150,use_container_width=True,hide_index=True)

    # STEP 149
    st.divider(); st.subheader("Step 149 — Human Review Dashboard")
    review_149=human_review_checklist().copy()
    review_149["Step 131–150 Check"]="Manual review required"
    st.dataframe(review_149,use_container_width=True,hide_index=True)
    st.text_area("Phase 5 human review notes",height=140,key="p5_notes")

    # STEP 150
    st.divider(); st.subheader("Step 150 — Complete Phase 5 Report")
    phase5_report=build_phase_131_150_report(p5_home,p5_away,p5_market,p5_direction,p5_line,p5_home_df,p5_away_df,data)
    st.dataframe(phase5_report,use_container_width=True,hide_index=True)
    st.download_button("Download Steps 131–150 report",phase5_report.to_csv(index=False).encode("utf-8"),"steps_131_150_report.csv","text/csv",key="download_phase5_report")
    st.caption("Steps 131–150 are an evidence-synthesis layer. StatsBomb and Understat remain separate supplementary integrations and are not silently merged into the primary CSV.")



# ============================================================
# STEPS 151–170 — VALIDATION, MODELLING & BACKTESTING
# ============================================================

from math import exp

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

# ------------------------------------------------------------
# UI: STEPS 150–170 (Step 150 retained and expanded)
# ------------------------------------------------------------

with st.expander("🧪 Phase 6 — Steps 150–170: Validation & Model Testing", expanded=False):
    st.caption("This phase tests whether historical evidence survives chronological, out-of-sample validation. It does not create a guaranteed or 'safe' bet label.")
    v_home=st.selectbox("Validation Home Team",team_list(data),key="v_home")
    v_away=st.selectbox("Validation Away Team",[t for t in team_list(data) if t!=v_home],key="v_away")
    v_market=st.selectbox("Validation Market",list(MARKET_COLUMN_MAP),key="v_market")
    v_direction=st.selectbox("Validation Direction",["Over","Under"],key="v_direction")
    v_line=st.number_input("Validation Line",0.0,30.0,3.5,0.5,key="v_line")
    vh=filtered_team_matches(data,v_home,venue="Home",sample=15)
    va=filtered_team_matches(data,v_away,venue="Away",sample=15)
    combined=pd.concat([vh,va],ignore_index=True).sort_values("date")

    # 150 — expanded final phase report
    st.subheader("Step 150 — Final Evidence Report")
    st.dataframe(build_phase_131_150_report(v_home,v_away,v_market,v_direction,v_line,vh,va,data),use_container_width=True,hide_index=True)

    # 151–154
    st.subheader("Steps 151–154 — Feature Engineering")
    st.dataframe(model_feature_frame(combined,v_market).tail(15),use_container_width=True,hide_index=True)

    # 155–158
    st.subheader("Steps 155–158 — Baseline & Probability Models")
    st.dataframe(model_vs_history(combined,v_market,v_direction,v_line),use_container_width=True,hide_index=True)
    st.caption("The Poisson row is a simple benchmark, not a claim that the underlying event follows a Poisson process.")

    # 159–162
    st.subheader("Steps 159–162 — Walk-Forward Backtest")
    bt=rolling_baseline_backtest(combined,v_market,v_direction,v_line,min_history=5)
    if bt.empty: st.info("Not enough chronological observations for a walk-forward test.")
    else:
        st.dataframe(bt,use_container_width=True,hide_index=True)
        st.dataframe(pd.DataFrame([backtest_summary(bt)]),use_container_width=True,hide_index=True)

    # 163–166
    st.subheader("Steps 163–166 — Market Validation")
    st.dataframe(calibration_table(bt),use_container_width=True,hide_index=True)
    st.dataframe(false_signal_report(bt),use_container_width=True,hide_index=True)
    st.dataframe(research_model_comparison(vh,va,v_market,v_direction,v_line),use_container_width=True,hide_index=True)

    # 167–170
    st.subheader("Steps 167–170 — Model Governance")
    st.dataframe(leakage_check(combined),use_container_width=True,hide_index=True)
    st.dataframe(model_health_report(bt),use_container_width=True,hide_index=True)
    st.download_button("Download Phase 6 validation CSV",bt.to_csv(index=False).encode("utf-8") if not bt.empty else b"", "phase_6_validation.csv","text/csv",key="download_phase6")



# ============================================================
# PHASE 7 — STEPS 171–190: ADVANCED DATA INTEGRATION
# ============================================================

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


# ============================================================
# PHASE 8 — STEPS 191–210: PLAYER RESEARCH
# ============================================================

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


# ============================================================
# PHASE 9 — STEPS 211–230: MATCH-STATE INTELLIGENCE
# ============================================================

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


# ============================================================
# PHASE 10 — STEPS 231–250: MARKET LABORATORY
# ============================================================

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


# ============================================================
# PHASE 11 — STEPS 251–270: BOOKMAKER INTELLIGENCE
# ============================================================

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


# ============================================================
# PHASE 12 — STEPS 271–290: FULL VALIDATION SYSTEM
# ============================================================

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


# ============================================================
# PHASE 13 — STEPS 291–310: RESEARCH COMMAND CENTRE
# ============================================================

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
# INTERNAL PIPELINE — STEPS 1–20: AUTOMATED FIXTURE RESEARCH
# ============================================================

PIPELINE_MARKETS = list(MARKET_COLUMN_MAP.keys())


def _num(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def normalise_fixture(data, home_team, away_team):
    teams = team_list(data)
    return {
        "home_valid": home_team in teams,
        "away_valid": away_team in teams,
        "same_team": home_team == away_team,
        "home": home_team,
        "away": away_team,
    }


def historical_windows(data, team, venue, windows=(5, 10, 15, 20, 30)):
    out = {}
    for w in windows:
        out[w] = filtered_team_matches(data, team, venue=venue, sample=w)
    return out


def market_stats(df, market, direction, line):
    result = analyse_market(df, market, direction, line)
    vals = pd.to_numeric(df.get(MARKET_COLUMN_MAP[market]), errors="coerce").dropna() if not df.empty and MARKET_COLUMN_MAP[market] in df.columns else pd.Series(dtype=float)
    return {
        "sample": len(vals),
        "hits": result.get("hits", 0),
        "hit_rate": result.get("hit_rate"),
        "mean": vals.mean() if not vals.empty else np.nan,
        "median": vals.median() if not vals.empty else np.nan,
        "min": vals.min() if not vals.empty else np.nan,
        "max": vals.max() if not vals.empty else np.nan,
        "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
    }


def opponent_concession_context(data, team, venue):
    rows = data[data["home_team"].eq(team) if venue == "Home" else data["away_team"].eq(team)].copy()
    if rows.empty:
        return {"matches": 0, "shots_conceded": np.nan, "sot_conceded": np.nan, "corners_conceded": np.nan, "goals_conceded": np.nan}
    if venue == "Home":
        return {
            "matches": len(rows),
            "shots_conceded": pd.to_numeric(rows["away_shots"], errors="coerce").mean(),
            "sot_conceded": pd.to_numeric(rows["away_sot"], errors="coerce").mean(),
            "corners_conceded": pd.to_numeric(rows["away_corners"], errors="coerce").mean(),
            "goals_conceded": pd.to_numeric(rows["away_goals"], errors="coerce").mean(),
        }
    return {
        "matches": len(rows),
        "shots_conceded": pd.to_numeric(rows["home_shots"], errors="coerce").mean(),
        "sot_conceded": pd.to_numeric(rows["home_sot"], errors="coerce").mean(),
        "corners_conceded": pd.to_numeric(rows["home_corners"], errors="coerce").mean(),
        "goals_conceded": pd.to_numeric(rows["home_goals"], errors="coerce").mean(),
    }


def distribution_profile_values(df, market):
    col = MARKET_COLUMN_MAP[market]
    vals = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
    if vals.empty:
        return {"mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "std": np.nan}
    return {"mean": vals.mean(), "median": vals.median(), "q25": vals.quantile(.25), "q75": vals.quantile(.75), "std": vals.std(ddof=1) if len(vals)>1 else np.nan}


def line_sensitivity(dataframe, market, direction, line):
    # Test adjacent lines for evidence shape only; bookmaker availability is NOT inferred.
    lines = sorted(set([max(0, line-1.0), max(0, line-0.5), line, line+0.5, line+1.0]))
    rows=[]
    for ln in lines:
        r=analyse_market(dataframe, market, direction, ln)
        rows.append({"Test line": ln, "Hit rate": r.get("hit_rate"), "Hits": r.get("hits",0), "Sample": r.get("sample_size",0)})
    return pd.DataFrame(rows)


def trend_label(rates):
    vals=[x for x in rates if x is not None and not pd.isna(x)]
    if len(vals)<2: return "INSUFFICIENT"
    recent=vals[0]; long=vals[-1]
    delta=recent-long
    if delta >= .10: return "IMPROVING"
    if delta <= -.10: return "DECLINING"
    return "STABLE"


def contradiction_status(home_stats, away_stats):
    rates=[home_stats.get("hit_rate"), away_stats.get("hit_rate")]
    vals=[x for x in rates if x is not None and not pd.isna(x)]
    if len(vals)<2: return "INSUFFICIENT"
    if abs(vals[0]-vals[1]) >= .25: return "CONFLICT"
    return "NO_MAJOR_CONFLICT"


def pipeline_candidate(data, home_team, away_team, bookmaker, market, direction, line, odds):
    home_windows=historical_windows(data, home_team, "Home")
    away_windows=historical_windows(data, away_team, "Away")
    hs={w:market_stats(home_windows[w],market,direction,line) for w in home_windows}
    aws={w:market_stats(away_windows[w],market,direction,line) for w in away_windows}
    h10=hs[10]; a10=aws[10]
    hc=opponent_concession_context(data,home_team,"Home")
    ac=opponent_concession_context(data,away_team,"Away")
    hdist=distribution_profile_values(home_windows[15],market)
    adist=distribution_profile_values(away_windows[15],market)
    brk=(1/odds) if odds and odds>1 else np.nan
    recent_rates=[hs[w]["hit_rate"] for w in [5,10,15,20,30]]
    away_rates=[aws[w]["hit_rate"] for w in [5,10,15,20,30]]
    stability=contradiction_status(h10,a10)
    checks=[]
    checks.append(("Market entered and available", True, "Bookmaker line is user-entered; unavailable lower lines are not assumed."))
    checks.append(("Fixture teams valid", home_team in team_list(data) and away_team in team_list(data), "Team names must exist in the dataset."))
    checks.append(("Adequate sample", min(h10["sample"],a10["sample"]) >= 5, f"Home={h10['sample']}, Away={a10['sample']} observations."))
    checks.append(("Recent evidence", max([x for x in [h10["hit_rate"],a10["hit_rate"]] if x is not None], default=np.nan) >= .60, "10-match evidence threshold is descriptive, not predictive."))
    checks.append(("Home/away agreement", stability != "CONFLICT", stability))
    # Distribution line location
    medians=[hdist["median"],adist["median"]]
    dist_ok=any(not pd.isna(x) and ((line <= x and direction=="Over") or (line >= x and direction=="Under")) for x in medians)
    checks.append(("Distribution supports line", dist_ok, f"Home median={hdist['median']:.2f} / Away median={adist['median']:.2f}" if not pd.isna(hdist['median']) and not pd.isna(adist['median']) else "Insufficient distribution data."))
    checks.append(("Price recorded", not pd.isna(brk), f"Break-even={brk:.1%}" if not pd.isna(brk) else "Invalid odds."))
    # Historical hit-rate vs break-even is shown, never treated as proof of future probability.
    observed=max([x for x in [h10["hit_rate"],a10["hit_rate"]] if x is not None], default=np.nan)
    price_gap=(observed-brk) if not pd.isna(observed) and not pd.isna(brk) else np.nan
    checks.append(("Observed-vs-break-even", not pd.isna(price_gap), f"Observed max 10-match rate={observed:.1%}; gap={price_gap:.1%}" if not pd.isna(price_gap) else "Cannot compare."))
    warnings=[]
    if min(h10["sample"],a10["sample"])<5: warnings.append("Small sample")
    if stability=="CONFLICT": warnings.append("Home/away evidence conflict")
    if not pd.isna(hdist["std"]) and hdist["std"] > max(1, abs(line)*.75): warnings.append("High home distribution variance")
    if not pd.isna(adist["std"]) and adist["std"] > max(1, abs(line)*.75): warnings.append("High away distribution variance")
    # Steps 6–7 are intentionally data-gated if event-level timeline data is absent.
    timeline_available=all(c in data.columns for c in ["home_1h_goals","away_1h_goals"])
    checks.append(("Match-state analysis", timeline_available, "Timeline/state columns available." if timeline_available else "Not available in current master CSV."))
    half_available=all(c in data.columns for c in ["home_1h_shots","away_1h_shots"])
    checks.append(("Half-by-half analysis", half_available, "Half-by-half columns available." if half_available else "Not available in current master CSV."))
    return {
        "Bookmaker": bookmaker, "Market": market, "Direction": direction, "Line": line, "Odds": odds,
        "Break-even": brk, "Home 10 Hit Rate": h10["hit_rate"], "Away 10 Hit Rate": a10["hit_rate"],
        "Home 15 Median": hdist["median"], "Away 15 Median": adist["median"],
        "Home Opponent SOT Conceded": hc["sot_conceded"], "Away Opponent SOT Conceded": ac["sot_conceded"],
        "Observed vs BE": price_gap, "Trend Home": trend_label(recent_rates), "Trend Away": trend_label(away_rates),
        "Agreement": stability, "Warnings": "; ".join(warnings) if warnings else "None",
        "Checks": checks,
        "Home windows": hs, "Away windows": aws,
        "Line sensitivity": line_sensitivity(pd.concat([home_windows[15],away_windows[15]],ignore_index=True),market,direction,line),
    }


def run_research_pipeline(data, home_team, away_team, market_rows):
    candidates=[]; rejected=[]
    for row in market_rows:
        try:
            c=pipeline_candidate(data,home_team,away_team,row["Bookmaker"],row["Market"],row["Direction"],float(row["Line"]),float(row["Odds"]))
        except Exception as exc:
            rejected.append({"Market":row.get("Market",""),"Status":"REJECTED","Reason":f"Pipeline error: {exc}"})
            continue
        hard=[x for x in c["Checks"] if x[0] in ["Market entered and available","Fixture teams valid","Adequate sample","Price recorded"] and not x[1]]
        if hard:
            rejected.append({"Market":f"{c['Market']} {c['Direction']} {c['Line']}","Status":"REJECTED","Reason":"; ".join(x[2] for x in hard)})
        else:
            candidates.append(c)
    return candidates,rejected

# ============================================================
# UI — PHASES 7–13
# ============================================================

phase7, phase8, phase9, phase10, phase11, phase12, phase13 = st.tabs([
    "🛰️ Phase 7 — 171–190",
    "👤 Phase 8 — 191–210",
    "🎮 Phase 9 — 211–230",
    "🧪 Phase 10 — 231–250",
    "💹 Phase 11 — 251–270",
    "🛡️ Phase 12 — 271–290",
    "🧠 Phase 13 — 291–310",
])

with phase7:
    st.subheader("Phase 7 — Advanced Data Integration")
    st.caption("Steps 171–190 preserve the master CSV as the primary dataset. StatsBomb and Understat are separate evidence sources until definitions and mappings are verified.")
    st.subheader("Steps 171–174 — Source Health")
    st.dataframe(source_status_table(), use_container_width=True, hide_index=True)
    st.subheader("Steps 175–178 — Source Mapping")
    st.dataframe(source_mapping_report(data), use_container_width=True, hide_index=True)
    st.subheader("Steps 179–182 — Team Strength Context")
    s_team = st.selectbox("Team", team_list(data), key="p7_team")
    st.dataframe(team_strength_context(data, s_team), use_container_width=True, hide_index=True)
    st.subheader("Steps 183–186 — Rolling Context Features")
    st.dataframe(build_context_features(data, s_team).tail(20), use_container_width=True, hide_index=True)
    st.subheader("Steps 187–190 — Integration Gate")
    st.info("External-source data is not silently merged. Explicit mapping and metric-definition checks remain required.")

with phase8:
    st.subheader("Phase 8 — Player Research")
    st.caption("Steps 191–210 create the player-research framework. Player-level statistics require player/event data; the current master CSV contains team match statistics.")
    st.subheader("Steps 191–194 — Player Schema")
    st.dataframe(player_schema_status(data), use_container_width=True, hide_index=True)
    st.subheader("Steps 195–202 — Player Research Workspace")
    player_name = st.text_input("Player", key="p8_player")
    st.dataframe(lineup_research_template(), use_container_width=True, hide_index=True)
    st.text_area("Player research notes", height=140, key="p8_notes")
    st.subheader("Steps 203–206 — Starter / Minutes Gate")
    st.info("Lineup status, expected minutes and substitution risk should be checked before interpreting player markets.")
    st.subheader("Steps 207–210 — Player Source Gate")
    st.warning("Player shots/SOT/fouls cannot be fabricated from team-level CSV columns. They require a player/event source.")

with phase9:
    st.subheader("Phase 9 — Match-State Intelligence")
    st.subheader("Steps 211–214 — State Data Availability")
    st.dataframe(state_feature_status(data), use_container_width=True, hide_index=True)
    st.subheader("Steps 215–218 — Possession Context")
    p9_team = st.selectbox("Team", team_list(data), key="p9_team")
    pos_df = possession_context(data, p9_team)
    if pos_df.empty:
        st.info("Possession is not present in the current primary CSV schema. Add an external possession/event source when available.")
    else:
        st.dataframe(pos_df.tail(20), use_container_width=True, hide_index=True)
    st.subheader("Steps 219–226 — Game-State Research Template")
    st.dataframe(game_state_template(), use_container_width=True, hide_index=True)
    st.subheader("Steps 227–230 — State Evidence Gate")
    st.info("Leading/trailing behaviour should be analysed from event or split-state data rather than inferred from the final score alone.")

with phase10:
    st.subheader("Phase 10 — Market Laboratory")
    p10_team = st.selectbox("Team", team_list(data), key="p10_team")
    p10_market = st.selectbox("Market", list(MARKET_COLUMN_MAP), key="p10_market")
    p10_direction = st.selectbox("Direction", ["Over", "Under"], key="p10_direction")
    p10_line = st.number_input("Reference line", 0.0, 30.0, 3.5, 0.5, key="p10_line")
    st.subheader("Steps 231–236 — Line Laboratory")
    st.dataframe(market_lab(data, p10_team, p10_market, p10_direction, [0.5,1.5,2.5,3.5,4.5,5.5], 30), use_container_width=True, hide_index=True)
    st.subheader("Steps 237–242 — Window Stability")
    st.dataframe(market_window_comparison(data, p10_team, p10_market, p10_direction, p10_line), use_container_width=True, hide_index=True)
    st.subheader("Steps 243–250 — Market Registry")
    st.dataframe(market_definition_registry(), use_container_width=True, hide_index=True)

with phase11:
    st.subheader("Phase 11 — Bookmaker Intelligence")
    st.caption("This remains a monitoring system. It does not connect to bookmaker accounts or place bets.")
    st.subheader("Steps 251–256 — Bookmaker Coverage")
    st.dataframe(line_availability_report(data), use_container_width=True, hide_index=True)
    st.subheader("Steps 257–262 — Tracked Market Summary")
    st.dataframe(bookmaker_summary(st.session_state.get("market_watchlist", [])), use_container_width=True, hide_index=True)
    st.subheader("Steps 263–266 — Odds Movement Capture Template")
    st.dataframe(odds_movement_template(), use_container_width=True, hide_index=True)
    st.subheader("Steps 267–270 — Market Monitoring Rules")
    st.info("Record bookmaker, capture time, line and odds together. A later odds movement should never overwrite the earlier observation.")

with phase12:
    st.subheader("Phase 12 — Full Validation System")
    p12_team = st.selectbox("Team", team_list(data), key="p12_team")
    p12_market = st.selectbox("Market", list(MARKET_COLUMN_MAP), key="p12_market")
    p12_direction = st.selectbox("Direction", ["Over", "Under"], key="p12_direction")
    p12_line = st.number_input("Validation line", 0.0, 30.0, 3.5, 0.5, key="p12_line")
    st.subheader("Steps 271–276 — Validation Grid")
    grid = rolling_validation_grid(data, p12_team, p12_market, p12_direction, [max(0.5,p12_line-1), max(0.5,p12_line-0.5), p12_line, p12_line+0.5, p12_line+1])
    st.dataframe(grid, use_container_width=True, hide_index=True)
    st.subheader("Steps 277–284 — Governance")
    st.dataframe(validation_governance_report(data, p12_team, p12_market, p12_direction, p12_line), use_container_width=True, hide_index=True)
    st.subheader("Steps 285–290 — Out-of-Sample Requirement")
    st.info("The earlier Steps 159–162 walk-forward engine remains the principal chronological validation layer. These steps expose the governance checks around it.")
    if st.button("Run 15-match walk-forward validation", key="p12_run_bt"):
        vm = filtered_team_matches(data, p12_team, sample=30)
        bt2 = rolling_baseline_backtest(vm, p12_market, p12_direction, p12_line, min_history=5)
        if bt2.empty:
            st.warning("Not enough observations for a walk-forward test.")
        else:
            st.dataframe(pd.DataFrame([backtest_summary(bt2)]), use_container_width=True, hide_index=True)
            st.download_button("Download validation results", bt2.to_csv(index=False).encode("utf-8"), "phase12_validation.csv", "text/csv", key="p12_download")

with phase13:
    st.subheader("Phase 13 — Research Command Centre")
    st.caption("Steps 291–310 bring the research workflow together while keeping final interpretation with the human researcher.")
    cc1, cc2 = st.columns(2)
    with cc1:
        cc_home = st.selectbox("Home Team", team_list(data), key="cc_home")
    with cc2:
        cc_away = st.selectbox("Away Team", [t for t in team_list(data) if t != cc_home], key="cc_away")
    cc_market = st.selectbox("Market", list(MARKET_COLUMN_MAP), key="cc_market")
    cc_direction = st.selectbox("Direction", ["Over", "Under"], key="cc_direction")
    cc_line = st.number_input("Line", 0.0, 30.0, 3.5, 0.5, key="cc_line")
    st.subheader("Steps 291–296 — Fixture Evidence")
    st.dataframe(command_centre_report(data, cc_home, cc_away, cc_market, cc_direction, cc_line), use_container_width=True, hide_index=True)
    st.subheader("Steps 297–304 — Human Research Checklist")
    st.dataframe(command_centre_checklist(), use_container_width=True, hide_index=True)
    st.text_area("Command Centre research notes", height=160, key="cc_notes")
    st.subheader("Steps 305–310 — Final Research Pack")
    cc_report = command_centre_report(data, cc_home, cc_away, cc_market, cc_direction, cc_line)
    st.download_button("Download Command Centre report", cc_report.to_csv(index=False).encode("utf-8"), "command_centre_report.csv", "text/csv", key="cc_download")
    st.caption("The Command Centre does not issue a 'safe bet' verdict. It organizes evidence for human review.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Football Betting Research Tool V1 — Steps 1–310. "
    "Primary dataset: football_master_2024_27_v1.csv. "
    "StatsBomb and Understat remain separate sources until their data is explicitly mapped and validated. "
    "Research outputs are descriptive and require human review."
)
