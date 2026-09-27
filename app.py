
import streamlit as st
import pandas as pd
import numpy as np
import os

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
# TAB 5 — MATCH RESEARCH
# ============================================================

with tab5:

    st.subheader("🔬 Match Research")

    st.caption(
        "Research a specific fixture by comparing both teams "
        "using historical match-by-match data."
    )

    # --------------------------------------------------------
    # FIXTURE SELECTION
    # --------------------------------------------------------

    teams = team_list(data)

    r1, r2 = st.columns(2)

    with r1:

        home_team = st.selectbox(
            "Home Team",
            teams,
            key="research_home_team"
        )

    with r2:

        away_team_options = [
            team
            for team in teams
            if team != home_team
        ]

        away_team = st.selectbox(
            "Away Team",
            away_team_options,
            key="research_away_team"
        )

    # --------------------------------------------------------
    # MATCH CONTEXT
    # --------------------------------------------------------

    st.divider()
    st.subheader("Match Context")

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

    # --------------------------------------------------------
    # FILTER BOTH TEAMS
    # --------------------------------------------------------

    home_matches = filtered_team_matches(
        data=data,
        team=home_team,
        season=research_season,
        competition=research_competition,
        venue="Home",
        sample=research_sample
    )

    away_matches = filtered_team_matches(
        data=data,
        team=away_team,
        season=research_season,
        competition=research_competition,
        venue="Away",
        sample=research_sample
    )

    home_all = filtered_team_matches(
        data=data,
        team=home_team,
        season=research_season,
        competition=research_competition,
        venue="All",
        sample=research_sample
    )

    away_all = filtered_team_matches(
        data=data,
        team=away_team,
        season=research_season,
        competition=research_competition,
        venue="All",
        sample=research_sample
    )

    # --------------------------------------------------------
    # SAMPLE WARNINGS
    # --------------------------------------------------------

    if len(home_matches) < 5:

        st.warning(
            f"{home_team}: only {len(home_matches)} "
            "home matches are available for these filters."
        )

    if len(away_matches) < 5:

        st.warning(
            f"{away_team}: only {len(away_matches)} "
            "away matches are available for these filters."
        )

    # --------------------------------------------------------
    # TEAM COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.subheader("Home vs Away Team Comparison")

    home_summary = match_team_summary(
        home_matches
    )

    away_summary = match_team_summary(
        away_matches
    )

    comparison_rows = [
        {
            "Metric": "Matches",
            "Home Team": home_summary["Matches"],
            "Away Team": away_summary["Matches"]
        },
        {
            "Metric": "Avg Shots",
            "Home Team": (
                f"{home_summary['Shots']:.2f}"
                if home_summary["Shots"] is not None
                else "—"
            ),
            "Away Team": (
                f"{away_summary['Shots']:.2f}"
                if away_summary["Shots"] is not None
                else "—"
            )
        },
        {
            "Metric": "Avg Shots on Target",
            "Home Team": (
                f"{home_summary['SOT']:.2f}"
                if home_summary["SOT"] is not None
                else "—"
            ),
            "Away Team": (
                f"{away_summary['SOT']:.2f}"
                if away_summary["SOT"] is not None
                else "—"
            )
        },
        {
            "Metric": "Avg Corners",
            "Home Team": (
                f"{home_summary['Corners']:.2f}"
                if home_summary["Corners"] is not None
                else "—"
            ),
            "Away Team": (
                f"{away_summary['Corners']:.2f}"
                if away_summary["Corners"] is not None
                else "—"
            )
        },
        {
            "Metric": "Avg Goals",
            "Home Team": (
                f"{home_summary['Goals']:.2f}"
                if home_summary["Goals"] is not None
                else "—"
            ),
            "Away Team": (
                f"{away_summary['Goals']:.2f}"
                if away_summary["Goals"] is not None
                else "—"
            )
        }
    ]

    st.dataframe(
        pd.DataFrame(comparison_rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ATTACK VS DEFENCE
    # --------------------------------------------------------

    st.divider()
    st.subheader("Attack vs Defence")

    st.caption(
        "Home attacking production is compared with away "
        "defensive concession, and vice versa."
    )

    home_attack = {
        "Shots": average_value(
            home_matches,
            "team_shots"
        ),
        "Shots on Target": average_value(
            home_matches,
            "team_sot"
        ),
        "Corners": average_value(
            home_matches,
            "team_corners"
        ),
        "Goals": average_value(
            home_matches,
            "team_goals"
        )
    }

    away_defence = {
        "Shots": average_value(
            away_matches,
            "opp_shots"
        ),
        "Shots on Target": average_value(
            away_matches,
            "opp_sot"
        ),
        "Corners": average_value(
            away_matches,
            "opp_corners"
        ),
        "Goals": average_value(
            away_matches,
            "opp_goals"
        )
    }

    away_attack = {
        "Shots": average_value(
            away_matches,
            "team_shots"
        ),
        "Shots on Target": average_value(
            away_matches,
            "team_sot"
        ),
        "Corners": average_value(
            away_matches,
            "team_corners"
        ),
        "Goals": average_value(
            away_matches,
            "team_goals"
        )
    }

    home_defence = {
        "Shots": average_value(
            home_matches,
            "opp_shots"
        ),
        "Shots on Target": average_value(
            home_matches,
            "opp_sot"
        ),
        "Corners": average_value(
            home_matches,
            "opp_corners"
        ),
        "Goals": average_value(
            home_matches,
            "opp_goals"
        )
    }

    attack_defence_rows = []

    for metric in [
        "Shots",
        "Shots on Target",
        "Corners",
        "Goals"
    ]:

        attack_defence_rows.append(
            {
                "Metric": metric,
                "Home Attack": (
                    f"{home_attack[metric]:.2f}"
                    if home_attack[metric] is not None
                    else "—"
                ),
                "Away Defence Conceded": (
                    f"{away_defence[metric]:.2f}"
                    if away_defence[metric] is not None
                    else "—"
                ),
                "Away Attack": (
                    f"{away_attack[metric]:.2f}"
                    if away_attack[metric] is not None
                    else "—"
                ),
                "Home Defence Conceded": (
                    f"{home_defence[metric]:.2f}"
                    if home_defence[metric] is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(attack_defence_rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CONTEXTUAL HOME / AWAY SAMPLES
    # --------------------------------------------------------

    st.divider()
    st.subheader("Contextual Home / Away Samples")

    context_rows = []

    context_metrics = [
        ("Shots", "team_shots", "opp_shots"),
        ("Shots on Target", "team_sot", "opp_sot"),
        ("Corners", "team_corners", "opp_corners"),
        ("Goals", "team_goals", "opp_goals")
    ]

    for label, team_col, opp_col in context_metrics:

        home_produced = average_value(
            home_matches,
            team_col
        )

        home_conceded = average_value(
            home_matches,
            opp_col
        )

        away_produced = average_value(
            away_matches,
            team_col
        )

        away_conceded = average_value(
            away_matches,
            opp_col
        )

        context_rows.append(
            {
                "Metric": label,
                f"{home_team} Home Produced": (
                    f"{home_produced:.2f}"
                    if home_produced is not None
                    else "—"
                ),
                f"{home_team} Home Conceded": (
                    f"{home_conceded:.2f}"
                    if home_conceded is not None
                    else "—"
                ),
                f"{away_team} Away Produced": (
                    f"{away_produced:.2f}"
                    if away_produced is not None
                    else "—"
                ),
                f"{away_team} Away Conceded": (
                    f"{away_conceded:.2f}"
                    if away_conceded is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(context_rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RECENT HOME FORM
    # --------------------------------------------------------

    st.divider()
    st.subheader(
        f"{home_team} — Recent Home Form"
    )

    if not home_matches.empty:

        home_display = home_matches[
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
        ].copy()

        st.dataframe(
            home_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            f"No home matches found for {home_team}."
        )

    # --------------------------------------------------------
    # RECENT AWAY FORM
    # --------------------------------------------------------

    st.subheader(
        f"{away_team} — Recent Away Form"
    )

    if not away_matches.empty:

        away_display = away_matches[
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
        ].copy()

        st.dataframe(
            away_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            f"No away matches found for {away_team}."
        )

    # --------------------------------------------------------
    # OVERALL FORM
    # --------------------------------------------------------

    st.divider()
    st.subheader("Overall Recent Form")

    form_columns = [
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

    left, right = st.columns(2)

    with left:

        st.caption(home_team)

        if not home_all.empty:

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

        else:

            st.info(
                "No data available."
            )

    with right:

        st.caption(away_team)

        if not away_all.empty:

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

        else:

            st.info(
                "No data available."
            )

    # --------------------------------------------------------
    # COMBINED MARKET VIEW
    # --------------------------------------------------------

    st.divider()
    st.subheader("Market Comparison")

    m1, m2, m3 = st.columns(3)

    with m1:

        research_market = st.selectbox(
            "Market",
            [
                "Shots",
                "Shots on Target",
                "Corners",
                "Goals"
            ],
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

    market_comparison = build_market_comparison(
        home_matches,
        away_matches,
        research_market,
        research_direction,
        research_line
    )

    st.dataframe(
        market_comparison,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # OPPONENT CONTEXT
    # --------------------------------------------------------

    st.subheader("Opponent Context")

    selected_market_column = MARKET_COLUMN_MAP[
        research_market
    ]

    opponent_column = {
        "Shots": "opp_shots",
        "Shots on Target": "opp_sot",
        "Corners": "opp_corners",
        "Goals": "opp_goals"
    }[research_market]

    # Home team's market vs away defensive context
    home_market_analysis = analyse_market(
        home_matches,
        research_market,
        research_direction,
        research_line
    )

    away_opponent_values = pd.to_numeric(
        away_matches[opponent_column],
        errors="coerce"
    ).dropna()

    # Away team's market vs home defensive context
    away_market_analysis = analyse_market(
        away_matches,
        research_market,
        research_direction,
        research_line
    )

    home_opponent_values = pd.to_numeric(
        home_matches[opponent_column],
        errors="coerce"
    ).dropna()

    if research_direction == "Over":

        away_defence_context_hits = (
            away_opponent_values < research_line
        ).sum()

        home_defence_context_hits = (
            home_opponent_values < research_line
        ).sum()

    else:

        away_defence_context_hits = (
            away_opponent_values > research_line
        ).sum()

        home_defence_context_hits = (
            home_opponent_values > research_line
        ).sum()

    away_defence_context_rate = (
        away_defence_context_hits
        /
        len(away_opponent_values)
        if len(away_opponent_values) > 0
        else None
    )

    home_defence_context_rate = (
        home_defence_context_hits
        /
        len(home_opponent_values)
        if len(home_opponent_values) > 0
        else None
    )

    opponent_context_df = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Market Hit Rate": fmt_pct(
                    home_market_analysis["hit_rate"]
                ),
                "Team Sample": home_market_analysis[
                    "sample_size"
                ],
                "Opponent Defensive Context": fmt_pct(
                    away_defence_context_rate
                ),
                "Opponent Sample": len(
                    away_opponent_values
                )
            },
            {
                "Team": away_team,
                "Market Hit Rate": fmt_pct(
                    away_market_analysis["hit_rate"]
                ),
                "Team Sample": away_market_analysis[
                    "sample_size"
                ],
                "Opponent Defensive Context": fmt_pct(
                    home_defence_context_rate
                ),
                "Opponent Sample": len(
                    home_opponent_values
                )
            }
        ]
    )

    st.dataframe(
        opponent_context_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Opponent Defensive Context is descriptive historical "
        "evidence from the relevant home/away sample. "
        "It is not a probability forecast."
    )

    # --------------------------------------------------------
    # HEAD-TO-HEAD
    # --------------------------------------------------------

    st.divider()
    st.subheader("Head-to-Head")

    h2h = get_match_history(
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
            h2h["competition"]
            == research_competition
        ]

    if not h2h.empty:

        h2h_display_columns = [
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

        available_h2h_columns = [
            c
            for c in h2h_display_columns
            if c in h2h.columns
        ]

        st.dataframe(
            h2h[available_h2h_columns].head(10),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"{len(h2h)} historical meeting(s) found "
            "for the selected filters."
        )

    else:

        st.info(
            "No previous meetings found for the selected filters."
        )

    # --------------------------------------------------------
    # MATCH RESEARCH SUMMARY
    # --------------------------------------------------------

    st.divider()
    st.subheader("Match Research Summary")

    summary_rows = [
        {
            "Area": "Home Attack — Shots",
            "Team": home_team,
            "Context": "Home",
            "Value": (
                f"{home_attack['Shots']:.2f}"
                if home_attack["Shots"] is not None
                else "—"
            )
        },
        {
            "Area": "Home Attack — SOT",
            "Team": home_team,
            "Context": "Home",
            "Value": (
                f"{home_attack['Shots on Target']:.2f}"
                if home_attack["Shots on Target"] is not None
                else "—"
            )
        },
        {
            "Area": "Away Defence — SOT Conceded",
            "Team": away_team,
            "Context": "Away",
            "Value": (
                f"{away_defence['Shots on Target']:.2f}"
                if away_defence["Shots on Target"] is not None
                else "—"
            )
        },
        {
            "Area": "Away Attack — Shots",
            "Team": away_team,
            "Context": "Away",
            "Value": (
                f"{away_attack['Shots']:.2f}"
                if away_attack["Shots"] is not None
                else "—"
            )
        },
        {
            "Area": "Away Attack — SOT",
            "Team": away_team,
            "Context": "Away",
            "Value": (
                f"{away_attack['Shots on Target']:.2f}"
                if away_attack["Shots on Target"] is not None
                else "—"
            )
        },
        {
            "Area": "Home Defence — SOT Conceded",
            "Team": home_team,
            "Context": "Home",
            "Value": (
                f"{home_defence['Shots on Target']:.2f}"
                if home_defence["Shots on Target"] is not None
                else "—"
            )
        }
    ]

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RESEARCH NOTES
    # --------------------------------------------------------

    st.divider()
    st.subheader("Research Notes")

    st.text_area(
        "Notes for this fixture",
        placeholder=(
            "Record lineup information, injuries, referee notes, "
            "tactical observations, team news, market observations, "
            "or anything else relevant to your research."
        ),
        height=180,
        key="match_research_notes"
    )

    st.caption(
        "This tab is a research workspace. Historical data describes "
        "previous matches and does not establish the probability "
        "of the next match."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "V1 deliberately avoids automatic 'safe bet' labels. "
    "The goal is to improve research quality and decision-making."
)
