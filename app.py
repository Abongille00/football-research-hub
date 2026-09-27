import streamlit as st
import pandas as pd
import numpy as np
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

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

OPPONENT_COLUMN_MAP = {
    "Shots": "opp_shots",
    "Shots on Target": "opp_sot",
    "Corners": "opp_corners",
    "Goals": "opp_goals"
}

WATCHLIST_FILE = "market_watchlist.csv"

MARKETS = [
    "Shots",
    "Shots on Target",
    "Corners",
    "Goals"
]

LINE_LADDER = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
    5.5,
    6.5,
    7.5,
    8.5,
    9.5,
    10.5,
    11.5,
    12.5,
    13.5,
    14.5,
    15.5,
    16.5,
    17.5,
    18.5,
    19.5,
    20.5
]


# ============================================================
# DATA VALIDATION
# ============================================================

def normalize_columns(df):
    """Normalize column names."""

    df = df.copy()

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    return df


def validate_required_columns(df):
    """Return missing required columns."""

    if df is None:
        return REQUIRED_COLUMNS.copy()

    return [
        c
        for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
    """Standardise and clean football dataset."""

    df = normalize_columns(df)

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
                .fillna("")
                .astype(str)
                .str.strip()
            )

    numeric_columns = [
        c
        for c in REQUIRED_COLUMNS
        if c not in text_columns + ["date"]
    ]

    for c in numeric_columns:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    if "home_team" in df.columns:
        df = df[df["home_team"].ne("")]

    if "away_team" in df.columns:
        df = df[df["away_team"].ne("")]

    if "date" in df.columns:

        df = df.sort_values(
            "date",
            ascending=False
        )

    return df.reset_index(drop=True)


# ============================================================
# TEAM MATCH TRANSFORMATION
# ============================================================

def team_matches(df, team):
    """Convert raw matches into selected team's perspective."""

    if df is None or df.empty:
        return pd.DataFrame()

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


# ============================================================
# FILTERING
# ============================================================

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

    if m.empty:
        return m

    if season != "All":
        m = m[
            m["season"].eq(season)
        ]

    if competition != "All":
        m = m[
            m["competition"].eq(competition)
        ]

    if venue != "All":
        m = m[
            m["venue"].eq(venue)
        ]

    if "date" in m.columns:

        m = m.sort_values(
            "date",
            ascending=False
        )

    return m.head(
        int(sample)
    ).reset_index(drop=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def average_value(df, column):
    """Return numeric average."""

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


def fmt_number(value):
    """Format a numeric value."""

    if value is None or pd.isna(value):
        return "—"

    return f"{value:.2f}"


def fmt_pct(value):
    """Format probability/rate."""

    if value is None or pd.isna(value):
        return "—"

    return f"{value * 100:.1f}%"


def sample_quality(sample_size):
    """Describe historical sample size."""

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


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyse_market(
    df,
    market,
    direction,
    line
):
    """Analyse a team market."""

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

    if column is None:
        return empty_result

    if column not in df.columns:
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

    column = MARKET_COLUMN_MAP.get(
        market
    )

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
        c
        for c in result_columns
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


def defensive_market_analysis(
    df,
    market,
    direction,
    line
):
    """
    Analyse opponent output against the selected team.

    Example:
    Home team's SOT conceded Over 3.5.
    """

    if df is None or df.empty:
        return {
            "hit_rate": None,
            "hits": 0,
            "misses": 0,
            "sample_size": 0
        }

    column = OPPONENT_COLUMN_MAP.get(
        market
    )

    if column not in df.columns:
        return {
            "hit_rate": None,
            "hits": 0,
            "misses": 0,
            "sample_size": 0
        }

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return {
            "hit_rate": None,
            "hits": 0,
            "misses": 0,
            "sample_size": 0
        }

    if direction == "Over":

        hits = int(
            (values > line).sum()
        )

    else:

        hits = int(
            (values < line).sum()
        )

    sample_size = len(values)

    return {
        "hit_rate": hits / sample_size,
        "hits": hits,
        "misses": sample_size - hits,
        "sample_size": sample_size
    }


def defensive_market_history(
    df,
    market,
    direction,
    line
):
    """Match-by-match defensive market history."""

    if df is None or df.empty:
        return pd.DataFrame()

    column = OPPONENT_COLUMN_MAP.get(
        market
    )

    if column not in df.columns:
        return pd.DataFrame()

    result = df[
        [
            "date",
            "home_team",
            "away_team",
            "venue",
            column
        ]
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
# LINE LADDER
# ============================================================

def build_line_ladder(
    df,
    market,
    direction,
    max_line=15.5
):
    """
    Test multiple lines automatically.

    Equality is deliberately not treated as a hit.
    """

    rows = []

    lines = [
        x
        for x in LINE_LADDER
        if x <= max_line
    ]

    for line in lines:

        result = analyse_market(
            df,
            market,
            direction,
            line
        )

        rows.append(
            {
                "Line": line,
                "Hits": result["hits"],
                "Misses": result["misses"],
                "Sample": result["sample_size"],
                "Hit Rate": fmt_pct(
                    result["hit_rate"]
                )
            }
        )

    return pd.DataFrame(rows)


def build_defensive_line_ladder(
    df,
    market,
    direction,
    max_line=15.5
):
    """Test defensive concession lines."""

    rows = []

    lines = [
        x
        for x in LINE_LADDER
        if x <= max_line
    ]

    for line in lines:

        result = defensive_market_analysis(
            df,
            market,
            direction,
            line
        )

        rows.append(
            {
                "Line": line,
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
# TREND ANALYSIS
# ============================================================

def split_recent_periods(
    df,
    recent_size=5
):
    """
    Split a chronologically sorted dataframe into:

    Last N
    Previous N
    """

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sorted_df = df.sort_values(
        "date",
        ascending=False
    ).reset_index(drop=True)

    recent = sorted_df.head(
        recent_size
    )

    previous = sorted_df.iloc[
        recent_size:recent_size * 2
    ]

    return recent, previous


def trend_average(
    df,
    column
):
    return average_value(
        df,
        column
    )


def build_trend_table(
    df,
    recent_size=5
):
    """Compare last N against previous N."""

    recent, previous = split_recent_periods(
        df,
        recent_size
    )

    rows = []

    metrics = [
        ("Shots", "team_shots"),
        ("Shots on Target", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]

    for label, column in metrics:

        recent_avg = trend_average(
            recent,
            column
        )

        previous_avg = trend_average(
            previous,
            column
        )

        if (
            recent_avg is not None
            and previous_avg is not None
        ):
            change = (
                recent_avg
                -
                previous_avg
            )
        else:
            change = None

        rows.append(
            {
                "Metric": label,
                f"Last {recent_size} Avg": fmt_number(
                    recent_avg
                ),
                f"Previous {recent_size} Avg": fmt_number(
                    previous_avg
                ),
                "Change": fmt_number(
                    change
                )
            }
        )

    return pd.DataFrame(rows)


def build_defensive_trend_table(
    df,
    recent_size=5
):
    """Compare defensive concession trends."""

    recent, previous = split_recent_periods(
        df,
        recent_size
    )

    rows = []

    metrics = [
        ("Shots Conceded", "opp_shots"),
        ("SOT Conceded", "opp_sot"),
        ("Corners Conceded", "opp_corners"),
        ("Goals Conceded", "opp_goals")
    ]

    for label, column in metrics:

        recent_avg = average_value(
            recent,
            column
        )

        previous_avg = average_value(
            previous,
            column
        )

        if (
            recent_avg is not None
            and previous_avg is not None
        ):
            change = (
                recent_avg
                -
                previous_avg
            )
        else:
            change = None

        rows.append(
            {
                "Metric": label,
                f"Last {recent_size} Avg": fmt_number(
                    recent_avg
                ),
                f"Previous {recent_size} Avg": fmt_number(
                    previous_avg
                ),
                "Change": fmt_number(
                    change
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# OPPONENT CONTEXT
# ============================================================

def opponent_for_team_match(row):
    """Return opponent from a team-perspective row."""

    if row.get("venue") == "Home":
        return row.get("away_team", "")

    return row.get("home_team", "")


def get_opponent_context(
    data,
    team_matches_df,
    metric,
    venue_filter="All"
):
    """
    Add opponent and opponent's historical defensive context.

    The context is descriptive:
    how many shots/SOT/corners/goals that opponent has
    historically conceded in the available dataset.
    """

    if (
        team_matches_df is None
        or team_matches_df.empty
    ):
        return pd.DataFrame()

    opponent_column = {
        "Shots": "team_shots",
        "Shots on Target": "team_sot",
        "Corners": "team_corners",
        "Goals": "team_goals"
    }.get(metric)

    if opponent_column is None:
        return pd.DataFrame()

    rows = []

    for _, row in team_matches_df.iterrows():

        opponent = opponent_for_team_match(
            row
        )

        if not opponent:
            continue

        opponent_matches = team_matches(
            data,
            opponent
        )

        if venue_filter != "All":
            opponent_matches = opponent_matches[
                opponent_matches["venue"]
                == venue_filter
            ]

        opponent_avg = average_value(
            opponent_matches,
            OPPONENT_COLUMN_MAP[metric]
        )

        team_value = row.get(
            MARKET_COLUMN_MAP[metric]
        )

        rows.append(
            {
                "Date": row.get("date"),
                "Opponent": opponent,
                "Venue": row.get("venue"),
                f"{metric} Produced": team_value,
                f"Opponent Avg {metric} Conceded":
                    opponent_avg
            }
        )

    return pd.DataFrame(rows)


def opponent_context_summary(
    data,
    team_matches_df,
    metric
):
    """Summarise opponent defensive context."""

    context = get_opponent_context(
        data,
        team_matches_df,
        metric
    )

    if context.empty:
        return context

    context[
        f"{metric} Produced"
    ] = pd.to_numeric(
        context[f"{metric} Produced"],
        errors="coerce"
    )

    context[
        f"Opponent Avg {metric} Conceded"
    ] = pd.to_numeric(
        context[
            f"Opponent Avg {metric} Conceded"
        ],
        errors="coerce"
    )

    context["Context"] = np.select(
        [
            context[
                f"Opponent Avg {metric} Conceded"
            ] <= context[
                f"Opponent Avg {metric} Conceded"
            ].quantile(0.33),

            context[
                f"Opponent Avg {metric} Conceded"
            ] >= context[
                f"Opponent Avg {metric} Conceded"
            ].quantile(0.67)
        ],
        [
            "Lower-concession context",
            "Higher-concession context"
        ],
        default="Middle-concession context"
    )

    return context


# ============================================================
# TEAM SUMMARY
# ============================================================

def team_summary(df):
    """Create descriptive team statistics."""

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


# ============================================================
# HEAD TO HEAD
# ============================================================

def get_match_history(
    data,
    home_team,
    away_team
):
    """Find historical meetings."""

    if data is None or data.empty:
        return pd.DataFrame()

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


# ============================================================
# MARKET COMPARISON
# ============================================================

def build_market_comparison(
    home_matches,
    away_matches,
    market,
    direction,
    line
):
    """Compare selected market."""

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

    return pd.DataFrame(
        [
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
                "Hit Rate": fmt_pct(
                    home_result[
                        "hit_rate"
                    ]
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
                "Hit Rate": fmt_pct(
                    away_result[
                        "hit_rate"
                    ]
                )
            }
        ]
    )


# ============================================================
# TEAM / SEASON / COMPETITION LISTS
# ============================================================

def team_list(data):
    """Return unique teams."""

    if data is None or data.empty:
        return []

    home = set(
        data["home_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    away = set(
        data["away_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        x
        for x in home | away
        if x
    )


def season_list(data):
    """Return clean season options."""

    if (
        data is None
        or data.empty
        or "season" not in data.columns
    ):
        return ["All"]

    values = [
        str(x).strip()
        for x in data["season"].dropna().unique()
        if str(x).strip()
    ]

    return [
        "All"
    ] + sorted(
        values,
        reverse=True
    )


def competition_list(data):
    """Return clean competition options."""

    if (
        data is None
        or data.empty
        or "competition" not in data.columns
    ):
        return ["All"]

    values = [
        str(x).strip()
        for x in data["competition"].dropna().unique()
        if str(x).strip()
    ]

    return [
        "All"
    ] + sorted(values)


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

        raw_data = pd.read_csv(
            uploaded
        )

        raw_data = normalize_columns(
            raw_data
        )

        missing = validate_required_columns(
            raw_data
        )

        if missing:

            st.error(
                "Your CSV is missing these columns: "
                + ", ".join(missing)
            )

            st.stop()

        data = clean_data(
            raw_data
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

    master_file = (
        "football_master_2024_27_v1.csv"
    )

    if os.path.exists(
        master_file
    ):

        try:

            raw_data = pd.read_csv(
                master_file
            )

            raw_data = normalize_columns(
                raw_data
            )

            missing = validate_required_columns(
                raw_data
            )

            if missing:

                st.error(
                    f"{master_file} is missing: "
                    + ", ".join(missing)
                )

                st.stop()

            data = clean_data(
                raw_data
            )

            st.sidebar.success(
                f"Using master dataset — "
                f"{len(data)} matches"
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

        data = clean_data(
            demo
        )

        st.sidebar.info(
            f"Using demo dataset — {len(data)} matches"
        )


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

missing = validate_required_columns(
    data
)

if missing:

    st.error(
        "Your dataset is missing these columns: "
        + ", ".join(missing)
    )

    st.stop()

if data.empty:

    st.error(
        "The dataset contains no usable matches."
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

    st.subheader(
        "Team Research Dashboard"
    )

    teams = team_list(
        data
    )

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
        [
            "All",
            "Home",
            "Away"
        ],
        key="dashboard_venue"
    )

    dashboard_sample = st.select_slider(
        "Recent sample",
        options=[
            5,
            10,
            15
        ],
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

    dashboard_size = len(
        dashboard_df
    )

    quality_label, quality_message = sample_quality(
        dashboard_size
    )

    if dashboard_size < 5:

        st.warning(
            f"⚠️ {quality_label}: "
            f"{quality_message}"
        )

    elif dashboard_size < 10:

        st.info(
            f"ℹ️ {quality_label}: "
            f"{quality_message}"
        )

    else:

        st.success(
            f"✓ {quality_label}: "
            f"{quality_message}"
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
        fmt_number(
            dashboard_metrics.get(
                "Shots Avg"
            )
        )
    )

    d3.metric(
        "Avg SOT",
        fmt_number(
            dashboard_metrics.get(
                "Shots on Target Avg"
            )
        )
    )

    d4.metric(
        "Avg Corners",
        fmt_number(
            dashboard_metrics.get(
                "Corners Avg"
            )
        )
    )

    d5.metric(
        "Avg Goals",
        fmt_number(
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

    for market_name in MARKETS:

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

                    overview_rows.append(
                        {
                            "Market": market_name,
                            "Direction": direction_name,
                            "Line": test_line,
                            "Hits": result["hits"],
                            "Sample": result[
                                "sample_size"
                            ],
                            "Hit Rate": fmt_pct(
                                result["hit_rate"]
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
            "No market data is available."
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
                "Avg Shots": fmt_number(
                    split_summary.get(
                        "Shots Avg"
                    )
                ),
                "Avg SOT": fmt_number(
                    split_summary.get(
                        "Shots on Target Avg"
                    )
                ),
                "Avg Corners": fmt_number(
                    split_summary.get(
                        "Corners Avg"
                    )
                ),
                "Avg Goals": fmt_number(
                    split_summary.get(
                        "Goals Avg"
                    )
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
            dashboard_df[
                available_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No matches available."
        )


# ============================================================
# TAB 2 — MARKET TESTER
# ============================================================

with tab2:

    st.subheader(
        "Test a Market"
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
        [
            "All",
            "Home",
            "Away"
        ],
        key="market_venue"
    )

    market = st.selectbox(
        "Market",
        MARKETS,
        key="market_type"
    )

    direction = st.selectbox(
        "Direction",
        [
            "Over",
            "Under"
        ],
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
        options=[
            5,
            10,
            15
        ],
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

    rate = analysis[
        "hit_rate"
    ]

    hits = analysis[
        "hits"
    ]

    misses = analysis[
        "misses"
    ]

    sample_size = analysis[
        "sample_size"
    ]

    breakeven = 1 / odds

    quality_label, quality_message = sample_quality(
        sample_size
    )

    if sample_size < 5:

        st.warning(
            f"⚠️ {quality_label}: "
            f"{quality_message}"
        )

    elif sample_size < 10:

        st.info(
            f"ℹ️ {quality_label}: "
            f"{quality_message}"
        )

    else:

        st.success(
            f"✓ {quality_label}: "
            f"{quality_message}"
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
            min(
                max(rate, 0),
                1
            )
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

    st.subheader(
        "Overall / Home / Away Comparison"
    )

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
                "Matches": venue_analysis[
                    "sample_size"
                ],
                "Hit Rate": fmt_pct(
                    venue_analysis[
                        "hit_rate"
                    ]
                ),
                "Avg Shots": fmt_number(
                    venue_summary.get(
                        "Shots Avg"
                    )
                ),
                "Avg SOT": fmt_number(
                    venue_summary.get(
                        "Shots on Target Avg"
                    )
                ),
                "Avg Corners": fmt_number(
                    venue_summary.get(
                        "Corners Avg"
                    )
                ),
                "Avg Goals": fmt_number(
                    venue_summary.get(
                        "Goals Avg"
                    )
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            comparison_data
        ),
        use_container_width=True,
        hide_index=True
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

    else:

        st.info(
            "No market history is available."
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
# TAB 3 — BOOKMAKER MARKET MONITOR
# ============================================================

with tab3:

    st.subheader(
        "Bookmaker Market Monitor"
    )

    # --------------------------------------------------------
    # LOAD WATCHLIST
    # --------------------------------------------------------

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
            placeholder=(
                "e.g. Chelsea vs Brentford"
            ),
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

    with col2:

        monitor_direction = st.selectbox(
            "Direction",
            [
                "Over",
                "Under"
            ],
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
            [
                5,
                10,
                15
            ],
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
            [
                "All",
                "Home",
                "Away"
            ],
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

        rate = analysis[
            "hit_rate"
        ]

        breakeven = (
            1 / monitor_odds
        )

        if rate is not None:

            historical_hit_rate = (
                rate * 100
            )

            break_even_rate = (
                breakeven * 100
            )

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

            edge_display = (
                f"{edge:+.1f} pp"
            )

            historical_display = (
                f"{historical_hit_rate:.1f}%"
            )

        else:

            historical_display = "—"
            edge_display = "—"

            historical_vs_breakeven = "—"

            break_even_rate = (
                breakeven * 100
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
            "Historical Hit Rate":
                historical_display,
            "Break-even":
                f"{break_even_rate:.1f}%",
            "Edge vs Break-even":
                edge_display,
            "Historical vs Break-even":
                historical_vs_breakeven,
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

    st.subheader(
        "Tracked Markets"
    )

    if st.session_state.market_watchlist:

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
                            >
                            market_line
                        )

                    else:

                        won_result = (
                            actual
                            <
                            market_line
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

        tracked_hit_rate_display = (
            f"{won / settled * 100:.1f}%"
            if settled > 0
            else "—"
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
            tracked_hit_rate_display
        )

        st.caption(
            f"Void markets: {void}"
        )

        st.subheader(
            "Edit / Settle Markets"
        )

        status_options = [
            "Watching",
            "Won",
            "Lost",
            "Void"
        ]

        for i in range(
            len(watchlist_df)
        ):

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

                        current_status = (
                            "Watching"
                        )

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

                    st.session_state.market_watchlist.pop(
                        i
                    )

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

        watchlist_df = pd.DataFrame(
            st.session_state.market_watchlist
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
        "CSV Format"
    )

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

    st.divider()

    st.subheader(
        "Data Quality Checks"
    )

    quality_rows = []

    for column in REQUIRED_COLUMNS:

        missing_count = int(
            data[column].isna().sum()
        )

        quality_rows.append(
            {
                "Column": column,
                "Missing Values":
                    missing_count,
                "Available Values":
                    len(data) - missing_count
            }
        )

    st.dataframe(
        pd.DataFrame(
            quality_rows
        ),
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
        "Research a specific fixture by comparing both teams "
        "using historical match-by-match data."
    )

    # ========================================================
    # STEP 19 — FIXTURE SELECTION + SEPARATE SAMPLE CONTROLS
    # ========================================================

    st.divider()

    st.subheader(
        "1. Fixture & Research Samples"
    )

    teams = team_list(
        data
    )

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

        if not away_team_options:

            st.error(
                "At least two teams are required."
            )

            st.stop()

        away_team = st.selectbox(
            "Away Team",
            away_team_options,
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

        h2h_sample = st.selectbox(
            "H2H sample",
            [
                5,
                10,
                15
            ],
            index=1,
            key="research_h2h_sample"
        )

    s1, s2, s3 = st.columns(3)

    with s1:

        home_sample = st.selectbox(
            "Home team's home sample",
            [
                5,
                10,
                15
            ],
            index=1,
            key="research_home_sample"
        )

    with s2:

        away_sample = st.selectbox(
            "Away team's away sample",
            [
                5,
                10,
                15
            ],
            index=1,
            key="research_away_sample"
        )

    with s3:

        overall_sample = st.selectbox(
            "Overall recent sample",
            [
                5,
                10,
                15
            ],
            index=1,
            key="research_overall_sample"
        )

    # ========================================================
    # FILTER MATCHES
    # ========================================================

    home_matches = filtered_team_matches(
        data=data,
        team=home_team,
        season=research_season,
        competition=research_competition,
        venue="Home",
        sample=home_sample
    )

    away_matches = filtered_team_matches(
        data=data,
        team=away_team,
        season=research_season,
        competition=research_competition,
        venue="Away",
        sample=away_sample
    )

    home_all = filtered_team_matches(
        data=data,
        team=home_team,
        season=research_season,
        competition=research_competition,
        venue="All",
        sample=overall_sample
    )

    away_all = filtered_team_matches(
        data=data,
        team=away_team,
        season=research_season,
        competition=research_competition,
        venue="All",
        sample=overall_sample
    )

    # ========================================================
    # SAMPLE WARNINGS
    # ========================================================

    if len(home_matches) < 5:

        st.warning(
            f"{home_team}: only "
            f"{len(home_matches)} home matches "
            "are available."
        )

    if len(away_matches) < 5:

        st.warning(
            f"{away_team}: only "
            f"{len(away_matches)} away matches "
            "are available."
        )

    # ========================================================
    # TEAM COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "2. Home vs Away Team Comparison"
    )

    home_summary = match_team_summary(
        home_matches
    )

    away_summary = match_team_summary(
        away_matches
    )

    comparison_rows = [
        {
            "Metric": "Matches",
            "Home Team":
                home_summary["Matches"],
            "Away Team":
                away_summary["Matches"]
        },
        {
            "Metric": "Avg Shots",
            "Home Team":
                fmt_number(
                    home_summary["Shots"]
                ),
            "Away Team":
                fmt_number(
                    away_summary["Shots"]
                )
        },
        {
            "Metric":
                "Avg Shots on Target",
            "Home Team":
                fmt_number(
                    home_summary["SOT"]
                ),
            "Away Team":
                fmt_number(
                    away_summary["SOT"]
                )
        },
        {
            "Metric": "Avg Corners",
            "Home Team":
                fmt_number(
                    home_summary["Corners"]
                ),
            "Away Team":
                fmt_number(
                    away_summary["Corners"]
                )
        },
        {
            "Metric": "Avg Goals",
            "Home Team":
                fmt_number(
                    home_summary["Goals"]
                ),
            "Away Team":
                fmt_number(
                    away_summary["Goals"]
                )
        }
    ]

    st.dataframe(
        pd.DataFrame(
            comparison_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 20 — ATTACK VS DEFENCE
    # ========================================================

    st.divider()

    st.subheader(
        "3. Attack vs Defence"
    )

    st.caption(
        "Attacking production is compared with the opponent's "
        "historical defensive concession."
    )

    attack_defence_rows = []

    for metric, team_col, opp_col in [
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
    ]:

        home_attack = average_value(
            home_matches,
            team_col
        )

        away_defence = average_value(
            away_matches,
            opp_col
        )

        away_attack = average_value(
            away_matches,
            team_col
        )

        home_defence = average_value(
            home_matches,
            opp_col
        )

        attack_defence_rows.append(
            {
                "Metric": metric,
                "Home Attack":
                    fmt_number(home_attack),
                "Away Defence Conceded":
                    fmt_number(away_defence),
                "Away Attack":
                    fmt_number(away_attack),
                "Home Defence Conceded":
                    fmt_number(home_defence)
            }
        )

    st.dataframe(
        pd.DataFrame(
            attack_defence_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 20B — DEFENSIVE MARKET TESTER
    # ========================================================

    st.subheader(
        "Defensive Market Tester"
    )

    dm1, dm2, dm3 = st.columns(3)

    with dm1:

        defensive_market = st.selectbox(
            "Defensive market",
            MARKETS,
            key="research_defensive_market"
        )

    with dm2:

        defensive_direction = st.selectbox(
            "Defensive direction",
            [
                "Over",
                "Under"
            ],
            key="research_defensive_direction"
        )

    with dm3:

        defensive_line = st.number_input(
            "Defensive line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="research_defensive_line"
        )

    home_defensive_result = defensive_market_analysis(
        home_matches,
        defensive_market,
        defensive_direction,
        defensive_line
    )

    away_defensive_result = defensive_market_analysis(
        away_matches,
        defensive_market,
        defensive_direction,
        defensive_line
    )

    defensive_comparison = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Venue": "Home",
                "Opponent output":
                    "Historical opponents",
                "Hits":
                    home_defensive_result["hits"],
                "Misses":
                    home_defensive_result["misses"],
                "Sample":
                    home_defensive_result[
                        "sample_size"
                    ],
                "Hit Rate":
                    fmt_pct(
                        home_defensive_result[
                            "hit_rate"
                        ]
                    )
            },
            {
                "Team": away_team,
                "Venue": "Away",
                "Opponent output":
                    "Historical opponents",
                "Hits":
                    away_defensive_result["hits"],
                "Misses":
                    away_defensive_result["misses"],
                "Sample":
                    away_defensive_result[
                        "sample_size"
                    ],
                "Hit Rate":
                    fmt_pct(
                        away_defensive_result[
                            "hit_rate"
                        ]
                    )
            }
        ]
    )

    st.dataframe(
        defensive_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "For example, SOT Over 3.5 here means historical "
        "opponents recorded more than 3.5 SOT against the team."
    )

    # ========================================================
    # CONTEXTUAL HOME / AWAY SAMPLES
    # ========================================================

    st.divider()

    st.subheader(
        "4. Contextual Home / Away Samples"
    )

    context_rows = []

    for label, team_col, opp_col in [
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
    ]:

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
                f"{home_team} Home Produced":
                    fmt_number(home_produced),
                f"{home_team} Home Conceded":
                    fmt_number(home_conceded),
                f"{away_team} Away Produced":
                    fmt_number(away_produced),
                f"{away_team} Away Conceded":
                    fmt_number(away_conceded)
            }
        )

    st.dataframe(
        pd.DataFrame(
            context_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STEP 21 — LINE LADDER
    # ========================================================

    st.divider()

    st.subheader(
        "5. Market Line Ladder"
    )

    st.caption(
        "Automatically tests multiple lines against the selected "
        "home and away samples. Equality is not counted as a hit."
    )

    ladder_market = st.selectbox(
        "Line ladder market",
        MARKETS,
        key="research_ladder_market"
    )

    ladder_direction = st.selectbox(
        "Line ladder direction",
        [
            "Over",
            "Under"
        ],
        key="research_ladder_direction"
    )

    ladder_max = st.selectbox(
        "Maximum line",
        [
            5.5,
            7.5,
            9.5,
            12.5,
            15.5,
            20.5
        ],
        index=3,
        key="research_ladder_max"
    )

    home_ladder = build_line_ladder(
        home_matches,
        ladder_market,
        ladder_direction,
        ladder_max
    )

    away_ladder = build_line_ladder(
        away_matches,
        ladder_market,
        ladder_direction,
        ladder_max
    )

    ladder_col1, ladder_col2 = st.columns(2)

    with ladder_col1:

        st.markdown(
            f"**{home_team} — Home**"
        )

        st.dataframe(
            home_ladder,
            use_container_width=True,
            hide_index=True
        )

    with ladder_col2:

        st.markdown(
            f"**{away_team} — Away**"
        )

        st.dataframe(
            away_ladder,
            use_container_width=True,
            hide_index=True
        )

    st.subheader(
        "Defensive Line Ladder"
    )

    home_def_ladder = build_defensive_line_ladder(
        home_matches,
        ladder_market,
        ladder_direction,
        ladder_max
    )

    away_def_ladder = build_defensive_line_ladder(
        away_matches,
        ladder_market,
        ladder_direction,
        ladder_max
    )

    def_col1, def_col2 = st.columns(2)

    with def_col1:

        st.markdown(
            f"**{home_team} — Defensive Concession**"
        )

        st.dataframe(
            home_def_ladder,
            use_container_width=True,
            hide_index=True
        )

    with def_col2:

        st.markdown(
            f"**{away_team} — Defensive Concession**"
        )

        st.dataframe(
            away_def_ladder,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # STEP 22 — RECENT TREND
    # ========================================================

    st.divider()

    st.subheader(
        "6. Recent Trend — Last 5 vs Previous 5"
    )

    st.caption(
        "This section is available when at least 10 relevant "
        "matches exist. Change is Last 5 minus Previous 5."
    )

    trend1, trend2 = st.columns(2)

    with trend1:

        st.markdown(
            f"**{home_team} — Home Trend**"
        )

        home_trend = build_trend_table(
            home_matches,
            recent_size=5
        )

        st.dataframe(
            home_trend,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f"**{home_team} — Defensive Trend**"
        )

        home_def_trend = build_defensive_trend_table(
            home_matches,
            recent_size=5
        )

        st.dataframe(
            home_def_trend,
            use_container_width=True,
            hide_index=True
        )

    with trend2:

        st.markdown(
            f"**{away_team} — Away Trend**"
        )

        away_trend = build_trend_table(
            away_matches,
            recent_size=5
        )

        st.dataframe(
            away_trend,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f"**{away_team} — Defensive Trend**"
        )

        away_def_trend = build_defensive_trend_table(
            away_matches,
            recent_size=5
        )

        st.dataframe(
            away_def_trend,
            use_container_width=True,
            hide_index=True
        )

    if len(home_matches) < 10:

        st.info(
            f"{home_team}: fewer than 10 home matches "
            "are available, so the Last 5 vs Previous 5 "
            "comparison is incomplete."
        )

    if len(away_matches) < 10:

        st.info(
            f"{away_team}: fewer than 10 away matches "
            "are available, so the Last 5 vs Previous 5 "
            "comparison is incomplete."
        )

    # ========================================================
    # STEP 23 — MATCH-BY-MATCH OPPONENT CONTEXT
    # ========================================================

    st.divider()

    st.subheader(
        "7. Opponent Context"
    )

    st.caption(
        "This section examines the historical defensive environment "
        "of the opponents faced. It does not predict the next match."
    )

    opponent_metric = st.selectbox(
        "Opponent-context metric",
        MARKETS,
        key="research_opponent_metric"
    )

    context_team = st.selectbox(
        "Team to inspect",
        [
            home_team,
            away_team
        ],
        key="research_context_team"
    )

    if context_team == home_team:

        context_source = home_matches

    else:

        context_source = away_matches

    opponent_context = opponent_context_summary(
        data,
        context_source,
        opponent_metric
    )

    if not opponent_context.empty:

        st.dataframe(
            opponent_context,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Lower-concession context means the opponent's "
            "historical average concession falls in the lower "
            "part of this sample. Higher-concession context means "
            "it falls in the higher part."
        )

        st.subheader(
            "Context Split"
        )

        context_group = (
            opponent_context
            .groupby("Context")
            .agg(
                Matches=(
                    "Opponent",
                    "count"
                ),
                Avg_Produced=(
                    f"{opponent_metric} Produced",
                    "mean"
                ),
                Avg_Opponent_Conceded=(
                    f"Opponent Avg "
                    f"{opponent_metric} Conceded",
                    "mean"
                )
            )
            .reset_index()
        )

        context_group[
            "Avg_Produced"
        ] = context_group[
            "Avg_Produced"
        ].round(2)

        context_group[
            "Avg_Opponent_Conceded"
        ] = context_group[
            "Avg_Opponent_Conceded"
        ].round(2)

        context_group = context_group.rename(
            columns={
                "Context":
                    "Opponent Context",
                "Avg_Produced":
                    f"Avg {opponent_metric} Produced",
                "Avg_Opponent_Conceded":
                    f"Avg Opponent {opponent_metric} Conceded"
            }
        )

        st.dataframe(
            context_group,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Opponent context could not be calculated "
            "for the selected sample."
        )

    # ========================================================
    # RECENT HOME FORM
    # ========================================================

    st.divider()

    st.subheader(
        f"8. {home_team} — Recent Home Form"
    )

    if not home_matches.empty:

        home_display_columns = [
            "date",
            "home_team",
            "away_team",
            "team_goals",
            "opp_goals",
            "team_shots",
            "team_sot",
            "team_corners"
        ]

        st.dataframe(
            home_matches[
                [
                    c
                    for c in home_display_columns
                    if c in home_matches.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            f"No home matches found for {home_team}."
        )

    # ========================================================
    # RECENT AWAY FORM
    # ========================================================

    st.subheader(
        f"{away_team} — Recent Away Form"
    )

    if not away_matches.empty:

        away_display_columns = [
            "date",
            "home_team",
            "away_team",
            "team_goals",
            "opp_goals",
            "team_shots",
            "team_sot",
            "team_corners"
        ]

        st.dataframe(
            away_matches[
                [
                    c
                    for c in away_display_columns
                    if c in away_matches.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            f"No away matches found for {away_team}."
        )

    # ========================================================
    # OVERALL FORM
    # ========================================================

    st.divider()

    st.subheader(
        "9. Overall Recent Form"
    )

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

        st.caption(
            home_team
        )

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

        st.caption(
            away_team
        )

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

    # ========================================================
    # MARKET COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "10. Market Comparison"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        research_market = st.selectbox(
            "Market",
            MARKETS,
            key="research_market"
        )

    with m2:

        research_direction = st.selectbox(
            "Direction",
            [
                "Over",
                "Under"
            ],
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

    # ========================================================
    # OPPONENT DEFENSIVE CONTEXT FOR SELECTED MARKET
    # ========================================================

    st.subheader(
        "Selected Market — Opponent Defensive Context"
    )

    home_market_analysis = analyse_market(
        home_matches,
        research_market,
        research_direction,
        research_line
    )

    away_market_analysis = analyse_market(
        away_matches,
        research_market,
        research_direction,
        research_line
    )

    home_def_context = defensive_market_analysis(
        away_matches,
        research_market,
        research_direction,
        research_line
    )

    away_def_context = defensive_market_analysis(
        home_matches,
        research_market,
        research_direction,
        research_line
    )

    opponent_context_df = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Team Market Hit Rate":
                    fmt_pct(
                        home_market_analysis[
                            "hit_rate"
                        ]
                    ),
                "Team Sample":
                    home_market_analysis[
                        "sample_size"
                    ],
                "Relevant Opponent Defence":
                    f"{away_team} historical "
                    f"{research_market} concession",
                "Defensive Context Hit Rate":
                    fmt_pct(
                        home_def_context[
                            "hit_rate"
                        ]
                    ),
                "Defensive Sample":
                    home_def_context[
                        "sample_size"
                    ]
            },
            {
                "Team": away_team,
                "Team Market Hit Rate":
                    fmt_pct(
                        away_market_analysis[
                            "hit_rate"
                        ]
                    ),
                "Team Sample":
                    away_market_analysis[
                        "sample_size"
                    ],
                "Relevant Opponent Defence":
                    f"{home_team} historical "
                    f"{research_market} concession",
                "Defensive Context Hit Rate":
                    fmt_pct(
                        away_def_context[
                            "hit_rate"
                        ]
                    ),
                "Defensive Sample":
                    away_def_context[
                        "sample_size"
                    ]
            }
        ]
    )

    st.dataframe(
        opponent_context_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # SELECTED MARKET MATCH HISTORY
    # ========================================================

    st.subheader(
        "Selected Market — Match-by-Match History"
    )

    hist1, hist2 = st.columns(2)

    with hist1:

        st.markdown(
            f"**{home_team} — {research_market} "
            f"{research_direction} {research_line}**"
        )

        home_market_history = market_history(
            home_matches,
            research_market,
            research_direction,
            research_line
        )

        if not home_market_history.empty:

            st.dataframe(
                home_market_history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No history available."
            )

    with hist2:

        st.markdown(
            f"**{away_team} — {research_market} "
            f"{research_direction} {research_line}**"
        )

        away_market_history = market_history(
            away_matches,
            research_market,
            research_direction,
            research_line
        )

        if not away_market_history.empty:

            st.dataframe(
                away_market_history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No history available."
            )

    # ========================================================
    # DEFENSIVE MATCH HISTORY
    # ========================================================

    st.subheader(
        "Defensive Match-by-Match History"
    )

    dh1, dh2 = st.columns(2)

    with dh1:

        st.markdown(
            f"**{home_team} — Opponent "
            f"{research_market}**"
        )

        home_def_history = defensive_market_history(
            home_matches,
            research_market,
            research_direction,
            research_line
        )

        if not home_def_history.empty:

            st.dataframe(
                home_def_history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No defensive history available."
            )

    with dh2:

        st.markdown(
            f"**{away_team} — Opponent "
            f"{research_market}**"
        )

        away_def_history = defensive_market_history(
            away_matches,
            research_market,
            research_direction,
            research_line
        )

        if not away_def_history.empty:

            st.dataframe(
                away_def_history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No defensive history available."
            )

    # ========================================================
    # HEAD TO HEAD
    # ========================================================

    st.divider()

    st.subheader(
        "11. Head-to-Head"
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

    h2h = h2h.head(
        h2h_sample
    )

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
            h2h[
                available_h2h_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"{len(h2h)} historical meeting(s) "
            "shown for the selected filters."
        )

    else:

        st.info(
            "No previous meetings found."
        )

    # ========================================================
    # STEP 24 — STRUCTURED RESEARCH CHECKLIST
    # ========================================================

    st.divider()

    st.subheader(
        "12. Pre-Match Research Checklist"
    )

    st.caption(
        "Use this as a manual confirmation checklist. "
        "Checking a box does not make a market safer or more likely."
    )

    checklist_col1, checklist_col2 = st.columns(2)

    with checklist_col1:

        st.markdown(
            "**Statistical Research**"
        )

        check_home_away = st.checkbox(
            "Home/away sample checked",
            key="check_home_away"
        )

        check_recent = st.checkbox(
            "Recent form checked",
            key="check_recent"
        )

        check_attack_defence = st.checkbox(
            "Attack vs defence checked",
            key="check_attack_defence"
        )

        check_line_ladder = st.checkbox(
            "Line ladder checked",
            key="check_line_ladder"
        )

        check_match_history = st.checkbox(
            "Match-by-match history checked",
            key="check_match_history"
        )

        check_h2h = st.checkbox(
            "H2H checked",
            key="check_h2h"
        )

        check_opponent_context = st.checkbox(
            "Opponent context checked",
            key="check_opponent_context"
        )

    with checklist_col2:

        st.markdown(
            "**Match Context**"
        )

        check_lineups = st.checkbox(
            "Lineups checked",
            key="check_lineups"
        )

        check_injuries = st.checkbox(
            "Injuries checked",
            key="check_injuries"
        )

        check_suspensions = st.checkbox(
            "Suspensions checked",
            key="check_suspensions"
        )

        check_tactics = st.checkbox(
            "Tactical setup checked",
            key="check_tactics"
        )

        check_match_context = st.checkbox(
            "Match context / motivation checked",
            key="check_match_context"
        )

        check_referee = st.checkbox(
            "Referee checked",
            key="check_referee"
        )

        check_market = st.checkbox(
            "Market line and odds recorded",
            key="check_market"
        )

    checklist_items = [
        check_home_away,
        check_recent,
        check_attack_defence,
        check_line_ladder,
        check_match_history,
        check_h2h,
        check_opponent_context,
        check_lineups,
        check_injuries,
        check_suspensions,
        check_tactics,
        check_match_context,
        check_referee,
        check_market
    ]

    completed = sum(
        checklist_items
    )

    total_checks = len(
        checklist_items
    )

    st.progress(
        completed / total_checks
    )

    st.metric(
        "Research checklist completed",
        f"{completed}/{total_checks}"
    )

    # ========================================================
    # RESEARCH NOTES
    # ========================================================

    st.divider()

    st.subheader(
        "13. Research Notes"
    )

    st.text_area(
        "Notes for this fixture",
        placeholder=(
            "Record lineup information, injuries, suspensions, "
            "referee notes, tactical observations, team news, "
            "market observations, opponent context, or anything "
            "else relevant to your research."
        ),
        height=180,
        key="match_research_notes"
    )

    # ========================================================
    # FINAL RESEARCH SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "14. Match Research Summary"
    )

    summary_rows = [
        {
            "Area":
                "Home Attack — Shots",
            "Team":
                home_team,
            "Context":
                "Home",
            "Value":
                fmt_number(
                    average_value(
                        home_matches,
                        "team_shots"
                    )
                )
        },
        {
            "Area":
                "Home Attack — SOT",
            "Team":
                home_team,
            "Context":
                "Home",
            "Value":
                fmt_number(
                    average_value(
                        home_matches,
                        "team_sot"
                    )
                )
        },
        {
            "Area":
                "Away Defence — SOT Conceded",
            "Team":
                away_team,
            "Context":
                "Away",
            "Value":
                fmt_number(
                    average_value(
                        away_matches,
                        "opp_sot"
                    )
                )
        },
        {
            "Area":
                "Away Attack — Shots",
            "Team":
                away_team,
            "Context":
                "Away",
            "Value":
                fmt_number(
                    average_value(
                        away_matches,
                        "team_shots"
                    )
                )
        },
        {
            "Area":
                "Away Attack — SOT",
            "Team":
                away_team,
            "Context":
                "Away",
            "Value":
                fmt_number(
                    average_value(
                        away_matches,
                        "team_sot"
                    )
                )
        },
        {
            "Area":
                "Home Defence — SOT Conceded",
            "Team":
                home_team,
            "Context":
                "Home",
            "Value":
                fmt_number(
                    average_value(
                        home_matches,
                        "opp_sot"
                    )
                )
        }
    ]

    st.dataframe(
        pd.DataFrame(
            summary_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Historical statistics describe previous matches. "
        "They are not probability forecasts."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "V1 deliberately avoids automatic 'safe bet' labels. "
    "The goal is to improve research quality and decision-making."
)
