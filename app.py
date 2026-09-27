import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import hashlib
from datetime import datetime

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Betting Research V1",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Betting Research Tool — V1")
st.caption(
    "Research assistant, not a prediction engine. "
    "Use match-by-match evidence, contextual splits and data quality checks."
)

MASTER_FILE = "football_master_2024_27_v1.csv"
WATCHLIST_FILE = "market_watchlist.csv"
ENRICHED_FILE = "football_enriched_master.csv"

# ============================================================
# REQUIRED MASTER SCHEMA
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

NUMERIC_MASTER_COLUMNS = [
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
# STEP 83 — DATA SOURCE REGISTRY
# ============================================================

DATA_SOURCE_REGISTRY = {
    "Master CSV": {
        "description": "Primary match-level dataset",
        "priority": 1,
        "type": "Core"
    },
    "StatsBomb": {
        "description": "Event, lineup and contextual football data",
        "priority": 2,
        "type": "Enrichment"
    },
    "Understat": {
        "description": "Shot-level and xG-oriented enrichment",
        "priority": 2,
        "type": "Enrichment"
    },
    "Other CSV": {
        "description": "User-provided compatible enrichment dataset",
        "priority": 3,
        "type": "Enrichment"
    }
}

# ============================================================
# STEP 84 — TEAM NAME NORMALIZATION
# ============================================================

TEAM_NAME_ALIASES = {
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united": "Manchester United",

    "man city": "Manchester City",
    "man city fc": "Manchester City",
    "manchester city fc": "Manchester City",

    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",

    "newcastle utd": "Newcastle United",
    "newcastle united": "Newcastle United",

    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",

    "brighton": "Brighton & Hove Albion",
    "brighton and hove albion": "Brighton & Hove Albion",

    "west ham utd": "West Ham United",
    "west ham united": "West Ham United",

    "nottingham forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",

    "psv": "PSV Eindhoven",
    "psv eindhoven": "PSV Eindhoven",

    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "inter milan": "Inter Milan",

    "ac milan": "AC Milan",

    "bayern": "Bayern Munich",
    "bayern munich": "Bayern Munich",

    "dortmund": "Borussia Dortmund",
    "borussia dortmund": "Borussia Dortmund",

    "atletico madrid": "Atletico Madrid",
    "atl. madrid": "Atletico Madrid",

    "barca": "Barcelona",
    "fc barcelona": "Barcelona",
    "barcelona": "Barcelona",

    "real madrid cf": "Real Madrid",
    "real madrid": "Real Madrid"
}


def normalize_team_name(name):
    """Normalize common football team naming differences."""

    if pd.isna(name):
        return ""

    value = str(name).strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    key = value.lower()

    if key in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[key]

    return value


# ============================================================
# STEP 85 — MATCH KEY GENERATION
# ============================================================

def normalize_date_value(value):
    """Convert a date into YYYY-MM-DD."""

    parsed = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%Y-%m-%d")


def make_match_key(
    date_value,
    home_team,
    away_team
):
    """
    Stable match identifier.

    Order is intentionally preserved:
    home team + away team + date.
    """

    date_part = normalize_date_value(
        date_value
    )

    home = normalize_team_name(
        home_team
    ).lower()

    away = normalize_team_name(
        away_team
    ).lower()

    raw = (
        f"{date_part}|{home}|{away}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]


def add_match_keys(df):
    """Add a stable match_key column."""

    df = df.copy()

    if not all(
        c in df.columns
        for c in [
            "date",
            "home_team",
            "away_team"
        ]
    ):
        return df

    df["match_key"] = df.apply(
        lambda row: make_match_key(
            row["date"],
            row["home_team"],
            row["away_team"]
        ),
        axis=1
    )

    return df


# ============================================================
# STEP 86 — DUPLICATE DETECTION
# ============================================================

def duplicate_report(df):
    """Identify duplicate match records."""

    if df is None or df.empty:
        return {
            "rows": 0,
            "duplicate_rows": 0,
            "unique_matches": 0
        }

    temp = add_match_keys(
        df
    )

    duplicate_mask = temp[
        "match_key"
    ].duplicated(
        keep=False
    )

    return {
        "rows": len(temp),
        "duplicate_rows": int(
            duplicate_mask.sum()
        ),
        "unique_matches": int(
            temp["match_key"].nunique()
        )
    }


# ============================================================
# STEP 87 — SCHEMA COMPATIBILITY
# ============================================================

ENRICHMENT_ALIASES = {
    "game_date": "date",
    "match_date": "date",

    "home": "home_team",
    "home_name": "home_team",
    "home_team_name": "home_team",

    "away": "away_team",
    "away_name": "away_team",
    "away_team_name": "away_team",

    "xg_home": "home_xg",
    "xg_away": "away_xg",

    "home_expected_goals": "home_xg",
    "away_expected_goals": "away_xg"
}


def standardize_external_columns(df):
    """Standardize common external dataset column names."""

    df = df.copy()

    renamed = {}

    for column in df.columns:

        clean = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        clean = re.sub(
            r"_+",
            "_",
            clean
        )

        if clean in ENRICHMENT_ALIASES:
            renamed[column] = (
                ENRICHMENT_ALIASES[clean]
            )
        else:
            renamed[column] = clean

    df = df.rename(
        columns=renamed
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    for c in [
        "home_team",
        "away_team"
    ]:

        if c in df.columns:
            df[c] = df[c].apply(
                normalize_team_name
            )

    return df


def compatibility_check(df):
    """Check whether an external dataset can be matched safely."""

    if df is None or df.empty:
        return {
            "compatible": False,
            "reason": "Dataset is empty.",
            "score": 0
        }

    columns = set(
        df.columns
    )

    required_match_fields = {
        "date",
        "home_team",
        "away_team"
    }

    available = (
        required_match_fields
        &
        columns
    )

    score = len(available) / 3

    if available == required_match_fields:
        return {
            "compatible": True,
            "reason": "Contains date, home_team and away_team.",
            "score": score
        }

    missing = (
        required_match_fields
        -
        available
    )

    return {
        "compatible": False,
        "reason": (
            "Missing match-identification fields: "
            + ", ".join(sorted(missing))
        ),
        "score": score
    }


# ============================================================
# CLEAN MASTER DATA
# ============================================================

def clean_data(df):
    """Standardise and clean the football dataset."""

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
                .replace(
                    "nan",
                    ""
                )
            )

    numeric_columns = [
        c
        for c in REQUIRED_COLUMNS
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

    df["home_team"] = df[
        "home_team"
    ].apply(
        normalize_team_name
    )

    df["away_team"] = df[
        "away_team"
    ].apply(
        normalize_team_name
    )

    if "date" in df.columns:

        df = df.sort_values(
            "date",
            ascending=False
        )

    return df


# ============================================================
# TEAM MATCH ENGINE
# ============================================================

def team_matches(
    df,
    team
):
    """Convert raw matches into selected team's perspective."""

    h = df[
        df["home_team"].eq(team)
    ].copy()

    h["team_goals"] = h[
        "home_goals"
    ]

    h["opp_goals"] = h[
        "away_goals"
    ]

    h["team_shots"] = h[
        "home_shots"
    ]

    h["opp_shots"] = h[
        "away_shots"
    ]

    h["team_sot"] = h[
        "home_sot"
    ]

    h["opp_sot"] = h[
        "away_sot"
    ]

    h["team_corners"] = h[
        "home_corners"
    ]

    h["opp_corners"] = h[
        "away_corners"
    ]

    h["venue"] = "Home"

    a = df[
        df["away_team"].eq(team)
    ].copy()

    a["team_goals"] = a[
        "away_goals"
    ]

    a["opp_goals"] = a[
        "home_goals"
    ]

    a["team_shots"] = a[
        "away_shots"
    ]

    a["opp_shots"] = a[
        "home_shots"
    ]

    a["team_sot"] = a[
        "away_sot"
    ]

    a["opp_sot"] = a[
        "home_sot"
    ]

    a["team_corners"] = a[
        "away_corners"
    ]

    a["opp_corners"] = a[
        "home_corners"
    ]

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
            m["competition"]
            == competition
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

    if (
        column is None
        or column not in df.columns
    ):
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

    size = int(
        len(values)
    )

    return {
        "hit_rate": hits / size,
        "hits": hits,
        "misses": size - hits,
        "sample_size": size
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

    if (
        column is None
        or column not in df.columns
    ):
        return pd.DataFrame()

    columns = [
        "date",
        "home_team",
        "away_team",
        "venue",
        column
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    result = df[
        columns
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
        ].map({
            True: "✓",
            False: "✗"
        })

    return result


def fmt_pct(value):

    if value is None:
        return "—"

    try:

        if pd.isna(value):
            return "—"

    except Exception:
        pass

    return f"{value * 100:.1f}%"


# ============================================================
# SAMPLE QUALITY
# ============================================================

def sample_quality(
    sample_size
):

    if sample_size == 0:

        return (
            "No data",
            "No matches are available."
        )

    if sample_size < 5:

        return (
            "Very small sample",
            "Fewer than 5 matches."
        )

    if sample_size < 10:

        return (
            "Small sample",
            "A limited historical sample."
        )

    if sample_size < 15:

        return (
            "Reasonable sample",
            "A useful historical sample."
        )

    return (
        "Strong sample",
        "15 or more matches."
    )


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================

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


def team_summary(
    df
):

    result = {
        "Matches": len(df)
    }

    for label, column in [
        ("Shots", "team_shots"),
        ("Shots on Target", "team_sot"),
        ("Corners", "team_corners"),
        ("Goals", "team_goals")
    ]:

        result[
            f"{label} Avg"
        ] = average_value(
            df,
            column
        )

    return result


def match_team_summary(
    matches
):

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
# FILTER LISTS
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
# H2H
# ============================================================

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

    home = analyse_market(
        home_matches,
        market,
        direction,
        line
    )

    away = analyse_market(
        away_matches,
        market,
        direction,
        line
    )

    return pd.DataFrame([
        {
            "Team": "Home",
            "Matches": home[
                "sample_size"
            ],
            "Hits": home[
                "hits"
            ],
            "Misses": home[
                "misses"
            ],
            "Hit Rate": fmt_pct(
                home[
                    "hit_rate"
                ]
            )
        },
        {
            "Team": "Away",
            "Matches": away[
                "sample_size"
            ],
            "Hits": away[
                "hits"
            ],
            "Misses": away[
                "misses"
            ],
            "Hit Rate": fmt_pct(
                away[
                    "hit_rate"
                ]
            )
        }
    ])


# ============================================================
# STEP 88 — ENRICHMENT PREVIEW
# ============================================================

def build_enrichment_preview(
    master,
    external
):

    master_keys = set(
        add_match_keys(
            master
        )["match_key"]
    )

    external_keyed = add_match_keys(
        external
    )

    external_keys = set(
        external_keyed[
            "match_key"
        ]
    )

    matched = (
        master_keys
        &
        external_keys
    )

    external_only = (
        external_keys
        -
        master_keys
    )

    master_only = (
        master_keys
        -
        external_keys
    )

    return {
        "master_rows": len(master),
        "external_rows": len(external),
        "master_unique": len(master_keys),
        "external_unique": len(external_keys),
        "matched": len(matched),
        "external_only": len(external_only),
        "master_only": len(master_only)
    }


# ============================================================
# STEP 89 — SAFE ENRICHMENT MERGE
# ============================================================

def safe_enrichment_merge(
    master,
    external,
    source_name
):
    """
    Add external fields without replacing core master statistics.

    Master columns always win when a column exists in both datasets.
    """

    master_work = add_match_keys(
        master.copy()
    )

    external_work = standardize_external_columns(
        external.copy()
    )

    external_work = add_match_keys(
        external_work
    )

    # Remove exact duplicate external match keys
    external_work = external_work.drop_duplicates(
        subset=["match_key"],
        keep="first"
    )

    # Never overwrite master core fields
    protected = set(
        REQUIRED_COLUMNS
    )

    enrichment_columns = [
        c
        for c in external_work.columns
        if c not in [
            "match_key",
            "date",
            "home_team",
            "away_team"
        ]
        and c not in protected
    ]

    if not enrichment_columns:

        return (
            master_work,
            [],
            0
        )

    external_subset = external_work[
        [
            "match_key"
        ]
        +
        enrichment_columns
    ].copy()

    # Prefix enrichment columns to make their origin explicit
    rename_map = {}

    for c in enrichment_columns:

        clean_source = (
            source_name
            .lower()
            .replace(" ", "_")
        )

        rename_map[c] = (
            f"{clean_source}_{c}"
        )

    external_subset = external_subset.rename(
        columns=rename_map
    )

    merged = master_work.merge(
        external_subset,
        on="match_key",
        how="left"
    )

    matched_rows = int(
        merged[
            list(rename_map.values())[0]
        ].notna().sum()
    ) if rename_map else 0

    return (
        merged,
        list(rename_map.values()),
        matched_rows
    )


# ============================================================
# STEP 90 — DATA PROVENANCE
# ============================================================

def add_provenance_columns(
    df,
    source_name
):

    df = df.copy()

    if "data_source" not in df.columns:

        df["data_source"] = (
            source_name
        )

    if "data_loaded_at" not in df.columns:

        df["data_loaded_at"] = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    return df


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
# LOAD PRIMARY DATA
# ============================================================

st.sidebar.header("1. Primary Data")

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

        primary_source = (
            f"Uploaded: {uploaded.name}"
        )

        st.sidebar.success(
            f"Loaded {len(data):,} matches"
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

            primary_source = (
                MASTER_FILE
            )

            st.sidebar.success(
                f"Primary dataset: {len(data):,} matches"
            )

        except Exception as exc:

            st.error(
                f"Could not read {MASTER_FILE}: {exc}"
            )

            st.stop()

    else:

        st.warning(
            f"{MASTER_FILE} was not found. "
            "Demo data is being used temporarily."
        )

        data = clean_data(
            demo
        )

        primary_source = (
            "Built-in demo dataset"
        )

        st.sidebar.info(
            f"Demo dataset: {len(data)} matches"
        )


# ============================================================
# MASTER VALIDATION
# ============================================================

missing = [
    c
    for c in REQUIRED_COLUMNS
    if c not in data.columns
]

if missing:

    st.error(
        "Primary dataset is missing required columns: "
        + ", ".join(missing)
    )

    st.stop()


data = add_provenance_columns(
    data,
    "Master CSV"
)

data = add_match_keys(
    data
)


# ============================================================
# DATA QUALITY PANEL
# ============================================================

with st.sidebar.expander(
    "Primary Dataset Health",
    expanded=False
):

    quality = duplicate_report(
        data
    )

    st.write(
        f"Rows: **{quality['rows']:,}**"
    )

    st.write(
        f"Unique match keys: **{quality['unique_matches']:,}**"
    )

    st.write(
        f"Duplicate rows: **{quality['duplicate_rows']:,}**"
    )

    missing_values = int(
        data[
            REQUIRED_COLUMNS
        ].isna().sum().sum()
    )

    st.write(
        f"Missing required values: **{missing_values:,}**"
    )


# ============================================================
# TABS
# ============================================================

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7
) = st.tabs(
    [
        "🔎 Team Research",
        "🎯 Market Tester",
        "📊 Bookmaker Monitor",
        "📥 Data Format",
        "🔬 Match Research",
        "🧠 Advanced Research",
        "🗄️ Data Integration"
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

    label, message = sample_quality(
        len(dashboard_df)
    )

    if len(dashboard_df) < 5:

        st.warning(
            f"⚠️ {label}: {message}"
        )

    elif len(dashboard_df) < 10:

        st.info(
            f"ℹ️ {label}: {message}"
        )

    else:

        st.success(
            f"✓ {label}: {message}"
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Matches",
        summary["Matches"]
    )

    c2.metric(
        "Avg Shots",
        (
            f"{summary['Shots Avg']:.2f}"
            if summary["Shots Avg"]
            is not None
            else "—"
        )
    )

    c3.metric(
        "Avg SOT",
        (
            f"{summary['Shots on Target Avg']:.2f}"
            if summary["Shots on Target Avg"]
            is not None
            else "—"
        )
    )

    c4.metric(
        "Avg Corners",
        (
            f"{summary['Corners Avg']:.2f}"
            if summary["Corners Avg"]
            is not None
            else "—"
        )
    )

    c5.metric(
        "Avg Goals",
        (
            f"{summary['Goals Avg']:.2f}"
            if summary["Goals Avg"]
            is not None
            else "—"
        )
    )

    st.divider()

    st.subheader(
        "Market Hit-Rate Overview"
    )

    overview = []

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

                if result[
                    "sample_size"
                ]:

                    overview.append(
                        {
                            "Market": market_name,
                            "Direction": direction_name,
                            "Line": test_line,
                            "Hits": result["hits"],
                            "Sample": result["sample_size"],
                            "Hit Rate": fmt_pct(
                                result["hit_rate"]
                            )
                        }
                    )

    if overview:

        st.dataframe(
            pd.DataFrame(
                overview
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
                "Matches": len(split_df),
                "Avg Shots": (
                    f"{split['Shots Avg']:.2f}"
                    if split["Shots Avg"] is not None
                    else "—"
                ),
                "Avg SOT": (
                    f"{split['Shots on Target Avg']:.2f}"
                    if split["Shots on Target Avg"] is not None
                    else "—"
                ),
                "Avg Corners": (
                    f"{split['Corners Avg']:.2f}"
                    if split["Corners Avg"] is not None
                    else "—"
                ),
                "Avg Goals": (
                    f"{split['Goals Avg']:.2f}"
                    if split["Goals Avg"] is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            split_rows
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

    available = [
        c
        for c in display_columns
        if c in dashboard_df.columns
    ]

    st.dataframe(
        dashboard_df[
            available
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
        "Team",
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
        list(
            MARKET_COLUMN_MAP.keys()
        ),
        key="market_type"
    )

    direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="market_direction"
    )

    line = st.number_input(
        "Line",
        0.0,
        30.0,
        3.5,
        0.5,
        key="market_line"
    )

    odds = st.number_input(
        "Decimal odds",
        1.01,
        100.0,
        1.30,
        0.01,
        key="market_odds"
    )

    n2 = st.select_slider(
        "Recent sample",
        [5, 10, 15],
        10,
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

    rate = analysis[
        "hit_rate"
    ]

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
        "Sample",
        analysis["sample_size"]
    )

    if rate is not None:

        difference = (
            rate
            -
            breakeven
        )

        d.metric(
            "Historical difference",
            f"{difference * 100:+.1f} pp"
        )

        st.progress(
            min(
                max(rate, 0),
                1
            )
        )

    st.caption(
        "Historical performance is descriptive. "
        "It is not a forecast of the next match."
    )

    st.divider()

    st.subheader(
        "Market History"
    )

    history = market_history(
        m,
        market,
        direction,
        line
    )

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Overall / Home / Away"
    )

    rows = []

    for v in [
        "All",
        "Home",
        "Away"
    ]:

        sample_df = filtered_team_matches(
            data,
            team,
            season2,
            competition2,
            v,
            n2
        )

        result = analyse_market(
            sample_df,
            market,
            direction,
            line
        )

        rows.append(
            {
                "Venue": v,
                "Matches": result[
                    "sample_size"
                ],
                "Hits": result[
                    "hits"
                ],
                "Misses": result[
                    "misses"
                ],
                "Hit Rate": fmt_pct(
                    result["hit_rate"]
                )
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
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

                if "Status" not in loaded:
                    loaded["Status"] = "Watching"

                if "Actual Result" not in loaded:
                    loaded["Actual Result"] = ""

                st.session_state[
                    "market_watchlist"
                ] = (
                    loaded
                    .fillna("")
                    .to_dict("records")
                )

            except Exception:

                st.session_state[
                    "market_watchlist"
                ] = []

        else:

            st.session_state[
                "market_watchlist"
            ] = []

    b1, b2 = st.columns(2)

    with b1:

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
            list(
                MARKET_COLUMN_MAP.keys()
            ),
            key="monitor_market"
        )

    with b2:

        monitor_direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="monitor_direction"
        )

        monitor_line = st.number_input(
            "Line",
            0.0,
            30.0,
            3.5,
            0.5,
            key="monitor_line"
        )

        monitor_odds = st.number_input(
            "Decimal odds",
            1.01,
            100.0,
            1.30,
            0.01,
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

        result = analyse_market(
            monitor_df,
            monitor_market,
            monitor_direction,
            monitor_line
        )

        rate = result[
            "hit_rate"
        ]

        break_even = (
            1 / monitor_odds
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
            "Historical Hit Rate": (
                fmt_pct(rate)
            ),
            "Break-even": (
                f"{break_even * 100:.1f}%"
            ),
            "Edge vs Break-even": (
                f"{(rate - break_even) * 100:+.1f} pp"
                if rate is not None
                else "—"
            ),
            "Status": "Watching",
            "Actual Result": ""
        }

        st.session_state[
            "market_watchlist"
        ].append(
            entry
        )

        pd.DataFrame(
            st.session_state[
                "market_watchlist"
            ]
        ).to_csv(
            WATCHLIST_FILE,
            index=False
        )

        st.success(
            "Market added."
        )

    st.divider()

    watchlist = st.session_state[
        "market_watchlist"
    ]

    if watchlist:

        watch_df = pd.DataFrame(
            watchlist
        )

        total = len(
            watch_df
        )

        watching = int(
            (
                watch_df["Status"]
                == "Watching"
            ).sum()
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

        settled = won + lost

        hit_rate_display = (
            f"{won / settled * 100:.1f}%"
            if settled
            else "—"
        )

        x1, x2, x3, x4, x5 = st.columns(5)

        x1.metric(
            "Total",
            total
        )

        x2.metric(
            "Watching",
            watching
        )

        x3.metric(
            "Won",
            won
        )

        x4.metric(
            "Lost",
            lost
        )

        x5.metric(
            "Tracked Hit Rate",
            hit_rate_display
        )

        st.dataframe(
            watch_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Watchlist CSV",
            watch_df.to_csv(
                index=False
            ).encode("utf-8"),
            "market_watchlist.csv",
            "text/csv"
        )

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
        "The primary dataset should contain one row per match."
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

    st.download_button(
        "Download Demo CSV",
        demo.to_csv(
            index=False
        ).encode("utf-8"),
        "football_demo.csv",
        "text/csv"
    )

    st.divider()

    st.subheader(
        "Primary Dataset Information"
    )

    info = pd.DataFrame([
        {
            "Property": "Primary file",
            "Value": MASTER_FILE
        },
        {
            "Property": "Loaded source",
            "Value": primary_source
        },
        {
            "Property": "Rows",
            "Value": f"{len(data):,}"
        },
        {
            "Property": "Columns",
            "Value": len(data.columns)
        },
        {
            "Property": "Teams",
            "Value": len(team_list(data))
        },
        {
            "Property": "Competitions",
            "Value": data[
                "competition"
            ].nunique()
        },
        {
            "Property": "Seasons",
            "Value": data[
                "season"
            ].nunique()
        }
    ])

    st.dataframe(
        info,
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
        "Compare a specific home/away fixture using historical "
        "match-by-match evidence."
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
        "Home vs Away Comparison"
    )

    hs = match_team_summary(
        home_matches
    )

    aws = match_team_summary(
        away_matches
    )

    comparison = pd.DataFrame([
        {
            "Metric": "Matches",
            "Home Team": hs["Matches"],
            "Away Team": aws["Matches"]
        },
        {
            "Metric": "Avg Shots",
            "Home Team": (
                f"{hs['Shots']:.2f}"
                if hs["Shots"] is not None
                else "—"
            ),
            "Away Team": (
                f"{aws['Shots']:.2f}"
                if aws["Shots"] is not None
                else "—"
            )
        },
        {
            "Metric": "Avg SOT",
            "Home Team": (
                f"{hs['SOT']:.2f}"
                if hs["SOT"] is not None
                else "—"
            ),
            "Away Team": (
                f"{aws['SOT']:.2f}"
                if aws["SOT"] is not None
                else "—"
            )
        },
        {
            "Metric": "Avg Corners",
            "Home Team": (
                f"{hs['Corners']:.2f}"
                if hs["Corners"] is not None
                else "—"
            ),
            "Away Team": (
                f"{aws['Corners']:.2f}"
                if aws["Corners"] is not None
                else "—"
            )
        },
        {
            "Metric": "Avg Goals",
            "Home Team": (
                f"{hs['Goals']:.2f}"
                if hs["Goals"] is not None
                else "—"
            ),
            "Away Team": (
                f"{aws['Goals']:.2f}"
                if aws["Goals"] is not None
                else "—"
            )
        }
    ])

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Attack vs Defence"
    )

    attack_defence_rows = []

    metric_map = [
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

    for label, team_col, opp_col in metric_map:

        attack_defence_rows.append(
            {
                "Metric": label,
                "Home Attack": (
                    f"{average_value(home_matches, team_col):.2f}"
                    if average_value(home_matches, team_col)
                    is not None
                    else "—"
                ),
                "Away Defence Conceded": (
                    f"{average_value(away_matches, opp_col):.2f}"
                    if average_value(away_matches, opp_col)
                    is not None
                    else "—"
                ),
                "Away Attack": (
                    f"{average_value(away_matches, team_col):.2f}"
                    if average_value(away_matches, team_col)
                    is not None
                    else "—"
                ),
                "Home Defence Conceded": (
                    f"{average_value(home_matches, opp_col):.2f}"
                    if average_value(home_matches, opp_col)
                    is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            attack_defence_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Market Comparison"
    )

    mc1, mc2, mc3 = st.columns(3)

    with mc1:

        research_market = st.selectbox(
            "Market",
            list(
                MARKET_COLUMN_MAP.keys()
            ),
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
            0.0,
            30.0,
            3.5,
            0.5,
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

        h2h_cols = [
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

        h2h_cols = [
            c
            for c in h2h_cols
            if c in h2h.columns
        ]

        st.dataframe(
            h2h[
                h2h_cols
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No H2H matches found for the selected filters."
        )

    st.divider()

    st.subheader(
        "Research Notes"
    )

    st.text_area(
        "Fixture notes",
        placeholder=(
            "Lineups, injuries, tactical observations, "
            "game state, team news, referee information, "
            "market observations..."
        ),
        height=180,
        key="match_research_notes"
    )


# ============================================================
# TAB 6 — ADVANCED RESEARCH
# ============================================================

with tab6:

    st.subheader(
        "🧠 Advanced Research"
    )

    st.caption(
        "Contextual filters designed to reduce reliance on simple "
        "10-match averages."
    )

    advanced_team = st.selectbox(
        "Team",
        team_list(data),
        key="advanced_team"
    )

    advanced_sample = st.selectbox(
        "Base sample",
        [5, 10, 15],
        index=2,
        key="advanced_sample"
    )

    base = filtered_team_matches(
        data,
        advanced_team,
        "All",
        "All",
        "All",
        advanced_sample
    )

    st.subheader(
        "Recent Production"
    )

    adv_rows = []

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

        produced = average_value(
            base,
            team_col
        )

        conceded = average_value(
            base,
            opp_col
        )

        adv_rows.append(
            {
                "Metric": label,
                "Produced Avg": (
                    f"{produced:.2f}"
                    if produced is not None
                    else "—"
                ),
                "Conceded Avg": (
                    f"{conceded:.2f}"
                    if conceded is not None
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            adv_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Market Sensitivity"
    )

    sensitivity_market = st.selectbox(
        "Market",
        list(
            MARKET_COLUMN_MAP.keys()
        ),
        key="sensitivity_market"
    )

    sensitivity_direction = st.selectbox(
        "Direction",
        ["Over", "Under"],
        key="sensitivity_direction"
    )

    sensitivity_lines = st.multiselect(
        "Lines to test",
        [
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
        ],
        default=[
            2.5,
            3.5,
            4.5
        ],
        key="sensitivity_lines"
    )

    sensitivity_rows = []

    for test_line in sensitivity_lines:

        result = analyse_market(
            base,
            sensitivity_market,
            sensitivity_direction,
            test_line
        )

        sensitivity_rows.append(
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

    if sensitivity_rows:

        st.dataframe(
            pd.DataFrame(
                sensitivity_rows
            ),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Home / Away Stability"
    )

    home_sample = filtered_team_matches(
        data,
        advanced_team,
        "All",
        "All",
        "Home",
        advanced_sample
    )

    away_sample = filtered_team_matches(
        data,
        advanced_team,
        "All",
        "All",
        "Away",
        advanced_sample
    )

    stability_rows = []

    for market_name, column in MARKET_COLUMN_MAP.items():

        home_avg = average_value(
            home_sample,
            column
        )

        away_avg = average_value(
            away_sample,
            column
        )

        stability_rows.append(
            {
                "Market": market_name,
                "Home Avg": (
                    f"{home_avg:.2f}"
                    if home_avg is not None
                    else "—"
                ),
                "Away Avg": (
                    f"{away_avg:.2f}"
                    if away_avg is not None
                    else "—"
                ),
                "Difference": (
                    f"{home_avg - away_avg:+.2f}"
                    if (
                        home_avg is not None
                        and away_avg is not None
                    )
                    else "—"
                )
            }
        )

    st.dataframe(
        pd.DataFrame(
            stability_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Use contextual differences as evidence to investigate, "
        "not as automatic betting signals."
    )


# ============================================================
# TAB 7 — DATA INTEGRATION
# ============================================================

with tab7:

    st.subheader(
        "🗄️ Data Integration — Steps 83–90"
    )

    st.caption(
        "External datasets are treated as enrichment. "
        f"{MASTER_FILE} remains the primary dataset."
    )

    # --------------------------------------------------------
    # STEP 83 — SOURCE REGISTRY
    # --------------------------------------------------------

    st.subheader(
        "Step 83 — Data Source Registry"
    )

    registry_rows = []

    for source, details in DATA_SOURCE_REGISTRY.items():

        registry_rows.append(
            {
                "Source": source,
                "Type": details["type"],
                "Priority": details["priority"],
                "Purpose": details["description"]
            }
        )

    st.dataframe(
        pd.DataFrame(
            registry_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # PRIMARY DATA STATUS
    # --------------------------------------------------------

    st.subheader(
        "Primary Dataset Status"
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Primary Rows",
        f"{len(data):,}"
    )

    p2.metric(
        "Teams",
        f"{len(team_list(data)):,}"
    )

    p3.metric(
        "Competitions",
        f"{data['competition'].nunique():,}"
    )

    p4.metric(
        "Unique Match Keys",
        f"{data['match_key'].nunique():,}"
    )

    st.success(
        f"Primary source remains: {primary_source}"
    )

    # --------------------------------------------------------
    # STEP 84 — NORMALIZATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Step 84 — Team Name Normalization"
    )

    st.write(
        "The integration layer normalizes common naming differences "
        "before attempting to match datasets."
    )

    normalization_rows = []

    for original, normalized in list(
        TEAM_NAME_ALIASES.items()
    )[:25]:

        normalization_rows.append(
            {
                "External/Common Name": original,
                "Normalized Name": normalized
            }
        )

    st.dataframe(
        pd.DataFrame(
            normalization_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # EXTERNAL UPLOAD
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "External Enrichment Dataset"
    )

    external_source = st.selectbox(
        "External source",
        [
            "StatsBomb",
            "Understat",
            "Other CSV"
        ],
        key="external_source"
    )

    external_file = st.file_uploader(
        "Upload external CSV for compatibility testing",
        type=["csv"],
        key="external_csv"
    )

    if external_file is not None:

        try:

            raw_external = pd.read_csv(
                external_file
            )

            external = standardize_external_columns(
                raw_external
            )

            st.success(
                f"Loaded external dataset: "
                f"{len(external):,} rows"
            )

            # ------------------------------------------------
            # STEP 87 — SCHEMA CHECK
            # ------------------------------------------------

            st.subheader(
                "Step 87 — Dataset Compatibility"
            )

            compatibility = compatibility_check(
                external
            )

            score = compatibility[
                "score"
            ]

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Compatibility",
                f"{score * 100:.0f}%"
            )

            s2.metric(
                "Rows",
                f"{len(external):,}"
            )

            s3.metric(
                "Columns",
                f"{len(external.columns):,}"
            )

            if compatibility[
                "compatible"
            ]:

                st.success(
                    compatibility["reason"]
                )

            else:

                st.error(
                    compatibility["reason"]
                )

            st.write(
                "Detected columns:"
            )

            st.code(
                ", ".join(
                    external.columns
                )
            )

            # ------------------------------------------------
            # STEP 85 — MATCH KEYS
            # ------------------------------------------------

            st.subheader(
                "Step 85 — Match-Key Generation"
            )

            if compatibility[
                "compatible"
            ]:

                external_keyed = add_match_keys(
                    external
                )

                st.write(
                    "Example generated keys:"
                )

                st.dataframe(
                    external_keyed[
                        [
                            "date",
                            "home_team",
                            "away_team",
                            "match_key"
                        ]
                    ].head(10),
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # STEP 86 — DUPLICATES
            # ------------------------------------------------

            st.subheader(
                "Step 86 — Duplicate Detection"
            )

            duplicate_info = duplicate_report(
                external
            )

            d1, d2, d3 = st.columns(3)

            d1.metric(
                "Rows",
                f"{duplicate_info['rows']:,}"
            )

            d2.metric(
                "Unique Matches",
                f"{duplicate_info['unique_matches']:,}"
            )

            d3.metric(
                "Duplicate Rows",
                f"{duplicate_info['duplicate_rows']:,}"
            )

            if duplicate_info[
                "duplicate_rows"
            ]:

                st.warning(
                    "Duplicate match keys were detected. "
                    "The safe merge will retain only one external "
                    "record per match key."
                )

            else:

                st.success(
                    "No duplicate match keys detected."
                )

            # ------------------------------------------------
            # STEP 88 — ENRICHMENT PREVIEW
            # ------------------------------------------------

            st.subheader(
                "Step 88 — Enrichment Preview"
            )

            if compatibility[
                "compatible"
            ]:

                preview = build_enrichment_preview(
                    data,
                    external
                )

                e1, e2, e3, e4, e5 = st.columns(5)

                e1.metric(
                    "Master",
                    f"{preview['master_rows']:,}"
                )

                e2.metric(
                    "External",
                    f"{preview['external_rows']:,}"
                )

                e3.metric(
                    "Matched",
                    f"{preview['matched']:,}"
                )

                e4.metric(
                    "External Only",
                    f"{preview['external_only']:,}"
                )

                e5.metric(
                    "Master Only",
                    f"{preview['master_only']:,}"
                )

                match_rate = (
                    preview["matched"]
                    /
                    preview["external_unique"]
                    if preview["external_unique"]
                    else 0
                )

                st.progress(
                    min(
                        max(
                            match_rate,
                            0
                        ),
                        1
                    )
                )

                st.caption(
                    f"External-to-master match rate: "
                    f"{match_rate * 100:.1f}%"
                )

            # ------------------------------------------------
            # STEP 89 — SAFE MERGE
            # ------------------------------------------------

            st.subheader(
                "Step 89 — Safe Enrichment Merge"
            )

            st.warning(
                "The merge does NOT replace master columns. "
                "External fields are prefixed with their source."
            )

            if (
                compatibility[
                    "compatible"
                ]
            ):

                if st.button(
                    "Build Enriched Dataset",
                    key="build_enriched"
                ):

                    merged, added_columns, matched_rows = (
                        safe_enrichment_merge(
                            data,
                            external,
                            external_source
                        )
                    )

                    merged = add_provenance_columns(
                        merged,
                        "Master + " + external_source
                    )

                    st.session_state[
                        "latest_enriched_data"
                    ] = merged

                    st.session_state[
                        "latest_enrichment_columns"
                    ] = added_columns

                    st.session_state[
                        "latest_matched_rows"
                    ] = matched_rows

                    st.success(
                        "Enriched dataset created in memory."
                    )

                if (
                    "latest_enriched_data"
                    in st.session_state
                ):

                    enriched = st.session_state[
                        "latest_enriched_data"
                    ]

                    added = st.session_state.get(
                        "latest_enrichment_columns",
                        []
                    )

                    matched_rows = st.session_state.get(
                        "latest_matched_rows",
                        0
                    )

                    z1, z2, z3 = st.columns(3)

                    z1.metric(
                        "Enriched Rows",
                        f"{len(enriched):,}"
                    )

                    z2.metric(
                        "New Enrichment Columns",
                        f"{len(added):,}"
                    )

                    z3.metric(
                        "Matched Rows",
                        f"{matched_rows:,}"
                    )

                    if added:

                        st.write(
                            "Added enrichment fields:"
                        )

                        st.code(
                            ", ".join(
                                added
                            )
                        )

                    else:

                        st.info(
                            "No non-core enrichment columns "
                            "were available to add."
                        )

                    st.subheader(
                        "Enriched Dataset Preview"
                    )

                    st.dataframe(
                        enriched.head(20),
                        use_container_width=True,
                        hide_index=True
                    )

                    # ----------------------------------------
                    # STEP 90 — PROVENANCE
                    # ----------------------------------------

                    st.subheader(
                        "Step 90 — Data Provenance"
                    )

                    provenance_cols = [
                        c
                        for c in [
                            "match_key",
                            "data_source",
                            "data_loaded_at"
                        ]
                        if c in enriched.columns
                    ]

                    if provenance_cols:

                        st.dataframe(
                            enriched[
                                provenance_cols
                            ].head(20),
                            use_container_width=True,
                            hide_index=True
                        )

                    # ----------------------------------------
                    # DOWNLOAD
                    # ----------------------------------------

                    enriched_csv = (
                        enriched
                        .to_csv(
                            index=False
                        )
                        .encode("utf-8")
                    )

                    st.download_button(
                        "Download Enriched Dataset",
                        enriched_csv,
                        ENRICHED_FILE,
                        "text/csv",
                        key="download_enriched"
                    )

                    st.caption(
                        "Downloading the enriched file does not "
                        "replace your primary master CSV."
                    )

            # ------------------------------------------------
            # RAW PREVIEW
            # ------------------------------------------------

            st.divider()

            with st.expander(
                "View External Dataset Preview"
            ):

                st.dataframe(
                    external.head(25),
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as exc:

            st.error(
                f"Could not process external dataset: {exc}"
            )

    else:

        st.info(
            "Upload a StatsBomb, Understat or other compatible "
            "CSV to run the Steps 83–90 integration checks."
        )

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Data Architecture"
    )

    architecture = pd.DataFrame([
        {
            "Layer": "Primary",
            "Dataset": MASTER_FILE,
            "Role": "Core match statistics",
            "Can overwrite master?": "No"
        },
        {
            "Layer": "Enrichment",
            "Dataset": "StatsBomb",
            "Role": "Events / contextual data",
            "Can overwrite master?": "No"
        },
        {
            "Layer": "Enrichment",
            "Dataset": "Understat",
            "Role": "xG / shot-level data",
            "Can overwrite master?": "No"
        },
        {
            "Layer": "Derived",
            "Dataset": ENRICHED_FILE,
            "Role": "Combined research dataset",
            "Can overwrite master?": "No"
        }
    ])

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The integration system is deliberately conservative: "
        "the original master dataset remains intact while external "
        "information is tested, matched and added as a separate layer."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Football Betting Research Tool — Steps 1–90. "
    "Historical data is descriptive and does not establish the "
    "probability of future outcomes. Primary dataset remains "
    f"{MASTER_FILE}."
)
