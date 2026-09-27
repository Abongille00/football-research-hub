
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os

st.set_page_config(page_title="Football Betting Research V1", page_icon="⚽", layout="wide")

st.title("⚽ Football Betting Research Tool — V1")
st.caption("Research assistant, not a prediction engine. Use match-by-match data to test a market before betting.")

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_goals", "away_goals",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners"
]

def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = [c for c in REQUIRED_COLUMNS if c not in ["date", "home_team", "away_team"]]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["home_team", "away_team"])

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

    out = pd.concat([h, a], ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date", ascending=False)
    return out

def hit_rate(series, line, over=True):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    hits = (s > line).sum() if over else (s < line).sum()
    return hits / len(s)

def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x*100:.1f}%"

# Demo data
demo = pd.DataFrame([
    ["2026-09-18","Chelsea","Brentford",2,0,17,7,5,2,6,3],
    ["2026-09-14","Everton","Chelsea",0,2,9,14,3,5,4,7],
    ["2026-09-07","Chelsea","Fulham",3,1,19,8,7,3,8,2],
    ["2026-08-30","Newcastle","Chelsea",1,1,12,11,4,4,5,5],
    ["2026-08-24","Chelsea","Wolves",2,1,16,10,6,3,7,4],
    ["2026-09-18","Bayern Munich","Union Berlin",3,1,20,7,8,2,8,2],
    ["2026-09-13","Mainz","Bayern Munich",0,3,6,18,2,7,3,8],
    ["2026-09-06","Bayern Munich","Freiburg",2,0,17,8,6,2,9,4],
    ["2026-08-30","Dortmund","Bayern Munich",1,2,11,13,4,5,5,6],
    ["2026-08-23","Bayern Munich","Leipzig",4,1,22,9,9,3,10,2],
], columns=REQUIRED_COLUMNS)

st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload match CSV", type=["csv"])
if uploaded:
    data = clean_data(pd.read_csv(uploaded))
    st.sidebar.success(f"Loaded {len(data)} matches")
else:
    data = clean_data(pd.read_csv("football_master_2024_27_v1.csv"))
    st.sidebar.success(f"Using master dataset — {len(data)} matches")

missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
if missing:
    st.error("Your CSV is missing these columns: " + ", ".join(missing))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Team Research",
    "🎯 Market Tester",
    "📊 Bookmaker Monitor",
    "📥 Data Format"
])

with tab1:
    teams = sorted(set(data.home_team.dropna()) | set(data.away_team.dropna()))

    team_search = st.text_input("Search team", "")

    filtered_teams = [
        t for t in teams
        if team_search.lower() in t.lower()
    ]

    team = st.selectbox("Team", filtered_teams)

    seasons = ["All"] + sorted(
        data["season"].dropna().unique().tolist(),
        reverse=True
    )
    season_filter = st.selectbox("Season", seasons)

    competitions = ["All"] + sorted(
        data["competition"].dropna().unique().tolist()
    )
    competition_filter = st.selectbox("Competition", competitions)

    venue_filter = st.selectbox("Venue", ["All", "Home", "Away"])

    n = st.slider(
        "Number of recent matches",
        3,
        min(15, max(3, len(data))),
        5
    )

    matches = team_matches(data, team)

    if season_filter != "All":
        matches = matches[matches["season"] == season_filter]

    if competition_filter != "All":
        matches = matches[matches["competition"] == competition_filter]

    if venue_filter != "All":
        matches = matches[matches["venue"] == venue_filter]

    recent = matches.head(n)

    # Convert match statistics into selected-team perspective
    recent = recent.copy()

    is_home = recent["home_team"].eq(team)

    recent["goals_for"] = recent["home_goals"].where(is_home, recent["away_goals"])
    recent["goals_against"] = recent["away_goals"].where(is_home, recent["home_goals"])

    recent["shots_for"] = recent["home_shots"].where(is_home, recent["away_shots"])
    recent["shots_against"] = recent["away_shots"].where(is_home, recent["home_shots"])

    recent["sot_for"] = recent["home_sot"].where(is_home, recent["away_sot"])
    recent["sot_against"] = recent["away_sot"].where(is_home, recent["home_sot"])

    recent["corners_for"] = recent["home_corners"].where(is_home, recent["away_corners"])
    recent["corners_against"] = recent["away_corners"].where(is_home, recent["home_corners"])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Matches", len(recent))

    if len(recent) > 0:
        c2.metric("Avg Goals", round(recent["goals_for"].mean(), 2))
        c3.metric("Avg Shots", round(recent["shots_for"].mean(), 2))
        c4.metric("Avg SOT", round(recent["sot_for"].mean(), 2))

    st.dataframe(recent)

with tab2:
    st.subheader("Test a market")

    teams = sorted(set(data.home_team.dropna()) | set(data.away_team.dropna()))
    team = st.selectbox("Team to test", teams, key="market_team")

    seasons = ["All"] + sorted(
        data["season"].dropna().unique().tolist(),
        reverse=True
    )
    season2 = st.selectbox("Season", seasons, key="market_season")

    competitions = ["All"] + sorted(
        data["competition"].dropna().unique().tolist()
    )
    competition2 = st.selectbox(
        "Competition",
        competitions,
        key="market_competition"
    )

    venue2 = st.selectbox(
        "Venue",
        ["All", "Home", "Away"],
        key="market_venue"
    )

    market = st.selectbox(
        "Market",
        ["Shots", "Shots on Target", "Corners", "Goals"]
    )

    direction = st.selectbox(
        "Direction",
        ["Over", "Under"]
    )

    line = st.number_input(
        "Line",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.5
    )

    odds = st.number_input(
        "Decimal odds",
        min_value=1.01,
        max_value=100.0,
        value=1.30,
        step=0.01
    )

    n2 = st.select_slider(
        "Recent sample",
        options=[5, 10, 15],
        value=10
    )

    m = team_matches(data, team)

    if season2 != "All":
        m = m[m["season"] == season2]

    if competition2 != "All":
        m = m[m["competition"] == competition2]

    if venue2 != "All":
        m = m[m["venue"] == venue2]

    m = m.head(n2)

    col_map = {
        "Shots": "team_shots",
        "Shots on Target": "team_sot",
        "Corners": "team_corners",
        "Goals": "team_goals"
    }

    col = col_map[market]

    rate = hit_rate(
        m[col],
        line,
        direction == "Over"
    )

    breakeven = 1 / odds

    a, b, c, d = st.columns(4)

    a.metric(
        "Historical hit rate",
        fmt_pct(rate)
    )

    b.metric(
        "Break-even probability",
        f"{breakeven*100:.1f}%"
    )

    c.metric(
        "Sample size",
        len(m)
    )

    if rate is not None:
        c4_text = "Above" if rate >= breakeven else "Below"
    else:
        c4_text = "—"

    d.metric(
        "Historical vs break-even",
        c4_text
    )

    if rate is not None:
        st.progress(min(max(rate, 0), 1))

    st.caption(
        "Historical hit rate is descriptive only. "
        "It does not establish the probability of the next match."
    )

    if len(m):
        result = m[
            ["date", "home_team", "away_team", "venue", col]
        ].copy()

        result["hit"] = (
            result[col] > line
            if direction == "Over"
            else result[col] < line
        )

        result["hit"] = result["hit"].map(
            {True: "✓", False: "✗"}
        )

        st.dataframe(
            result,
            
            use_container_width=True,
            hide_index=True
        )

with tab3:
    st.subheader("Bookmaker Market Monitor")

    WATCHLIST_FILE = "market_watchlist.csv"

    # ---------------------------------------------------------
    # LOAD WATCHLIST
    # ---------------------------------------------------------
    if "market_watchlist" not in st.session_state:
        if os.path.exists(WATCHLIST_FILE):
            try:
                loaded = pd.read_csv(WATCHLIST_FILE)

                # Make sure older watchlists have required columns
                if "Status" not in loaded.columns:
                    loaded["Status"] = "Watching"

                if "Actual Result" not in loaded.columns:
                    loaded["Actual Result"] = ""

                st.session_state.market_watchlist = (
                    loaded.fillna("").to_dict("records")
                )

            except Exception:
                st.session_state.market_watchlist = []
        else:
            st.session_state.market_watchlist = []

    # ---------------------------------------------------------
    # ADD NEW MARKET
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        bookmaker = st.selectbox(
            "Bookmaker",
            ["SportyBet", "SunBet", "Virgin Bet"],
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

        teams = sorted(
            set(data.home_team.dropna()) |
            set(data.away_team.dropna())
        )

        monitor_team = st.selectbox(
            "Team",
            teams,
            key="monitor_team"
        )

        market = st.selectbox(
            "Market",
            ["Shots", "Shots on Target", "Corners", "Goals"],
            key="monitor_market"
        )

    with col2:
        direction = st.selectbox(
            "Direction",
            ["Over", "Under"],
            key="monitor_direction"
        )

        line = st.number_input(
            "Line",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            key="monitor_line"
        )

        odds = st.number_input(
            "Decimal odds",
            min_value=1.01,
            max_value=100.0,
            value=1.30,
            step=0.01,
            key="monitor_odds"
        )

        sample = st.selectbox(
            "Historical sample",
            [5, 10, 15],
            index=1,
            key="monitor_sample"
        )

        monitor_season = st.selectbox(
            "Season",
            ["All"] + sorted(
                data["season"].dropna().unique().tolist(),
                reverse=True
            ),
            key="monitor_season"
        )

        monitor_competition = st.selectbox(
            "Competition",
            ["All"] + sorted(
                data["competition"].dropna().unique().tolist()
            ),
            key="monitor_competition"
        )

        monitor_venue = st.selectbox(
            "Venue",
            ["All", "Home", "Away"],
            key="monitor_venue"
        )

    if st.button("Analyse & Add Market", key="add_market"):

        m = team_matches(data, monitor_team)

        if monitor_season != "All":
            m = m[m["season"] == monitor_season]

        if monitor_competition != "All":
            m = m[m["competition"] == monitor_competition]

        if monitor_venue != "All":
            m = m[m["venue"] == monitor_venue]

        m = m.head(sample)

        col_map = {
            "Shots": "team_shots",
            "Shots on Target": "team_sot",
            "Corners": "team_corners",
            "Goals": "team_goals"
        }

        col = col_map[market]

        if len(m) > 0:
            rate = hit_rate(
                m[col],
                line,
                direction == "Over"
            )
        else:
            rate = None

        breakeven = 1 / odds

        if rate is not None:
            historical_hit_rate = rate * 100
            break_even_rate = breakeven * 100
            edge = historical_hit_rate - break_even_rate

            historical_vs_breakeven = (
                "Above"
                if rate >= breakeven
                else "Below"
            )

            edge_display = f"{edge:+.1f} pp"
            historical_display = f"{historical_hit_rate:.1f}%"
        else:
            historical_display = "-"
            edge_display = "-"
            historical_vs_breakeven = "-"
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
            "Market": market,
            "Direction": direction,
            "Line": line,
            "Odds": odds,
            "Historical Hit Rate": historical_display,
            "Break-even": f"{break_even_rate:.1f}%",
            "Edge vs Break-even": edge_display,
            "Historical vs Break-even": historical_vs_breakeven,
            "Status": "Watching",
            "Actual Result": ""
        }

        st.session_state.market_watchlist.append(entry)

        pd.DataFrame(
            st.session_state.market_watchlist
        ).to_csv(
            WATCHLIST_FILE,
            index=False
        )

        st.success("Market added to watchlist.")

    # ---------------------------------------------------------
    # WATCHLIST
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Tracked Markets")

    if st.session_state.market_watchlist:

        watchlist_df = pd.DataFrame(
            st.session_state.market_watchlist
        ).fillna("")

        # -----------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------
        total = len(watchlist_df)

        watching = (
            watchlist_df["Status"] == "Watching"
        ).sum()

        won = (
            watchlist_df["Status"] == "Won"
        ).sum()

        lost = (
            watchlist_df["Status"] == "Lost"
        ).sum()

        void = (
            watchlist_df["Status"] == "Void"
        ).sum()

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

        # -----------------------------------------------------
        # INDIVIDUAL MARKET EDITOR
        # -----------------------------------------------------
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
                    new_odds = st.number_input(
                        "Odds",
                        min_value=1.01,
                        max_value=100.0,
                        value=float(row.get("Odds", 1.30)),
                        step=0.01,
                        key=f"edit_odds_{i}"
                    )

                    new_line = st.number_input(
                        "Line",
                        min_value=0.0,
                        max_value=30.0,
                        value=float(row.get("Line", 3.5)),
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

                    actual_value = st.number_input(
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
                            ).strip() != ""
                            else 0.0
                        ),
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

                    st.success("Market updated.")

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

                    st.success("Market deleted.")

                    st.rerun()

        # -----------------------------------------------------
        # AUTOMATIC RESULT CHECK
        # -----------------------------------------------------
        for i, item in enumerate(
            st.session_state.market_watchlist
        ):

            actual_text = str(
                item.get("Actual Result", "")
            ).strip()

            if (
                actual_text != ""
                and item.get("Status") == "Watching"
            ):

                try:
                    actual = float(actual_text)
                    market_line = float(item["Line"])
                    market_direction = item["Direction"]

                    if market_direction == "Over":
                        won_result = actual > market_line
                    else:
                        won_result = actual < market_line

                    item["Status"] = (
                        "Won"
                        if won_result
                        else "Lost"
                    )

                except (ValueError, TypeError):
                    pass

        # Save any automatic status updates
        pd.DataFrame(
            st.session_state.market_watchlist
        ).to_csv(
            WATCHLIST_FILE,
            index=False
        )

        # -----------------------------------------------------
        # WATCHLIST TABLE
        # -----------------------------------------------------
        watchlist_df = pd.DataFrame(
            st.session_state.market_watchlist
        )

        st.dataframe(
            watchlist_df,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------
        # DOWNLOAD
        # -----------------------------------------------------
        csv_watchlist = watchlist_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Watchlist CSV",
            csv_watchlist,
            "market_watchlist.csv",
            "text/csv",
            key="download_watchlist"
        )

        # -----------------------------------------------------
        # CLEAR
        # -----------------------------------------------------
        if st.button(
            "Clear Monitor",
            key="clear_monitor"
        ):

            st.session_state.market_watchlist = []

            if os.path.exists(WATCHLIST_FILE):
                os.remove(WATCHLIST_FILE)

            st.rerun()

    else:
        st.info("No markets added yet.")

with tab4:
    st.subheader("CSV format")
    st.write("Your CSV should contain one row per match with these columns:")
    st.code(",".join(REQUIRED_COLUMNS))
    st.dataframe(demo.head(5), use_container_width=True, hide_index=True)

    csv_bytes = demo.to_csv(index=False).encode("utf-8")
    st.download_button("Download demo CSV", csv_bytes, "football_demo.csv", "text/csv")

st.divider()
st.caption("V1 deliberately avoids automatic 'safe bet' labels. The goal is to improve research quality and decision-making.")
