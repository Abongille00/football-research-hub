
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

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
    data = clean_data(demo)
    st.sidebar.info("Using demo data. Upload your own CSV to replace it.")

missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
if missing:
    st.error("Your CSV is missing these columns: " + ", ".join(missing))
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Team Research", "🎯 Market Tester", "📥 Data Format"])

with tab1:
        teams = sorted(set(data.home_team.dropna()) | set(data.away_team.dropna()))
    team_search = st.text_input("Search team", "")
    filtered_teams = [t for t in teams if team_search.lower() in t.lower()]
    team = st.selectbox("Team", filtered_teams)

    venue_filter = st.selectbox("Venue", ["All", "Home", "Away"])
    n = st.slider("Number of recent matches", 3, min(15, max(3, len(data))), 5)

    matches = team_matches(data, team)

    if venue_filter != "All":
        matches = matches[matches["venue"] == venue_filter]

    recent = matches.head(n)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Matches", len(recent))
    c2.metric("Avg shots", f"{recent.team_shots.mean():.1f}" if len(recent) else "—")
    c3.metric("Avg SOT", f"{recent.team_sot.mean():.1f}" if len(recent) else "—")
    c4.metric("Avg corners", f"{recent.team_corners.mean():.1f}" if len(recent) else "—")

    st.subheader("Match-by-match record")
    display_cols = ["date","home_team","away_team","team_goals","opp_goals",
                    "team_shots","opp_shots","team_sot","opp_sot",
                    "team_corners","opp_corners","venue"]
    st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Distribution")
    chart_df = recent[["date","team_shots","team_sot","team_corners"]].set_index("date")
    st.line_chart(chart_df)

with tab2:
    st.subheader("Test a market")
    teams = sorted(set(data.home_team.dropna()) | set(data.away_team.dropna()))
    team = st.selectbox("Team to test", teams, key="market_team")
    market = st.selectbox("Market", ["Shots", "Shots on Target", "Corners", "Goals"])
    direction = st.selectbox("Direction", ["Over", "Under"])
    line = st.number_input("Line", min_value=0.0, max_value=30.0, value=3.5, step=0.5)
    odds = st.number_input("Decimal odds", min_value=1.01, max_value=100.0, value=1.30, step=0.01)

    n2 = st.slider("Recent sample", 5, 15, 10)
    venue2 = st.selectbox("Venue filter", ["All", "Home", "Away"], key="market_venue")
    m = team_matches(data, team)
    if venue2 != "All":
        m = m[m["venue"] == venue2]
    m = m.head(n2)

    col_map = {
        "Shots":"team_shots",
        "Shots on Target":"team_sot",
        "Corners":"team_corners",
        "Goals":"team_goals"
    }
    col = col_map[market]
    rate = hit_rate(m[col], line, direction == "Over")
    breakeven = 1 / odds

    a,b,c,d = st.columns(4)
    a.metric("Historical hit rate", fmt_pct(rate))
    b.metric("Break-even probability", f"{breakeven*100:.1f}%")
    c.metric("Sample size", len(m))
    if rate is not None:
        c3 = "Above" if rate > breakeven else "Below"
    else:
        c3 = "—"
    d.metric("Historical vs break-even", c3)

    if rate is not None:
        st.progress(min(max(rate,0),1))
    st.caption("Historical hit rate is descriptive only. It does not establish the probability of the next match.")

    if len(m):
        result = m[["date","home_team","away_team",col,"venue"]].copy()
        result["hit"] = result[col] > line if direction == "Over" else result[col] < line
        result["hit"] = result["hit"].map({True:"✅", False:"❌"})
        st.dataframe(result, use_container_width=True, hide_index=True)

    st.warning("Before betting, investigate opponent strength, game state, lineup news, tactical matchup and whether the historical sample is actually comparable.")

with tab3:
    st.subheader("CSV format")
    st.write("Your CSV should contain one row per match with these columns:")
    st.code(",".join(REQUIRED_COLUMNS))
    st.dataframe(demo.head(5), use_container_width=True, hide_index=True)

    csv_bytes = demo.to_csv(index=False).encode("utf-8")
    st.download_button("Download demo CSV", csv_bytes, "football_demo.csv", "text/csv")

st.divider()
st.caption("V1 deliberately avoids automatic 'safe bet' labels. The goal is to improve research quality and decision-making.")
