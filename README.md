# Football Betting Research Tool — V1

A free Streamlit research dashboard for football match data.

## What V1 does

- Upload a CSV of match-by-match data.
- Research a team's recent home/away matches.
- View shots, shots on target, goals and corners.
- Test an over/under market against a historical sample.
- Calculate historical hit rate and odds break-even probability.
- Keep the tool descriptive rather than pretending to predict the future.

## Run locally

Install Python 3.10–3.14, then:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit app normally opens in your browser.

## CSV columns

```text
date,home_team,away_team,home_goals,away_goals,home_shots,away_shots,home_sot,away_sot,home_corners,away_corners
```

## Next versions

V2 can add:
- automatic data importing
- opponent-strength filters
- last 5/10/15 comparison
- home/away splits
- team defensive concessions
- similar-opponent filtering
- player markets
- xG
- cards
- a betting journal
- backtesting
