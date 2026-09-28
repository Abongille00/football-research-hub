# Football Deep Research Engine — Final Integrated Build

## What this build does

One fixture in -> automatic evidence processing -> candidate survival -> research report.

The primary dataset remains `football_master_2024_27_v1.csv`.
Supplementary layers never silently replace primary match data.

## Required files next to `app.py`

- `app.py`
- `dataset34_pipeline.py`
- `event_tactical_engine.py`
- `football_master_2024_27_v1.csv`
- `football_dataset1_match_context.csv` (optional but included)
- `football_dataset2_player_evidence.csv` (optional but included)
- `requirements.txt`

## Dataset 3 + 4 population

Run once in an environment with internet access:

```bash
python build_dataset_3_4.py
```

This downloads and normalizes:

- Wyscout public event data -> Dataset 3
- Impect public open data -> Dataset 4

The resulting normalized CSVs are created beside `app.py`.

The app does not fabricate missing rows. If source downloads are unavailable, it reports those layers as unavailable/empty.

## Sources

Wyscout public event dataset:
https://github.com/koenvo/wyscout-soccer-match-event-dataset

Impect Open Data:
https://github.com/ImpectAPI/open-data

StatsBomb Open Data:
https://github.com/hudl/open-data

## Deployment

For Streamlit Cloud, put all required files in the same repository directory as `app.py` and set the main file to `app.py`.

Do not commit API credentials. If credentials are later added for a licensed provider, use Streamlit Secrets/environment variables.

## User workflow

1. Enter home team.
2. Enter away team.
3. Select competition context.
4. Enter only the bookmaker minimum line + odds actually available.
5. Run the research engine.
6. Review surviving events and counter-evidence.

The engine does not assume that lower bookmaker lines exist when the supplied minimum line is higher.

## Important evidence rule

Historical rates, Poisson benchmarks, event features and model outputs are research evidence. They are not guarantees of the next match outcome.
