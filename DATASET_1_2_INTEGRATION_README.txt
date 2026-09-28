Football Research Engine — Dataset 1 & Dataset 2 integration

Primary dataset remains football_master_2024_27_v1.csv.

Dataset 1: football_dataset1_match_context.csv
- 16,614 actual rows copied from the authoritative primary CSV.
- Core goals/shots/SOT/corners are real observed primary records.
- Supplementary context fields (possession, fouls, cards, offsides, saves, etc.) are blank because those values are not present in the primary source.
- No values were fabricated.

Dataset 2: football_dataset2_player_evidence.csv
- 10 actual StatsBomb Open Data evidence rows based on public match 15946 (Barcelona vs Deportivo Alavés) as documented by the Kloppy StatsBomb example.
- This is a seed/validation sample, not the complete StatsBomb corpus.
- Rows are explicitly marked observed_event_sample so the engine does not mistake the sample for complete player-match aggregates.

App integration:
- Both files are loaded separately.
- Supplementary health is displayed in Advanced Data & Sources.
- Fixture-specific supplementary-source availability is shown to the research engine.
- Primary data remains authoritative.
- Missing external values are not inferred or silently merged.
