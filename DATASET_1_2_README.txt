Football Research Engine — Dataset 1 & Dataset 2
Prepared: 2026-09-28

Dataset 1: football_dataset1_match_context.csv
Purpose: supplementary match-level contextual evidence. It is NOT a replacement for the authoritative
football_master_2024_27_v1.csv. Core goals/shots/SOT/corners remain in the primary dataset.
Recommended free source: Football-Data.co.uk match-statistics files. The source documents availability
of shots, shots on target, corners, fouls, offsides, bookings/red cards and referees. Possession,
saves, free kicks, crosses and attack counts are not guaranteed by that source and must remain blank
unless obtained from another explicitly recorded source.
Important use restriction: Football-Data.co.uk states its free data are intended for private individuals
and not commercial or data-training products using automated bots/scrapers/AI.

Dataset 2: football_dataset2_player_evidence.csv
Purpose: supplementary player-level evidence from open event/lineup data.
Recommended free source: StatsBomb Open Data. StatsBomb provides selected competitions/seasons as JSON
matches, events and lineups. The app should derive player-match rows from those files and preserve
source/source_match_id. Do NOT infer missing player facts. Blank means unavailable.
StatsBomb requires attribution if research/analysis based on its data is published/shared.

Important:
- No synthetic/fabricated values are included.
- No source is silently merged into the primary CSV.
- data_status should be one of: observed, derived, unavailable, pending.
- source_updated_at should reflect the source timestamp where available.
- source_match_id is retained for traceability.
