# Football Deep Research Engine — Stage A-C update

Replace the deployed `app.py` and keep `source_acquisition.py`, `data_pipeline.py` and the primary CSV beside it.

Stage A: future fixture identity is separate from historical results. `All` is treated as no competition restriction.
Stage B: historical home/away/H2H samples are built independently of whether the target fixture exists in the primary CSV.
Stage C: missing D3/D4/player/context layers are reported as unavailable/limited rather than inventing evidence; they do not block valid historical research by themselves.

Acquired public-source records are marked UNVERIFIED unless explicitly verified. Verified values take precedence during reconciliation.
