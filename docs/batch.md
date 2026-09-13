# Batch generation and reuse

`bulk_convert_soma.py --help` documents local source archive, metadata, bind,
roster, output, cache, batch size and free-disk guard options. Generation can
use the same direct backend or invoke.ps1; failures are bisected to isolate a
bad capture. A persisted ledger supports resume, and temporary source batches
are removed after their output has been recorded.

`--shard-count N --shard-index K` partitions work for independent workers.
Use a distinct output/ledger per worker and avoid overlapping source shards.
GPU memory and budget determine safe worker count; do not assume linear speedup.
SOMA floor/crawl-type filtering in this legacy bulk entrypoint is explicit in
GROUND_PATTERN; inspect it before choosing the corpus. It is not format-generic.

Model generation, evaluation and rendering are distinct jobs. Evaluate existing
NPZs without rerunning the model; render only clips you want to inspect. No
account-specific Modal launch/session scripts are redistributed.
