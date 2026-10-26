# Saved comparison evidence

Start with the [technical report](report.md). These files describe the completed
30 September 2026 comparison; viewing or verifying them does not execute a model.

| Evidence | Contents |
|---|---|
| [Summary](summary.json), [test table](test_metrics.csv) | Four-system metrics, matched budgets, format diagnosis and qualified memory metadata |
| [Epoch table](epochs.csv), `trials/` | Training journals, all ten epoch summaries and original training summaries |
| [Memory/runtime table](memory_runtime.csv), `benchmarks/` | Observed allocations, latency and the 1,200 measured requests |
| `evaluations/` | All 11,896 test predictions, metrics and the frozen protocol |
| `figures/` | Five figures copied unchanged from the original run |
| [Configuration](config.json), [environment](environment.json), `preparation/` | Effective settings, recorded package versions, prompts, data audits and architecture |
| `inputs/`, `initial_adapter/` | Small identity/configuration/license records; no weights |
| `source_snapshot/` | Original experimental source, requirements and historical notebook |
| `original_reports/` | Original reports retained unchanged, including their documented limitations |
| [Publication manifest](publication_manifest.json) | Exported-file hashes, original paths, derivations and exclusions |

`manifest.json` and `identity.json` are copies of the **original full-run** records.
They intentionally refer to some weights and other files excluded from this
publication. `publication_manifest.json` is the inventory for this audit bundle.
The manually authored `README.md` and `report.md` are separate from that export
manifest; the final publication verification receipt records their hashes.

The original training summaries retain the legacy sampling metadata defect.
Use the qualified values in `summary.json` and the explanation in `report.md`:
maximum memory observations are retained, while unavailable cumulative sampling
metadata is not reconstructed. Single-interval benchmark coverage is preserved.

From the project root, verify hashes, all test predictions, and benchmark scores
and timings without installing the model stack:

```python
from pathlib import Path
from src.publication import verify_publication

verify_publication(Path("results/run_001"))
```

The public directory is not a resumable checkpoint. The complete original
`runs/run_001/` and shared `data/` remain local and ignored by Git. Reproduction
instructions are in the [project README](../../README.md); dataset/model
attribution and exceptions to the project code license are in
[NOTICE](../../NOTICE.md).
