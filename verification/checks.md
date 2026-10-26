# Verification

## Submission readiness — 26 October 2026

The owner-created [GitHub repository](https://github.com/gour6380/Structured-Request-Adaptation)
is public and was empty when checked. The local folder is ready for its first
content push; this check did not initialize a repository, commit or push files.

- The offline verifier again checked **94 manifest-covered evidence files**,
  rescored **11,896 test predictions** and **1,200 benchmark requests**, and
  reproduced the saved metrics and derived tables.
- **27 standalone Git inclusion/exclusion checks passed.** Existing ignore
  rules need no functional changes. The executed notebook, figures, JSON/JSONL
  evidence and published `environment.json` remain eligible. Data, environments,
  raw runs/jobs, archives, weights, optimizer states and credential patterns are
  excluded. No tested token/private-key patterns or files at least 50 MiB were
  found in the eligible list.
- README clone/setup instructions now use the actual repository. Documentation
  distinguishes viewing saved results, starting a new run, and resuming from a
  complete original archive. Relative links in the current publication documents
  resolve to included files.
- Notebook bytes (including outputs), source, tests and published result files
  remain unchanged. Original run/job file sizes and modification timestamps and
  the parent Growth ignore file are unchanged.
- The project `.venv/` and `data/` remain absent following the owner's cleanup.
  Neither is needed for this offline check. No packages were installed and no
  notebook, model, training, GPU evaluation or project test suite was run.

Receipt: [submission-readiness-2026-10-26.json](submission-readiness-2026-10-26.json).
Current candidate list: [github-files.txt](github-files.txt). The historical
test results below retain their original date and scope.

## Publication preparation — 30 September 2026

The completed remote run and this local publication preparation are separate
pieces of evidence. No model download, notebook execution, training or model
evaluation was performed while preparing the publication. Bounded tests use
fixtures and tiny CPU models.

### Checks performed on 30 September

- **371 tests passed; 2 optional native-MPS tests skipped** in 36.76 seconds.
  Receipt: [publication-tests.xml](publication-tests.xml). The suite ran with
  Hub/Transformers offline and native-MPS tests disabled.
- Ruff lint and formatting checks passed for **45 current source/test files**.
- All **54 notebook code cells and 74 saved outputs**, execution counts and
  metadata are preserved. Only Markdown explanations changed; the original
  blank trailing cell is retained.
- The offline verifier checked **94 manifest-covered public files**, rescored
  all **11,896 test predictions** and **1,200 benchmark predictions**, and
  reproduced metric, latency and throughput summaries from saved rows.
- The original **300 registered run artifacts** passed the preceding full audit;
  before/after preservation checks cover **512 original run/data/job files and
  the parent ignore file**. No frozen source, model, checkpoint or data artifact
  was edited.
- Future memory aggregation now sums coverage across disjoint intervals and
  avoids counting live snapshots twice. Regression tests cover checkpoint-boundary
  finalization, committed-prefix resume, legacy unknown metadata, prior errors
  and live swap observations. Original allocation maxima remain unchanged.
- Notebook tests now validate the executed full configuration and preserved
  outputs instead of requiring the superseded output-free development notebook.
- Standalone Git eligibility is checked using temporary Git metadata outside the
  project. The parent Growth ignore rules stay unchanged. The candidate file
  list is [github-files.txt](github-files.txt); integrity, link, secret-pattern
  and exclusion checks are recorded in [publication.json](publication.json).

## Scope of the recorded experiment

The original run used **two NVIDIA A40 GPUs**, Linux/Python **3.12.3**, PyTorch
**2.8.0+cu128**, Transformers **5.17.0**, PEFT **0.21.0** and bitsandbytes **0.50.2**.
Its 32 worker jobs succeeded and 33 recorded stages completed. Both arms finished
five epochs, 3,600 updates and 1,172,405 supervised tokens; every test system has
2,974 predictions. This verifies that particular full-model execution, not
compatibility or identical numerical behavior on every machine.

The original notebook snapshot contains development defaults, while the executed
notebook and frozen run configuration establish full/16×1/evaluation32 execution.
Historical outputs and source remain unchanged. The published report documents
this distinction, strict code-fence failures, and the legacy sampler metadata
correction. Real interruption/resume of the full remote model was not demonstrated
by this uninterrupted run; implementation tests are narrower evidence.

That local verification used the existing native Python 3.13.11 environment.
No environment or kernel was created and no dependency installation was required.
This is a local release candidate: no project Git repository, commit, remote,
push, release tag, ZIP or public publication was created.

## Historical pre-upload receipt

The text below describes the earlier, pre-execution delivery. Its test count,
output-free notebook and pending-remote statements are historical, not the state
of the completed run or current notebook.

# Historical verification — 30 September 2026

## Simplified installation

The notebook installs `requirements.txt` directly into its current Python using
`python -m pip install -r requirements.txt`. Workers use that same interpreter.
The custom environment installer, permission probes, pip bridge and kernel
registration have been removed, along with their obsolete tests.

Linux requirements allow compatible preinstalled CUDA PyTorch 2.8 or newer in
the 2.x series to remain installed. Core model-library versions remain pinned;
utilities have bounded requirements. Every run records actual installed package
versions. This direct dependency list is not a complete transitive lockfile.

## Local checks

- **350 tests passed; 2 optional native-MPS checks skipped** (`tests.xml`).
- A real child-process test verifies default worker execution using the current
  interpreter without a project environment. Scheduling, device masking, error
  handling, checkpoints, data checks and scientific metrics retain their tests.
- Notebook tests verify direct pip arguments without installing packages, an
  explicit absolute project path, recovery from a deleted working directory,
  and the one-epoch development preset. The full five-epoch protocol remains
  available by removing `preset="development"` and creating a new run.
- Ruff lint and formatting passed for all 42 Python source/test files.
- All 53 notebook code cells parse, contain no saved outputs, and have Markdown
  explanations. The notebook schema is valid.
- Direct requirements are satisfied by the existing native Python 3.13.11 test
  interpreter. No new environment or dependencies were installed during this
  verification. Tiny CPU fixtures are not a full-model training run.
- The complete upload ZIP excludes environments, caches, old setup code and
  runtime artifacts. Dataset checksums and ZIP contents are verified.

## Pending remote checks

Installation into the remote image, full-model CUDA/NF4 compatibility, memory fit
and runtime remain pending on the selected machine. Execute the notebook's
preflight and pilot before a full comparison. No experiment results are claimed.
