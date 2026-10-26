# Structured Request Adaptation

[GitHub repository](https://github.com/gour6380/Structured-Request-Adaptation) ·
[Technical report](results/run_001/report.md) ·
[Saved evidence](results/run_001/README.md) ·
[Verification](verification/checks.md)

A notebook-driven comparison of **LoRA and NF4 QLoRA** on
**Qwen2.5-1.5B-Instruct**, converting English requests into structured JSON:

```json
{"intent":"alarm_set","slots":[{"type":"time","value":"seven am"}]}
```

Start with the [measured comparison](results/run_001/report.md). The
[executed notebook](notebooks/lora_qlora.ipynb) retains the original code,
outputs and execution counts from the completed run. Viewing it does not require
installing dependencies or renting a GPU. It contains substantial text output;
the report provides a shorter route through the results and figures.

The recorded run used two NVIDIA A40 GPUs, Python **3.12.3**, PyTorch
**2.8.0+cu128**, BF16 computation and one paired seed, **2027**. Both training arms
completed five epochs over 11,514 examples, with microbatch **16**, accumulation
**1**, evaluation batch **32** and **3,600 optimizer updates** each.

| System | Whole-record accuracy | Valid JSON |
|---|---:|---:|
| Unquantized base | 5.68% | 82.21% |
| NF4 base | 0.27% | 7.46% |
| LoRA, final epoch 5 | **72.46%** | 99.97% |
| QLoRA, final epoch 5 | **72.06%** | 100.00% |

These [full-test measurements](results/run_001/test_metrics.csv) use the same
2,974 requests. A whole-record success requires valid JSON/schema, the correct
intent, and all extracted slots to match. LoRA's 12-example lead is an observation
from one paired seed, not evidence of general superiority. QLoRA used less observed
allocated GPU memory but took longer; reserved training memory was higher. The
report gives the separate measurements and their limits.

The weak unadapted NF4 baseline largely reflects code-fenced outputs
that violate the strict JSON contract; the report separates this formatting
failure from claims about general language understanding.

One notebook and one shared implementation handle supported hardware. Imports,
plots and saved-result exports never start missing experiments.

## Device selection and scheduling

Devices are selected once before run creation, in this order: **CUDA → MPS → CPU**.

| Available hardware | Arm scheduling | Training microbatch × accumulation | Evaluation batch |
|---|---|---|---|
| At least two CUDA GPUs, each selected GPU with at least 24 GiB reported memory | LoRA on the first GPU, QLoRA on the second; concurrent | 4 × 4 | 8 |
| One CUDA GPU with at least 24 GiB reported memory | LoRA then QLoRA; serial | 4 × 4 | 8 |
| Smaller CUDA GPU(s) | Parallel only when two are available; otherwise serial | 1 × 16 | 1 |
| MPS | Serial | 1 × 16 | 1 |
| CPU | Serial | 1 × 16 | 1 |

These are defaults for a **new** configuration. The recorded A40 run explicitly
overrode them with 16 × 1 training and evaluation batch 32. Do not copy these
larger overrides to MPS or a smaller GPU without a new compatibility/pilot check.

The first two visible CUDA GPUs are used; additional GPUs are recorded but idle.
A mixed pair uses the smaller GPU's batching policy and one common precision.
Prefer matching GPUs when comparing memory and timing: different GPU models
confound a method-only hardware comparison. Two GPUs have separate memory pools;
this workflow does not distribute one model across both GPUs.

CUDA uses native BF16 when all selected GPUs support it, otherwise FP16 with
loss scaling. MPS uses BF16 on a supported macOS installation; CPU uses FP32.
Adapters and optimizer state remain FP32. CPU/accelerator and quantization
compatibility are checked explicitly. Unsupported NF4 kernels, OOM or nonfinite
values stop the stage rather than silently moving devices or changing batches.

Both arms have the same **effective batch of 16**, including a correctly weighted
final partial window. The hardware policy and actual configuration are saved.
Change batches before creating a run; changing device, precision or configuration
requires a new comparison. The small-memory defaults are conservative, not a
promise of fit on every device.

## Setup

To run a new experiment, use a notebook kernel with **Python 3.12 or 3.13** on
Linux x86_64 or macOS arm64. Python 3.13 is preferred for a new local setup; the
published measurement used Python 3.12.3. Clone the project:

```sh
git clone https://github.com/gour6380/Structured-Request-Adaptation.git
cd Structured-Request-Adaptation
```

Open `notebooks/lora_qlora.ipynb` in Jupyter or a notebook-capable editor. Save a
separate notebook copy before running if you
want to keep the published outputs visible unchanged.

The first cells set `PROJECT_ROOT` explicitly, change the working directory, and
install dependencies directly into the current notebook Python:

```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")
])
```

The recorded notebook uses `/workspace/structured-request-adaptation`. Edit its
`PROJECT_ROOT` cell to the absolute path of your copy, for example:

```python
PROJECT_ROOT = Path("/absolute/path/to/Structured-Request-Adaptation")
```

No separate environment or kernel setup is required. If installation updates packages already imported in the kernel,
restart the kernel, rerun the imports/path cells, skip the completed install cell,
and continue. All workers use the same Python interpreter as the notebook.

To install from a terminal using the notebook's Python interpreter:

```sh
python -m pip install -r requirements.txt
```

Keep Internet access enabled for dependencies and model downloads. On Linux, a
compatible preinstalled PyTorch 2.x is reused, including the CUDA build supplied
by the machine image. Installation does not change the GPU driver. Direct
dependencies are listed in `requirements.txt`; each run records the versions
actually installed.

### Files needed on another machine

Clone the repository or copy these files into the chosen project directory:

| Files | Purpose |
|---|---|
| `src/`, `notebooks/`, `requirements.txt` | Implementation, notebook and dependencies |
| `README.md`, `LICENSE`, `NOTICE.md`, `licenses/` | Setup, protocol and attribution |
| `data/` (optional local copy) | About 42 MiB of verified files; avoids downloading again |
| `tests/`, `ruff.toml`, `verification/`, `.gitignore` | Checks and repository settings |
| `results/` | Published evidence; not required to train a new model |

Copy the complete `data/massive-1.1/en-US/` directory: `archive.tar.gz`,
`en-US.jsonl`, `LICENSE`, and `manifest.json`. Keep names and contents unchanged;
do not unpack `archive.tar.gz`. Alternatively omit `data/` entirely and let the
dataset-acquisition cell download it. A partial dataset fails integrity checks.
Git excludes data downloads, environments, caches, credentials, jobs and full run
archives. A repository clone therefore acquires the dataset and pinned model in
their explicit notebook cells. The notebook creates `runs/` and `jobs/` when
needed. The dataset directory is also created on acquisition; it does not need
to exist in a fresh checkout. Preserve the full local run and shared data before deleting a remote
machine or its storage; the small public evidence bundle cannot reload a model.

### Quick test and full comparison

The preserved configuration cell records the completed **full** run:

```python
config = ExperimentConfig.for_hardware(
    hardware,
    model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    preset="full",
    microbatch_size=16,
    gradient_accumulation=1,
    evaluation_batch_size=32,
)
```

For a **new quick test**, replace that cell in your working copy with:

```python
config = ExperimentConfig.for_hardware(
    hardware,
    model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    preset="development",
)
```

This uses 128 training examples, 32 validation examples and one epoch, while still
acquiring the complete model. To check only compatibility and speed, stop after
inspecting both pilots. To exercise the remaining workflow, continue: development
runs skip official test evaluation and report its results as unavailable.

For a new full run using the conservative defaults for your hardware, use the
same short configuration with `preset="full"`. This lets MPS, CPU and smaller
CUDA devices choose their lower batch sizes. Use `RUN_TO_RESUME = None`, then
create a fresh run. Never alter an existing run's frozen configuration.

## Notebook flow

Each action has an explanatory Markdown cell and its own code cell:

1. Set the project path, install requirements, and inspect available hardware.
2. Define `ExperimentConfig.for_hardware(hardware, model_revision=...)` and create
   `run_001`, `run_002`, etc., or explicitly select a compatible run to resume.
3. Acquire/verify the shared MASSIVE archive, convert annotations, and inspect
   split/duplicate audits.
4. Acquire pinned model/tokenizer snapshots, prepare NF4, tokenize and inspect
   loss masking, architecture and adapter counts.
5. Run compatibility checks, then the bounded timing pilot for each arm.
6. Inspect the pilots before continuing to validation baselines and training.
7. Freeze both final adapters, evaluate all four systems on test, and benchmark.
8. Collect saved results, generate figures/report, make a prediction and inspect status.

The two arm-launch cells use the same interface on every device. With one device,
`start_arm` queues the second arm and `wait` advances it only after the first
finishes. A failed or cancelled arm cancels its peer, including queued work.
Coordinator stages cannot mutate shared preparation while arm workers run.

The saved notebook shows the **completed full protocol**: five epochs, all
**11,514 training / 2,033 validation / 2,974 test** requests and seed **2027**.
Each full training arm performs **3,600 optimizer updates**. Follow the quick-test
configuration above for development; development runs cannot evaluate the
official test set. Later saved cells have outputs with unset execution counters;
the committed worker results and artifact audit establish their completion.

The pilot uses disposable adapters, up to two warmup updates plus twenty measured
updates and up to 32 validation requests. It preserves initial adapters and main
training checkpoints. Smaller development data use fewer updates and explicit
warmup counts. Its training-only projections exclude model preparation,
checkpoint writes, validation, test and latency benchmarks. Serial time adds both
arms; parallel time follows the slower arm. No end-to-end ETA is claimed without
measurements on the selected hardware.

## Experimental controls

Four systems separate quantization from adaptation: unquantized base, NF4 base,
LoRA and QLoRA. The two adapted systems share saved initial adapters, prepared
examples, seeded permutations, supervised-token budgets and update boundaries.
Final five-epoch adapters are the primary equal-budget comparison. Best-validation
adapters are also saved; selection uses whole-record accuracy, then lower answer
NLL, then earlier epoch.

Training uses explicit PyTorch AdamW: learning rate `2e-4`, betas `(0.9, 0.999)`,
epsilon `1e-8`, no weight decay, 3% warmup, cosine decay and gradient clipping at
1.0. LoRA rank/alpha are 16/32 with dropout 0.05 on attention and MLP projections.
The sequence limit is 2,048 tokens and greedy generation permits 256 new tokens.
Non-reentrant gradient checkpointing is enabled and training KV caching disabled.

Only assistant answer tokens and EOS contribute to loss. Gradients use the
actual supervised-token count across each accumulation window. Right-padded
training and left-padded generation preserve attention/label masks. Evaluation
retains per-example NLL, generated tokens and strict scores; timing is recorded
once per generation batch. Batch-one inference benchmarks remain separate.

Targets preserve slot occurrences, casing and punctuation. Annotation removal
must reconstruct the original utterance. Official splits stay intact, with exact
and normalized duplicate audits; there is no silent relabeling or split movement.
Inputs contain only the utterance and a fixed contract/allowed-label instruction.
The project acquires the official archive directly and preserves its **CC BY 4.0**
attribution and checksums; no remote dataset-loading script is executed.

The primary metric is **whole-record exact accuracy**: schema-valid JSON, correct
intent and the exact slot multiset. Additional metrics include parse/schema
validity, intent accuracy/macro-F1, slot micro-P/R/F1, answer NLL and failure
categories. Invalid responses stay in the denominator. Parsing rejects prose,
fences, duplicate keys, unknown labels, extra fields and wrong value types.
There is no repair, retry or constrained decoding. A secondary test slice excludes
normalized requests seen in training/validation; it does not rule out semantic
overlap or pretrained contamination.

Full validation runs after every epoch. Test evaluation requires completed,
matched five-epoch budgets and a frozen protocol. Inference benchmarks use 100
fixed validation requests, five warmups and three measured repetitions at batch
one. Timing synchronizes accelerators. Parallel workers share CPU, RAM and disk,
so their timings are workload measurements, not isolated hardware benchmarks.

## Files, logs and recovery

```text
src/                         implementation, hardware and worker control
notebooks/lora_qlora.ipynb    executed notebook with preserved measured outputs
tests/                       bounded correctness and orchestration checks
requirements.txt             direct dependencies and platform markers
README.md                    setup, protocol and recovery
verification/                dated local checks and saved-run audit
results/run_001/              public report, figures and compact derived evidence
LICENSE, NOTICE.md, licenses/ project license and third-party attribution
data/massive-1.1/en-US/       shared archive, JSONL, license and checksums
runs/run_XXX/                configuration, source/environment/model snapshots,
                             preparation, adapters, checkpoints, metrics and reports
jobs/                        worker requests, status and logs
```

TensorBoard, CSV/JSONL and Python logging retain losses, learning rates, gradient
norms, correctness, throughput and observed memory. Pilot activity includes
process CPU/RSS, device allocation and GPU utilization where available. CUDA
utilization is queried by physical UUID. MPS unified memory overlaps RSS; report
these separately. Sampled maxima can miss allocation peaks.

For a run created with your current source, set `RUN_TO_RESUME` to an exact
`run_XXX` name and rerun the relevant notebook cells. `None` always creates a new
run; it does not discover or resume the latest one.
Source, dependencies, configuration, backend and artifact identities must match.
The controller records explicit compatible replacement sessions when physical
GPU UUIDs change. It does not promise bitwise reproducibility across hosts.
Completed stages reuse verified outputs instead of recomputing them.

The recorded `run_001` has an immutable source snapshot from before publication
utilities were added. Do not reopen it through the changed current source or
rewrite its identity hashes. A genuine resume requires its original source,
recorded dependencies and complete local artifacts. Read or export its saved
evidence offline as described below instead. The original notebook snapshot
contains the unsaved development template; `config.json`, final counters and the
executed notebook establish that the measured experiment used the full settings.

Checkpoints commit atomically every 50 optimizer updates, at epoch boundaries and
at completion. They include adapter weights, optimizer/scheduler/scaler state,
RNG state, next-example position, counters and validation-selection state.
Incomplete saves and uncommitted logs are reconciled on recovery. Keep the latest
two recovery checkpoints, best adapter and final adapter. Frozen base weights are
saved once; QLoRA reloads its saved NF4 base and unmerged adapter.

For unattended execution on a remote machine, save the notebook with the intended
`RUN_TO_RESUME` value and execute it in a persistent terminal session such as tmux:

```sh
mkdir -p jobs
python -m jupyter nbconvert --to notebook --execute notebooks/lora_qlora.ipynb --ExecutePreprocessor.timeout=-1 --output executed --output-dir jobs
```

A saved `None` deliberately creates another run. Finish/cancel active workers
before starting a second execution. Remote jobs can continue after the local
computer disconnects; a machine/kernel shutdown still interrupts them. Stop or
release rented compute after saving results; notebook completion does not manage
billing. TensorBoard can be started separately:

```sh
python -m tensorboard.main --logdir runs --port 6006
```

## Read and export saved evidence

The [public report](results/run_001/report.md) and files alongside it are a
compact publication derived from the completed archive. They include measured
results and figures; they do not contain base weights, adapters or optimizer
states. Full run archives belong in `runs/`, with the separately acquired dataset
in `data/`; neither is included in Git. Keep both when archiving or transferring
a resumable experiment. Published-result verification works without either.

From the project directory, verify the published evidence without a full local
archive or any machine-learning dependencies:

```python
from pathlib import Path
from src.publication import verify_publication

verify_publication(Path("results/run_001"))
```

Or run this single command from the checkout root (Python 3.12 or newer; no
package installation or GPU required):

```sh
python -B -c 'from pathlib import Path; from src.publication import verify_publication; print(verify_publication(Path("results/run_001")))'
```

If you also have the full local run, export to a **new** destination:

```python
from src.publication import export_publication

export_publication(Path("runs/run_001"), Path("results/reexport_run_001"))
```

The exporter refuses to overwrite an existing directory. It generates the
evidence and compact tables; the separately authored narrative is `report.md`.

Export reads saved measurements and validates provenance. It does not install
dependencies, load models, download data, train, evaluate or reopen the run with
the current experiment source. Verification of the published bundle does not
require the excluded model files. See the report for source paths and the scope
of the recorded audit.

## Verification and limitations

The [26 October submission check](verification/submission-readiness-2026-10-26.json)
reverified saved evidence, standalone Git inclusion, and documentation links.
The **371 passed / 2 skipped** test result is from **30 September**, not a new
test run. See [`verification/checks.md`](verification/checks.md) for dated scope.
After installing dependencies, run bounded tests with `python -m pytest tests`.
Tests use fixtures
and tiny CPU models. The completed A40 run separately provides full-model CUDA
execution evidence for its recorded environment, not for every supported device.
Preflight and the pilot remain necessary for a new setup. Successful checkpoint
saves and local recovery tests do not demonstrate that this remote run actually
recovered from a mid-training interruption.

A single paired seed supports a measured case study, not a claim of general
superiority or training-seed stability. Concurrent timings may include shared-host
contention. Memory values are maximum observed samples, not guaranteed peaks.
The quantized baseline's poor strict score largely reflects fenced output;
the saved-prediction diagnosis is not a general assessment of NF4 or language
understanding, and no additional model run was needed for that diagnosis.

This is a controlled adaptation of [LoRA](https://arxiv.org/abs/2106.09685),
[QLoRA](https://arxiv.org/abs/2305.14314) and
[MASSIVE](https://arxiv.org/abs/2204.08582), using ordinary AdamW rather than the
QLoRA paper's paged optimizer. Model architecture and tokenizer come from the
pinned [official model repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

Project code is available under the [MIT License](LICENSE). Dataset text and
third-party assets retain their own terms; see [NOTICE.md](NOTICE.md).
