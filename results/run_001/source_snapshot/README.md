# Structured request adaptation

A notebook-driven comparison of **LoRA and NF4 QLoRA** on
**Qwen2.5-1.5B-Instruct**, converting English requests into structured JSON:

```json
{"intent":"alarm_set","slots":[{"type":"time","value":"seven am"}]}
```

Open [`notebooks/lora_qlora.ipynb`](notebooks/lora_qlora.ipynb). One notebook and
one shared implementation handle the supported hardware. Imports, plots and
reports never start missing experiments.

## Device selection and scheduling

Devices are selected once before run creation, in this order: **CUDA → MPS → CPU**.

| Available hardware | Arm scheduling | Training microbatch × accumulation | Evaluation batch |
|---|---|---|---|
| At least two CUDA GPUs, each selected GPU with at least 24 GiB reported memory | LoRA on the first GPU, QLoRA on the second; concurrent | 4 × 4 | 8 |
| One CUDA GPU with at least 24 GiB reported memory | LoRA then QLoRA; serial | 4 × 4 | 8 |
| Smaller CUDA GPU(s) | Parallel only when two are available; otherwise serial | 1 × 16 | 1 |
| MPS | Serial | 1 × 16 | 1 |
| CPU | Serial | 1 × 16 | 1 |

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

Use a notebook kernel with **Python 3.12 or 3.13** on Linux x86_64 or macOS arm64.
Python 3.13 is preferred. Upload and extract the complete project, then open
`notebooks/lora_qlora.ipynb`.

The first cells set `PROJECT_ROOT` explicitly, change the working directory, and
install dependencies directly into the current notebook Python:

```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")
])
```

The notebook defaults to `/workspace/structured-request-adaptation`; edit that
one path if your project is elsewhere. No separate environment or kernel setup
is required. If installation updates packages already imported in the kernel,
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

### Upload files

Upload and extract the current full project ZIP into `/workspace`, or copy these
files into `/workspace/structured-request-adaptation/`:

| Files | Purpose |
|---|---|
| `src/`, `notebooks/`, `requirements.txt` | Implementation, notebook and dependencies |
| `README.md` | Setup and protocol instructions |
| `data/` | About 42 MiB of verified dataset files; avoids downloading again |
| `tests/`, `ruff.toml`, `verification/`, `.gitignore` | Optional checks and repository settings |

Copy the complete `data/massive-1.1/en-US/` directory: `archive.tar.gz`,
`en-US.jsonl`, `LICENSE`, and `manifest.json`. Keep names and contents unchanged;
do not unpack `archive.tar.gz`. Alternatively omit `data/` entirely and let the
dataset-acquisition cell download it. A partial dataset fails integrity checks.
Do not upload old Python environments, caches, credentials or prior job folders.
The notebook creates `runs/` and `jobs/` when needed. Save results before deleting
the remote machine or its storage.

### Quick test and full comparison

The notebook starts with this configuration:

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
runs skip official test evaluation and report its results as unavailable. Remove
`preset="development"` and create a fresh run for the full five-epoch comparison;
do not alter an existing run's frozen configuration.

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

The notebook defaults to the **one-epoch development preset**. The full protocol
uses **five epochs**, all **11,514 training / 2,033 validation / 2,974 test**
requests and seed **2027**. Each full training arm performs **3,600 optimizer
updates**. Remove `preset="development"` and create a fresh run to use it.
Development runs cannot evaluate the official test set.

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
notebooks/lora_qlora.ipynb    one notebook entry point
tests/                       bounded correctness and orchestration checks
requirements.txt             direct dependencies and platform markers
README.md                    setup, protocol and recovery
verification/                current local verification receipts
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

Set `RUN_TO_RESUME` to an exact `run_XXX` name and rerun the relevant notebook cells.
Source, dependencies, configuration, backend and artifact identities must match.
The controller records explicit compatible replacement sessions when physical
GPU UUIDs change. It does not promise bitwise reproducibility across hosts.
Completed stages reuse verified outputs instead of recomputing them.

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

## Verification and limitations

Run bounded tests with `python -m pytest tests`. Current checks are
recorded in [`verification/checks.md`](verification/checks.md). Tests use fixtures
and tiny CPU models; they do not establish full-model accelerator compatibility,
long-run memory fit or completion time. Execute preflight and inspect the pilot
on the chosen hardware. A single paired seed supports a measured case study,
not a claim of general superiority or training-seed stability.

This is a controlled adaptation of [LoRA](https://arxiv.org/abs/2106.09685),
[QLoRA](https://arxiv.org/abs/2305.14314) and
[MASSIVE](https://arxiv.org/abs/2204.08582), using ordinary AdamW rather than the
QLoRA paper's paged optimizer. Model architecture and tokenizer come from the
pinned [official model repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
