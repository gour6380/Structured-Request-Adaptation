# Structured request adaptation — run_001

Status: COMPLETE measured comparison.

One paired seed (2027), official MASSIVE en-US splits, Qwen/Qwen2.5-1.5B-Instruct. Primary adapted-model results use the final checkpoint after 5 full epochs. Validation-selected best checkpoints are supplementary and are not silently substituted.

## Frozen configuration and provenance

```json
{
  "adam_betas": [
    0.9,
    0.999
  ],
  "adam_epsilon": 1e-08,
  "answer_only_logits": true,
  "benchmark_repetitions": 3,
  "benchmark_requests": 100,
  "benchmark_warmup": 5,
  "checkpoint_interval": 50,
  "dataset_url": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
  "epochs": 5,
  "evaluation_batch_size": 32,
  "expected_split_counts": [
    11514,
    2033,
    2974
  ],
  "gradient_accumulation": 1,
  "gradient_clip": 1.0,
  "learning_rate": 0.0002,
  "locale": "en-US",
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "lora_rank": 16,
  "max_new_tokens": 256,
  "max_sequence_length": 2048,
  "microbatch_size": 16,
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "preset": "full",
  "protocol_version": "1.0",
  "seed": 2027,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "telemetry_interval_seconds": 0.1,
  "train_example_limit": null,
  "validation_example_limit": null,
  "warmup_ratio": 0.03,
  "weight_decay": 0.0
}
```

Immutable model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Execution backend: `cuda`; compute dtype: `bfloat16`.

The exact OS, Python, backend, package versions and runtime flags are recorded in `../environment.json`; source and pinned dependency files are copied into `../source_snapshot/` with hashes. `../identity.json`, `../manifest.json` and `../evaluations/frozen_protocol.json` identify this run and its registered artifacts.

## Protocol and interpretation

LoRA uses the unquantized frozen backbone; QLoRA uses the frozen quantized backbone. Compare each adapted arm to its own unadapted base, and compare base to quantized_base to expose the quantization effect. A direct LoRA–QLoRA difference alone does not isolate adaptation from quantization.

Whole-record exact match requires valid JSON, the known schema, correct intent and exact case-sensitive slot-value/type multisets. Slot order is ignored, duplicate occurrences retained. Invalid generations stay in denominators; no JSON repair, retry or constrained decoding is used. The novel-input subset excludes normalized utterances observed in either training or validation.

## Official test metrics

| System | N | Record exact | JSON valid | Schema valid | Intent acc. | Intent macro F1 | Slot P | Slot R | Slot F1 | Answer NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 2974 | 0.0568 | 0.8221 | 0.5928 | 0.3285 | 0.3480 | 0.1416 | 0.1517 | 0.1465 | 0.8251 |
| quantized_base | 2974 | 0.0027 | 0.0746 | 0.0319 | 0.0182 | 0.0272 | 0.1386 | 0.0082 | 0.0154 | 1.0221 |
| lora | 2974 | 0.7246 | 0.9997 | 0.9956 | 0.8995 | 0.8714 | 0.8090 | 0.8171 | 0.8130 | 0.1009 |
| qlora | 2974 | 0.7206 | 1.0000 | 0.9966 | 0.8927 | 0.8694 | 0.8074 | 0.8188 | 0.8131 | 0.1031 |

## Novel-input subset and failures

| System | Novel N | Record exact | JSON valid | Schema valid | Intent acc. | Intent macro F1 | Slot P | Slot R | Slot F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 2948 | 0.0570 | 0.8216 | 0.5926 | 0.3277 | 0.3474 | 0.1423 | 0.1523 | 0.1471 |
| quantized_base | 2948 | 0.0027 | 0.0746 | 0.0315 | 0.0176 | 0.0270 | 0.1402 | 0.0082 | 0.0156 |
| lora | 2948 | 0.7239 | 0.9997 | 0.9956 | 0.8993 | 0.8713 | 0.8088 | 0.8169 | 0.8128 |
| qlora | 2948 | 0.7201 | 1.0000 | 0.9966 | 0.8928 | 0.8695 | 0.8071 | 0.8187 | 0.8129 |

- base: novel n=2948, whole-record exact=0.0570, intent macro F1=0.3474, slot F1=0.1471; failure categories={'extra_slots': 1611, 'invalid_json': 529, 'invalid_structure': 192, 'missing_slots': 1511, 'unknown_intent': 131, 'unknown_slot_type': 413, 'wrong_intent': 1276, 'wrong_values': 422}.
- quantized_base: novel n=2948, whole-record exact=0.0027, intent macro F1=0.0270, slot F1=0.0156; failure categories={'extra_slots': 100, 'invalid_json': 2752, 'invalid_structure': 96, 'missing_slots': 1960, 'unknown_intent': 5, 'unknown_slot_type': 27, 'wrong_intent': 72, 'wrong_values': 16}.
- lora: novel n=2948, whole-record exact=0.7239, intent macro F1=0.8713, slot F1=0.8128; failure categories={'extra_slots': 359, 'invalid_json': 1, 'missing_slots': 337, 'unknown_intent': 4, 'unknown_slot_type': 8, 'wrong_intent': 298, 'wrong_values': 143}.
- qlora: novel n=2948, whole-record exact=0.7201, intent macro F1=0.8695, slot F1=0.8129; failure categories={'extra_slots': 370, 'missing_slots': 334, 'unknown_intent': 7, 'unknown_slot_type': 4, 'wrong_intent': 319, 'wrong_values': 145}.

## Runtime and memory

Benchmark: the same seeded 100 validation requests for all systems, 5 excluded warmups, 3 repetitions, batch size one. Median/p95 include prompt tokenization, device transfer, generation and decoding, with device synchronization. Correctness is recorded for every request in every repetition. Generated-token throughput includes EOS tokens if emitted; model loading and scoring are outside the latency interval.

- base: median=1.0523 s, p95=1.7431 s, generated tokens/s=38.7374; per-repetition record accuracy=[0.02, 0.02, 0.02]; memory={'cuda_allocated_bytes': 4064040960, 'cuda_reserved_bytes': 4129292288, 'elapsed_seconds': 346.5027754427865, 'measurement': 'maximum_observed_samples', 'qualification': 'RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.', 'rss_bytes': 1866530816, 'sample_interval_seconds': 0.1, 'samples': 3431, 'sampling_errors': [], 'system_swap_activity_observed': False, 'system_swap_out_bytes_delta': 0}.
- quantized_base: median=1.6411 s, p95=2.5409 s, generated tokens/s=27.6167; per-repetition record accuracy=[0.0, 0.0, 0.0]; memory={'cuda_allocated_bytes': 2121381888, 'cuda_reserved_bytes': 2604662784, 'elapsed_seconds': 530.6901934589259, 'measurement': 'maximum_observed_samples', 'qualification': 'RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.', 'rss_bytes': 1921851392, 'sample_interval_seconds': 0.1, 'samples': 5251, 'sampling_errors': [], 'system_swap_activity_observed': False, 'system_swap_out_bytes_delta': 0}.
- lora: median=0.9665 s, p95=2.0159 s, generated tokens/s=20.3698; per-repetition record accuracy=[0.83, 0.83, 0.83]; memory={'cuda_allocated_bytes': 3710695424, 'cuda_reserved_bytes': 4192206848, 'elapsed_seconds': 306.0436582551338, 'measurement': 'maximum_observed_samples', 'qualification': 'RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.', 'rss_bytes': 1929588736, 'sample_interval_seconds': 0.1, 'samples': 3030, 'sampling_errors': [], 'system_swap_activity_observed': False, 'system_swap_out_bytes_delta': 0}.
- qlora: median=1.3229 s, p95=2.7635 s, generated tokens/s=14.7573; per-repetition record accuracy=[0.82, 0.82, 0.82]; memory={'cuda_allocated_bytes': 2193467904, 'cuda_reserved_bytes': 2680160256, 'elapsed_seconds': 419.64607323380187, 'measurement': 'maximum_observed_samples', 'qualification': 'RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.', 'rss_bytes': 1961549824, 'sample_interval_seconds': 0.1, 'samples': 4152, 'sampling_errors': [], 'system_swap_activity_observed': False, 'system_swap_out_bytes_delta': 0}.
- lora training: epochs=5, updates=3600, target tokens=1172405, training loop seconds=9249.854290399235, wall seconds=11185.990893645212; memory={'cuda_allocated_bytes': 7560002560, 'cuda_reserved_bytes': 12855541760, 'elapsed_seconds': 0.0016513369046151638, 'measurement': 'maximum_observed_samples', 'qualification': 'RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.', 'rss_bytes': 3149119488, 'sample_interval_seconds': 0.1, 'samples': 2, 'sampling_errors': [], 'system_swap_activity_observed': False, 'system_swap_out_bytes_delta': 0}.
- qlora training: epochs=5, updates=3600, target tokens=1172405, training loop seconds=11681.646294836886, wall seconds=14171.725563864224; memory={'cuda_allocated_bytes': 6124876288, 'cuda_reserved_bytes': 21632122880, 'elapsed_seconds': 0.002158941701054573, 'measurement': 'maximum_observed_samples', 'qualification': 'RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.', 'rss_bytes': 3027161088, 'sample_interval_seconds': 0.1, 'samples': 2, 'sampling_errors': [], 'system_swap_activity_observed': False, 'system_swap_out_bytes_delta': 0}.

RSS, MPS allocated/driver or CUDA allocated/reserved measures are overlapping and must not be added. Memory telemetry reports maximum observed samples, not an exact peak. System swap activity is machine-wide and is disclosed without attributing it to this process. Cold/model-loading latency is not measured by the warm benchmark.

## Paired comparisons

QLoRA minus LoRA metric deltas: {'joint_exact_match': -0.00403496973772699, 'schema_valid_rate': 0.0010087424344317197, 'intent_accuracy': -0.00672494956287828, 'intent_macro_f1': -0.0020272290435772167, 'slot_f1': 4.3016303178933235e-05, 'answer_token_nll': 0.002171886084877603}.
Primary whole-record exact-match difference: -0.403 percentage points. The paired run shows an observed quality drop for QLoRA. No non-inferiority margin or multi-seed uncertainty was registered, so this does not establish equivalent quality or statistical superiority.
- training/rss_bytes: QLoRA−LoRA=-0.1136 GiB; relative reduction=0.0387; observed reduction supports the expected memory benefit for this measure.
- training/cuda_allocated_bytes: QLoRA−LoRA=-1.3366 GiB; relative reduction=0.1898; observed reduction supports the expected memory benefit for this measure.
- training/cuda_reserved_bytes: QLoRA−LoRA=+8.1738 GiB; relative reduction=-0.6827; no observed reduction; the expected memory benefit is not supported for this measure.
- warm_inference/rss_bytes: QLoRA−LoRA=+0.0298 GiB; relative reduction=-0.0166; no observed reduction; the expected memory benefit is not supported for this measure.
- warm_inference/cuda_allocated_bytes: QLoRA−LoRA=-1.4130 GiB; relative reduction=0.4089; observed reduction supports the expected memory benefit for this measure.
- warm_inference/cuda_reserved_bytes: QLoRA−LoRA=-1.4082 GiB; relative reduction=0.3607; observed reduction supports the expected memory benefit for this measure.
Whole-fit samples include training and validation; loading, validation NLL and generation phase samples are stored separately. Interpret mixed memory measures separately and qualify any result with recorded swapping/sampling errors.
QLoRA minus LoRA runtime/throughput deltas (units match the source metric): {'training_seconds': 2431.7920044376515, 'wall_seconds': 2985.7346702190116, 'median_seconds': 0.356441140640527, 'p95_seconds': 0.7476169613888493, 'generated_tokens_per_second': -5.612456223806046}.

- base vs quantized_base: left-only correct=162, right-only correct=1, both correct=7, both wrong=2804.
- base vs lora: left-only correct=17, right-only correct=2003, both correct=152, both wrong=802.
- base vs qlora: left-only correct=19, right-only correct=1993, both correct=150, both wrong=812.
- quantized_base vs lora: left-only correct=1, right-only correct=2148, both correct=7, both wrong=818.
- quantized_base vs qlora: left-only correct=1, right-only correct=2136, both correct=7, both wrong=830.
- lora vs qlora: left-only correct=127, right-only correct=115, both correct=2028, both wrong=704.

## Representative error categories

First three errors by sorted example ID per system/category; an example can occur in several categories. Full utterances, gold records, raw output and token IDs are in `error_examples.json`.

- base / missing_slots: example `0`.
- base / extra_slots: example `0`.
- base / wrong_values: example `0`.
- base / invalid_structure: example `100`.
- base / missing_slots: example `100`.
- base / wrong_intent: example `10016`.
- base / missing_slots: example `10016`.
- base / extra_slots: example `10016`.
- base / extra_slots: example `10019`.
- base / wrong_values: example `10019`.
- base / invalid_json: example `10027`.
- base / invalid_structure: example `10031`.
- base / wrong_intent: example `10038`.
- base / wrong_intent: example `10053`.
- base / unknown_slot_type: example `10055`.
- base / unknown_intent: example `10076`.
- base / wrong_values: example `10076`.
- base / unknown_slot_type: example `10100`.
- base / invalid_json: example `10105`.
- base / invalid_json: example `10132`.
- base / unknown_slot_type: example `10166`.
- base / unknown_intent: example `10216`.
- base / unknown_intent: example `10286`.
- base / invalid_structure: example `10667`.
- quantized_base / invalid_json: example `0`.
- quantized_base / missing_slots: example `0`.
- quantized_base / invalid_json: example `100`.
- quantized_base / missing_slots: example `100`.
- quantized_base / invalid_structure: example `10016`.
- quantized_base / missing_slots: example `10016`.
- quantized_base / invalid_json: example `10019`.
- quantized_base / invalid_structure: example `10023`.
- quantized_base / invalid_structure: example `10061`.
- quantized_base / wrong_intent: example `10247`.
- quantized_base / extra_slots: example `10247`.
- quantized_base / wrong_intent: example `10260`.
- quantized_base / extra_slots: example `10292`.
- quantized_base / wrong_intent: example `11343`.
- quantized_base / extra_slots: example `11343`.
- quantized_base / wrong_values: example `11767`.
- quantized_base / unknown_slot_type: example `11867`.
- quantized_base / unknown_slot_type: example `12073`.
- quantized_base / wrong_values: example `12073`.
- quantized_base / unknown_slot_type: example `12121`.
- quantized_base / wrong_values: example `12306`.
- quantized_base / unknown_intent: example `1301`.
- quantized_base / unknown_intent: example `13140`.
- quantized_base / unknown_intent: example `14505`.
- lora / missing_slots: example `100`.
- lora / extra_slots: example `100`.
- lora / missing_slots: example `10016`.
- lora / extra_slots: example `10016`.
- lora / missing_slots: example `10031`.
- lora / wrong_values: example `10045`.
- lora / extra_slots: example `1007`.
- lora / wrong_intent: example `10076`.
- lora / wrong_values: example `10076`.
- lora / wrong_intent: example `10078`.
- lora / wrong_intent: example `10097`.
- lora / wrong_values: example `10406`.
- lora / unknown_slot_type: example `11825`.
- lora / unknown_slot_type: example `12835`.
- lora / unknown_slot_type: example `13513`.
- lora / invalid_json: example `151`.
- lora / unknown_intent: example `3080`.
- lora / unknown_intent: example `41`.
- lora / unknown_intent: example `6290`.
- qlora / missing_slots: example `100`.
- qlora / extra_slots: example `100`.
- qlora / wrong_values: example `10019`.
- qlora / missing_slots: example `10031`.
- qlora / wrong_intent: example `10061`.
- qlora / missing_slots: example `1007`.
- qlora / extra_slots: example `1007`.
- qlora / extra_slots: example `10076`.
- qlora / wrong_values: example `10076`.
- qlora / wrong_intent: example `10078`.
- qlora / wrong_intent: example `10097`.
- qlora / wrong_values: example `1047`.
- qlora / unknown_slot_type: example `11460`.
- qlora / unknown_intent: example `12104`.
- qlora / unknown_intent: example `13618`.
- qlora / unknown_slot_type: example `14170`.
- qlora / unknown_slot_type: example `5578`.
- qlora / unknown_intent: example `6290`.

## Limits and incomplete work

This is one paired seed on one task/model/hardware/version combination, without multi-seed uncertainty estimates or a universal superiority claim. The official test may contain normalized duplicates of train/dev inputs; both full and novel-input results are reported. No causal claim is made from incomparable training losses or unpaired prompts. Schema/intent/slot metrics assess this structured task, not general language-model ability.

All required stored evidence is present. Interpret observed differences within the limits above.

## Artifacts

Raw per-request predictions and token IDs: `evaluations/{system}/test/predictions.jsonl`. Benchmarks: `benchmarks/{system}/requests.jsonl`. Training/validation curves: `trials/{arm}/metrics/`. Figures are rendered only by explicit plot calls; this report does not fit, evaluate or benchmark any model.


Assigned devices: lora: NVIDIA A40 (44.42 GiB); qlora: NVIDIA A40 (44.42 GiB). Selected GPU models and capacities match.

Execution and timing qualifications: [execution context](execution.md).
