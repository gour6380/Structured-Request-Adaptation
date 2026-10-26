# Execution context

Backend: cuda. Arm execution: concurrent. Compute dtype: bfloat16.

Assigned devices: lora: NVIDIA A40 (44.42 GiB); qlora: NVIDIA A40 (44.42 GiB). Selected GPU models and capacities match.

Each arm has its own process. CUDA workers see only their assigned GPU as local CUDA 0. CPU and MPS arms run serially. The arms share saved initialization, data order, effective batch and update budget.

Concurrent workers share host CPU, RAM and storage; timings may include contention. Memory samples are observed maxima, not guaranteed allocation peaks.

Original identities are in backend_binding.json and environment.json. Compatible replacement sessions are recorded in sessions/session_*.json; this does not imply bitwise cross-device reproducibility.
