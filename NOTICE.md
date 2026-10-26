# Attribution and third-party material

Original project code and documentation are licensed under the
[MIT License](LICENSE), copyright © 2026 Gourav Karwasara. This license does not
replace the terms covering the dataset, model files or third-party dependencies.

## MASSIVE 1.1

The experiment uses the official MASSIVE 1.1 English (`en-US`) dataset by Amazon.
Copyright Amazon.com Inc. or its affiliates. The dataset is provided under
**Creative Commons Attribution 4.0 International**, including its disclaimer of
warranties. The original notice and complete license text are preserved in
[licenses/MASSIVE-CC-BY-4.0.txt](licenses/MASSIVE-CC-BY-4.0.txt).

- [Dataset repository and attribution](https://github.com/alexa/massive)
- [Original notice](https://github.com/alexa/massive/blob/main/NOTICE.md)
- [MASSIVE paper](https://arxiv.org/abs/2204.08582)
- [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

Project transformations select `en-US`, retain the official train/validation/test
partitions, and convert annotations into JSON intent/slot targets. Utterance and
slot-value text are preserved; derived records add deterministic targets,
duplicate-audit information and evaluation fields. Dataset excerpts in notebook
outputs and representative errors remain subject to the dataset license. Model
predictions are labeled separately from reference annotations. No endorsement by
Amazon is implied.

The full downloaded dataset is excluded from Git. Its acquisition cell records
SHA-256 checksums and preserves the original license alongside the local cache.

## Qwen2.5-1.5B-Instruct

The frozen base model and tokenizer are from the Qwen Team's
[Qwen2.5-1.5B-Instruct repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its saved model license is
**Apache License 2.0**, copied unchanged into
[licenses/Qwen-Apache-2.0.txt](licenses/Qwen-Apache-2.0.txt).

Base model weights, quantized snapshots and trained adapters are excluded from
Git. They are not relicensed by this project's MIT License. The notebook acquires
the pinned upstream files and records the source revision and file checksums.

## Methods and dependencies

This project implements a controlled adaptation of
[LoRA](https://arxiv.org/abs/2106.09685) and
[QLoRA](https://arxiv.org/abs/2305.14314), with ordinary AdamW in place of the QLoRA
paper's paged optimizer. These citations acknowledge the methods; they do not
imply endorsement or a reproduction of every condition in those papers.

PyTorch, Transformers, PEFT, bitsandbytes and other dependencies retain their
respective licenses. `requirements.txt` specifies direct dependencies; the
recorded run preserves the versions actually installed.
