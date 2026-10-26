"""The frozen, editable notebook experiment configuration."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

SYSTEMS = ("base", "quantized_base", "lora", "qlora")
ARMS = ("lora", "qlora")
SPLITS = ("train", "validation", "test")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class ExperimentConfig:
    """No environment variables or CLI flags override these settings."""

    protocol_version: str = "1.0"
    preset: str = "full"
    train_example_limit: int | None = None
    validation_example_limit: int | None = None
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = ""  # Resolved to an immutable Hub commit before run creation.
    dataset_url: str = (
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
    )
    locale: str = "en-US"
    seed: int = 2027
    epochs: int = 5
    microbatch_size: int = 1
    gradient_accumulation: int = 16
    evaluation_batch_size: int = 1
    answer_only_logits: bool = True
    learning_rate: float = 2e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    gradient_clip: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_MODULES
    max_sequence_length: int = 2048
    max_new_tokens: int = 256
    checkpoint_interval: int = 50
    benchmark_requests: int = 100
    benchmark_warmup: int = 5
    benchmark_repetitions: int = 3
    telemetry_interval_seconds: float = 0.1
    expected_split_counts: tuple[int, int, int] = (11514, 2033, 2974)

    def validate(self, *, require_revision: bool = True) -> None:
        if self.preset not in ("full", "development"):
            raise ValueError("preset must be full or development")
        limits = (
            ("train_example_limit", self.train_example_limit, 11514),
            ("validation_example_limit", self.validation_example_limit, 2033),
        )
        for name, value, maximum in limits:
            if self.preset == "full":
                if value is not None:
                    raise ValueError("The full preset must consume all official examples")
            elif type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Development {name} must be an integer from 1 to {maximum}")
        if self.model_id != "Qwen/Qwen2.5-1.5B-Instruct" or self.locale != "en-US":
            raise ValueError("This protocol implements Qwen2.5-1.5B-Instruct and MASSIVE en-US")
        if require_revision and not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Resolve the model revision before creating a run")
        for name in (
            "epochs",
            "microbatch_size",
            "gradient_accumulation",
            "evaluation_batch_size",
            "lora_rank",
            "lora_alpha",
            "max_sequence_length",
            "max_new_tokens",
            "checkpoint_interval",
            "benchmark_requests",
            "benchmark_warmup",
            "benchmark_repetitions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.preset == "full" and self.effective_batch_size != 16:
            raise ValueError("The full protocol requires an effective batch of 16 examples")
        if type(self.answer_only_logits) is not bool:
            raise ValueError("answer_only_logits must be a boolean")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
            or self.max_new_tokens >= self.max_sequence_length
        ):
            raise ValueError("Invalid seed or sequence/generation budget")
        if tuple(self.target_modules) != TARGET_MODULES:
            raise ValueError("Both arms must target the seven registered projection types")
        if tuple(self.expected_split_counts) != (11514, 2033, 2974):
            raise ValueError("Official MASSIVE split counts must remain fixed")
        if self.preset == "development" and (
            self.benchmark_requests > self.validation_example_limit
            or self.benchmark_warmup > self.benchmark_requests
        ):
            raise ValueError("Development benchmark requests/warmup exceed the validation sample")
        for name in (
            "learning_rate",
            "adam_epsilon",
            "gradient_clip",
            "telemetry_interval_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Warmup and dropout must lie in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Invalid AdamW beta values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimizer update, except the retained final partial window."""
        return self.microbatch_size * self.gradient_accumulation

    @classmethod
    def for_hardware(cls, hardware: dict[str, Any], **overrides: Any) -> ExperimentConfig:
        """Choose explicit batching once from the discovered execution plan."""
        from .hardware import validate_hardware_plan

        validate_hardware_plan(hardware)
        if overrides.get("preset") == "development":
            return cls.development(**{**hardware["batch_defaults"], **overrides})
        values = {
            "preset": "full",
            "epochs": 5,
            **hardware["batch_defaults"],
        }
        values.update(overrides)
        result = cls(**values)
        result.validate(require_revision=bool(result.model_revision))
        return result

    @classmethod
    def development(cls, **overrides: Any) -> ExperimentConfig:
        """Small paired workflow check; it does not support held-out test claims."""
        values = {
            "preset": "development",
            "epochs": 1,
            "train_example_limit": 128,
            "validation_example_limit": 32,
            "benchmark_requests": 16,
            "benchmark_warmup": 2,
            "benchmark_repetitions": 1,
        }
        values.update(overrides)
        result = cls(**values)
        result.validate(require_revision=bool(result.model_revision))
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        values = dict(data)
        for key in ("adam_betas", "target_modules", "expected_split_counts"):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result


def resolve_model_revision(config: ExperimentConfig) -> ExperimentConfig:
    """Explicit metadata-only notebook action; never download model weights."""
    from dataclasses import replace

    from huggingface_hub import HfApi

    config.validate(require_revision=False)
    if config.model_revision:
        config.validate()
        return config
    info = HfApi().model_info(config.model_id, revision="main")
    result = replace(config, model_revision=info.sha or "")
    result.validate()
    return result
