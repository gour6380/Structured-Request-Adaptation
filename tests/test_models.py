"""Structural tests use tiny local modules, never model downloads."""

from types import SimpleNamespace

import pytest
import torch

from src.config import ExperimentConfig
from src.models import _adapter_inventory, _check_adapter_load, _shared_precision


class TinyTiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embedding = torch.nn.Embedding(11, 4, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(4, 11, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.embedding.weight
        self.q_proj = torch.nn.Linear(4, 4, bias=True, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


def test_matched_precision_preserves_projection_storage_and_tied_head():
    model = TinyTiedModel()
    run = SimpleNamespace(config=ExperimentConfig())
    _shared_precision(run, model)
    assert model.q_proj.weight.dtype == torch.bfloat16
    assert model.q_proj.bias.dtype == torch.float32
    assert model.norm.weight.dtype == torch.float32
    assert model.embedding.weight.dtype == torch.float32
    assert model.embedding.weight is model.lm_head.weight


def test_broken_tied_embeddings_are_not_silently_accepted():
    model = TinyTiedModel()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    with pytest.raises(ValueError, match="remain tied"):
        _shared_precision(SimpleNamespace(config=ExperimentConfig()), model)


def test_incomplete_adapter_targets_fail():
    model = TinyTiedModel()
    model.config.num_hidden_layers = 1
    with pytest.raises(ValueError, match="Adapter target mismatch"):
        _adapter_inventory(SimpleNamespace(config=ExperimentConfig()), model)


def test_missing_adapter_keys_are_rejected_even_with_expected_missing_base_keys():
    with pytest.raises(ValueError, match="lora_A"):
        _check_adapter_load(
            SimpleNamespace(
                missing_keys=["base.weight", "projection.lora_A.default.weight"], unexpected_keys=[]
            )
        )
    _check_adapter_load(SimpleNamespace(missing_keys=["base.weight"], unexpected_keys=[]))


def test_both_base_formats_reconstruct_and_share_identical_initial_adapters(tmp_path, monkeypatch):
    """Real tiny CPU NF4 round trip; no Hub access or pretrained weights required.

    This test is authored for an explicitly authorized later test run. It executes
    both unquantized and bitsandbytes CPU quantization paths, never a mock quantizer.
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config, Qwen2ForCausalLM

    import src.models as models
    from src.io import atomic_json, read_json, sha256_file

    class Run:
        def __init__(self):
            self.path = tmp_path
            self.config = ExperimentConfig()
            self.environment = {
                "device": "cpu",
                "compute_dtype": "float32",
                "use_grad_scaler": False,
            }
            atomic_json(self.path / "manifest.json", {"artifacts": {}})

        def verify_identity(self):
            pass

        def require_stage(self, name):
            pass

        def register_artifacts(self, paths):
            manifest = read_json(self.path / "manifest.json")
            for path in paths:
                manifest["artifacts"][str(path.relative_to(self.path))] = {
                    "sha256": sha256_file(path)
                }
            atomic_json(self.path / "manifest.json", manifest)

        def verify_artifacts(self, prefixes=None):
            for name, record in read_json(self.path / "manifest.json")["artifacts"].items():
                if prefixes is None or any(
                    name == prefix or name.startswith(prefix + "/") for prefix in prefixes
                ):
                    assert sha256_file(self.path / name) == record["sha256"]

    run = Run()
    torch.manual_seed(17)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    base = Qwen2ForCausalLM(config)
    base_path = tmp_path / "inputs" / "base_model"
    base.save_pretrained(base_path, safe_serialization=True)
    del base
    quantized = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        dtype=torch.float32,
        device_map={"": "cpu"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
            llm_int8_skip_modules=["lm_head"],
        ),
    )
    assert len(models._quantized_layers(quantized)) == 7
    quantized.save_pretrained(tmp_path / "inputs" / "quantized_base", safe_serialization=True)
    del quantized
    atomic_json(
        tmp_path / "inputs" / "quantization.json", {"backend": models._quantization_backend(run)}
    )
    run.register_artifacts(models._files(tmp_path / "inputs"))
    monkeypatch.setattr(models, "_tokenizer", lambda _: None)
    initial = models._prepare_adapter(run, models._base(run, False), "lora")
    initial.save_pretrained(
        tmp_path / "initial_adapter", safe_serialization=True, save_embedding_layers=False
    )
    run.register_artifacts(models._files(tmp_path / "initial_adapter"))
    initial_state = {
        name: value.clone() for name, value in get_peft_model_state_dict(initial).items()
    }
    del initial
    inputs = torch.tensor([[1, 2, 3, 4, 5]])
    for arm in ("lora", "qlora"):
        model, _ = models.load_system(run, arm, checkpoint="initial", training=True)
        state = get_peft_model_state_dict(model)
        assert state.keys() == initial_state.keys()
        for name in state:
            torch.testing.assert_close(state[name], initial_state[name], rtol=0, atol=0)
        frozen = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if not value.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad], lr=0.01
        )
        model(input_ids=inputs, labels=inputs, use_cache=False).loss.backward()
        assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name, value in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        model.eval()
        model.gradient_checkpointing_disable()
        with torch.inference_mode():
            expected = model(input_ids=inputs, use_cache=False).logits.clone()
        trained_path = tmp_path / f"test_{arm}_adapter"
        model.save_pretrained(trained_path, safe_serialization=True, save_embedding_layers=False)
        del model, optimizer, state, frozen
        restored, _ = models.load_system(run, arm, checkpoint="initial", training=True)
        result = set_peft_model_state_dict(
            restored, load_file(trained_path / "adapter_model.safetensors")
        )
        _check_adapter_load(result)
        restored.eval()
        restored.gradient_checkpointing_disable()
        with torch.inference_mode():
            actual = restored(input_ids=inputs, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
        del restored


def _rotary_container():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    config = Qwen2Config(hidden_size=1536, num_attention_heads=12, rope_theta=1000000.0)
    container = torch.nn.Module()
    container.rotary = Qwen2RotaryEmbedding(config)
    container.register_buffer("sentinel", torch.tensor([1.0, 2.0]))
    return container


def test_rotary_guard_is_idempotent_and_preserves_cpu_outputs_state_and_deepcopy():
    import copy

    from src.models import _install_mps_rotary_guard

    model = _rotary_container()
    x = torch.zeros(1, 5, 1536, dtype=torch.bfloat16)
    positions = torch.tensor([[0, 255, 257, 511, 639]])
    before = model.rotary(x, positions)
    keys = list(model.state_dict())
    buffer_dtypes = {name: tensor.dtype for name, tensor in model.named_buffers()}
    original = model.rotary.forward
    assert _install_mps_rotary_guard(model) == ["rotary"]
    guarded = model.rotary.forward
    assert _install_mps_rotary_guard(model) == ["rotary"]
    assert model.rotary.forward is guarded
    assert model.rotary._forward_without_mps_rope_guard == original
    assert list(model.state_dict()) == keys
    assert {name: tensor.dtype for name, tensor in model.named_buffers()} == buffer_dtypes
    after = model.rotary(x, positions)
    for expected, actual in zip(before, after, strict=True):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cloned = copy.deepcopy(model)
    assert cloned.rotary.forward.__self__ is cloned.rotary
    assert cloned.rotary._forward_without_mps_rope_guard.__self__ is cloned.rotary
    for expected, actual in zip(before, cloned.rotary(x, positions), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rotary_guard_disables_only_mps_autocast_and_restores_context(monkeypatch):
    from contextlib import contextmanager

    from src.models import _mps_fp32_rotary_forward

    calls = []

    @contextmanager
    def autocast(**kwargs):
        assert kwargs == {"device_type": "mps", "enabled": False}
        calls.append("enter-disabled")
        try:
            yield
        finally:
            calls.append("restore")

    def original(x, position_ids):
        assert calls == ["enter-disabled"]
        calls.append("original")
        assert position_ids == "positions"
        return "original-output"

    monkeypatch.setattr(torch, "autocast", autocast)
    module = SimpleNamespace(_forward_without_mps_rope_guard=original)
    assert (
        _mps_fp32_rotary_forward(
            module, SimpleNamespace(device=SimpleNamespace(type="mps")), "positions"
        )
        == "original-output"
    )
    assert calls == ["enter-disabled", "original", "restore"]


def test_rotary_guard_leaves_non_mps_autocast_untouched(monkeypatch):
    from src.models import _mps_fp32_rotary_forward

    monkeypatch.setattr(torch, "autocast", lambda **_: pytest.fail("CPU context was changed"))
    module = SimpleNamespace(_forward_without_mps_rope_guard=lambda *args, **kwargs: (args, kwargs))
    x = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    assert _mps_fp32_rotary_forward(module, x, "positions", extra=True) == (
        (x, "positions"),
        {"extra": True},
    )


def test_preflight_rotary_check_covers_long_positions_and_detects_corruption():
    from src.models import _check_rotary_precision, _install_mps_rotary_guard

    model = _rotary_container()
    _install_mps_rotary_guard(model)
    run = SimpleNamespace(
        config=ExperimentConfig(), environment={"device": "cpu", "compute_dtype": "float32"}
    )
    result = _check_rotary_precision(run, model)
    assert result["passed"]
    assert {255, 256, 257, 511, 512, 513, 639, 1023, 2047}.issubset(result["positions"])
    assert result["maximum_absolute_errors"]["rotary"]["torch.float32"] == 0.0
    original = model.rotary.forward

    def corrupted(x, positions):
        cos, sin = original(x, positions)
        return cos + 0.5, sin

    model.rotary.forward = corrupted
    with pytest.raises(AssertionError):
        _check_rotary_precision(run, model)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_mps_rotary_guard_matches_fp32_frequency_reference(dtype):
    """Opt in explicitly with RUN_NATIVE_MPS_TESTS=1; never runs a full model."""
    import os

    from src.models import _install_mps_rotary_guard

    if os.environ.get("RUN_NATIVE_MPS_TESTS") != "1" or not torch.backends.mps.is_available():
        pytest.skip("Explicit native MPS rotary verification was not requested/available")
    model = _rotary_container()
    positions = torch.tensor([[0, 255, 256, 257, 511, 512, 513, 639, 1023, 2047]])
    x = torch.zeros(1, positions.shape[1], 1536, dtype=dtype)
    reference = model.rotary(x, positions)
    model.to("mps")
    _install_mps_rotary_guard(model)
    with torch.autocast("mps", dtype=torch.bfloat16):
        actual = model.rotary(x.to("mps"), positions.to("mps"))
        assert torch.is_autocast_enabled("mps")
    for expected, value in zip(reference, actual, strict=True):
        assert value.dtype == dtype
        torch.testing.assert_close(
            value.cpu(), expected, rtol=0, atol=0.004 if dtype == torch.bfloat16 else 2e-5
        )
