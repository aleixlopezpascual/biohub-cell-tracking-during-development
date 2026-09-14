from __future__ import annotations

import pytest

from scripts.continue_royerlab_training import _install_empty_attention_guard, _restore_rng


def test_empty_attention_guard_keeps_all_masked_batch_row_finite() -> None:
    torch = pytest.importorskip("torch")

    class AttentionBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(4, 1, batch_first=True)

        def forward(self, q, kv, kv_mask=None):
            output, _ = self.attention(
                q,
                kv,
                kv,
                key_padding_mask=None if kv_mask is None else ~kv_mask,
            )
            return output

    class Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([AttentionBlock(), AttentionBlock()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = Transformer()

    model = Model()
    assert _install_empty_attention_guard(model) == 2
    assert _install_empty_attention_guard(model) == 0

    query = torch.randn(2, 2, 4, requires_grad=True)
    key_value = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[False, False, False], [True, True, False]])
    original_key_value = key_value.detach().clone()
    original_mask = mask.clone()

    output = model.transformer.blocks[0](query, key_value, kv_mask=mask)
    output.sum().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key_value.grad).all()
    assert torch.equal(key_value.detach(), original_key_value)
    assert torch.equal(mask, original_mask)


def test_restore_rng_moves_mapped_generator_states_back_to_cpu() -> None:
    torch = pytest.importorskip("torch")
    original_cpu = torch.get_rng_state()
    original_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        cpu_state = original_cpu.to("cuda") if torch.cuda.is_available() else original_cpu.tolist()
        state: dict[str, object] = {"torch_cpu": cpu_state}
        if torch.cuda.is_available() and original_cuda is not None:
            state["torch_cuda"] = [value.to("cuda") for value in original_cuda]

        _restore_rng(torch, state)

        assert torch.equal(torch.get_rng_state(), original_cpu)
        if original_cuda is not None:
            assert all(
                torch.equal(actual, expected)
                for actual, expected in zip(torch.cuda.get_rng_state_all(), original_cuda)
            )
    finally:
        torch.set_rng_state(original_cpu)
        if original_cuda is not None:
            torch.cuda.set_rng_state_all(original_cuda)
