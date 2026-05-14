"""Phase 4 gate: shape smoke + param count."""
import torch

from specdec_af.models.prefix_encoder import PrefixEncoder


def test_shape_smoke():
    pe = PrefixEncoder()
    x = torch.randn(4, 12 * 768)
    y = pe(x)
    assert y.shape == (4, 512)


def test_distinct_prefixes_distinct_outputs():
    pe = PrefixEncoder()
    x = torch.randn(2, 12 * 768)
    y = pe(x)
    # Different inputs should produce different outputs at init.
    assert not torch.allclose(y[0], y[1])


def test_param_count():
    pe = PrefixEncoder()
    total = sum(p.numel() for p in pe.parameters())
    # Linear(9216, 512) = 9216*512 + 512 = 4,719,104 ; GELU has no params.
    expected = 9216 * 512 + 512
    assert total == expected, f"got {total}, expected {expected}"
