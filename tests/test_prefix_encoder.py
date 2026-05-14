"""Phase 4 gate: shape smoke + param count + config round-trip."""
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
    assert not torch.allclose(y[0], y[1])


def test_param_count_default_deep():
    """Default deep MLP: 9216 → 2048 → 1024 → 512. Verify exact count.

    Breakdown:
      Linear(9216, 2048) + bias = 9216*2048 + 2048 = 18,876,416
      LayerNorm(2048)            = 4,096
      Linear(2048, 1024) + bias  = 2*1024*1024 + 1024 = 2,098,176
      LayerNorm(1024)            = 2,048
      Linear(1024, 512) + bias   = 1024*512 + 512 = 524,800
      ---------------------------------------------------
      Total                       = 21,505,536
    """
    pe = PrefixEncoder()
    total = sum(p.numel() for p in pe.parameters())
    expected = 18_876_416 + 4_096 + 2_098_176 + 2_048 + 524_800
    assert total == expected, f"got {total:,}, expected {expected:,}"


def test_param_count_shallow_baseline():
    """`hidden_dims=()` recovers the original shallow Linear+GELU baseline."""
    pe = PrefixEncoder(hidden_dims=())
    total = sum(p.numel() for p in pe.parameters())
    expected = 9216 * 512 + 512  # 4,719,104
    assert total == expected, f"got {total:,}, expected {expected:,}"


def test_get_config_roundtrip():
    """get_config() returns the constructor args. Round-trip-able."""
    pe = PrefixEncoder(hidden_dims=(1024,), d_out=256)
    cfg = pe.get_config()
    pe2 = PrefixEncoder(**cfg)
    assert pe2.get_config() == cfg
    # Param counts match (architecture is recovered).
    assert sum(p.numel() for p in pe.parameters()) == sum(p.numel() for p in pe2.parameters())
