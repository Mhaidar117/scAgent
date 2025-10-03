"""Quality gates and assertions for scQC Agent."""

from .assertions import (
    assert_qc_fields_present,
    assert_pct_mt_range,
    assert_neighbors_nonempty,
    assert_latent_shape,
    QualityGateError,
)

__all__ = [
    "assert_qc_fields_present",
    "assert_pct_mt_range", 
    "assert_neighbors_nonempty",
    "assert_latent_shape",
    "QualityGateError",
]
