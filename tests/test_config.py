from pathlib import Path

import pytest

from ree_prospectivity.config import DownscaleConfig, ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "split_policy", "strategy"),
    [
        ("paper_figure.toml", "paper_faithful", "legacy_diagonal"),
        ("leakage_safe.toml", "group_before_augmentation", "uniform"),
    ],
)
def test_versioned_configurations_load(
    filename: str,
    split_policy: str,
    strategy: str,
) -> None:
    config = ExperimentConfig.from_toml(PROJECT_ROOT / "configs" / filename)

    assert config.data.channels == 28
    assert config.generator.scale_factor == 4
    assert config.split_policy == split_policy
    assert config.downscale.strategy == strategy


def test_legacy_downscale_rejects_non_historical_shape() -> None:
    with pytest.raises(ValueError, match="legacy_diagonal requires"):
        DownscaleConfig(
            strategy="legacy_diagonal",
            block_size=2,
            samples_per_block=2,
        )
