import numpy as np
import torch

from ree_prospectivity.reproducibility import seed_everything


def test_seed_everything_replays_numpy_and_torch_sequences() -> None:
    seed_everything(42)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)

    seed_everything(42)
    second_numpy = np.random.random(4)
    second_torch = torch.rand(4)

    np.testing.assert_array_equal(first_numpy, second_numpy)
    torch.testing.assert_close(first_torch, second_torch, rtol=0, atol=0)
