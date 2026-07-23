"""Training loops for GAN augmentation and prospectivity classification."""

from ree_prospectivity.training.classifier import (
    ClassifierTrainingResult,
    train_classifier,
)
from ree_prospectivity.training.gan import GANTrainingResult, train_gan

__all__ = [
    "ClassifierTrainingResult",
    "GANTrainingResult",
    "train_classifier",
    "train_gan",
]
