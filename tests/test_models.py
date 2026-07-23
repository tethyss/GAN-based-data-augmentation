import torch
from torch.nn import functional as functional

from ree_prospectivity.models import (
    PatchDiscriminator,
    ProspectivityCNN,
    SRGenerator,
)


def test_generator_and_discriminator_shapes() -> None:
    generator = SRGenerator(channels=4, features=16, residual_blocks=2)
    discriminator = PatchDiscriminator(channels=4, base_features=8)
    low_resolution = torch.randn(2, 4, 8, 8)

    generated = generator(low_resolution)
    logits = discriminator(generated)

    assert generated.shape == (2, 4, 32, 32)
    assert generated.min() >= -1
    assert generated.max() <= 1
    assert logits.shape == (2, 1, 2, 2)


def test_gan_components_support_backpropagation() -> None:
    generator = SRGenerator(channels=2, features=8, residual_blocks=1)
    discriminator = PatchDiscriminator(channels=2, base_features=4)
    generated = generator(torch.randn(1, 2, 8, 8))

    loss = functional.binary_cross_entropy_with_logits(
        discriminator(generated),
        torch.ones(1, 1, 2, 2),
    )
    loss.backward()

    assert generator.input_layer[0].weight.grad is not None
    assert discriminator.classifier[-1].weight.grad is not None


def test_prospectivity_classifier_shape_and_gradient() -> None:
    classifier = ProspectivityCNN(
        channels=4,
        filters=(8, 16, 32),
        hidden_dims=(32, 16),
        dropout=0,
    )
    inputs = torch.randn(3, 4, 32, 32)
    labels = torch.tensor([0.0, 1.0, 0.0])

    logits = classifier(inputs)
    loss = functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    assert logits.shape == (3,)
    assert classifier.features[0].weight.grad is not None
