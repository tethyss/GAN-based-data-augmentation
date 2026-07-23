# ADR 0001: Use PyTorch for the maintained implementation

## Status

Accepted for the reconstruction branch.

## Context

The publication reports Python 3.8 and TensorFlow 2.1.0, while the supplied scripts use
historical standalone-Keras imports and cannot run without missing modules. Recreating that
environment would preserve software archaeology but would not provide a maintainable,
industry-facing project.

The model equations and tensor operations do not depend on TensorFlow. The original
scripts remain available in `legacy/` for behavioral comparison.

## Decision

Use stable PyTorch 2.x for maintained data, model, training, and evaluation code.

- Models expose logits rather than embedding loss-specific sigmoid operations.
- Randomness is passed explicitly where possible and seeded centrally otherwise.
- Tensor shapes are validated at public boundaries.
- Paper ambiguities remain configuration choices instead of being silently resolved.
- CPU is the reference test platform; CUDA is an optional execution device.

Exact framework, Python, accelerator, and lock-file versions are intentionally deferred
until the workflow structure and experiment contracts have been reviewed.

## Consequences

Numerical parity with the historical Keras scripts is not expected because initialization,
kernel layout, and backend algorithms differ. Reproduction claims will therefore concern
the documented method and reported evaluation protocol, not bit-for-bit parity.

The three-channel VGG perceptual loss in the legacy code is not included in the first model
slice because the publication describes 28-channel output and does not document a valid
mapping into ImageNet RGB feature space. That decision requires a separate experiment.
