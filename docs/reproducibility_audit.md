# Reproducibility audit

## Source publication

Li, T., Zuo, R., Zhao, X., and Zhao, K. (2022), *Ore Geology Reviews* 142,
104693. DOI: 10.1016/j.oregeorev.2022.104693.

The local PDF is reference material only and is excluded from the public repository.

## Published workflow

1. Compile geological, geochemical, and geomorphological rasters at 1 km resolution.
2. Extract `32 x 32 x 28` windows around 24 deposits and 24 negative sites.
3. Randomly downscale each window to `8 x 8` by averaging two sampled pixels per
   `4 x 4` block.
4. Train a super-resolution-inspired GAN and augment each original sample 200 times.
5. Train a CNN on 9,600 augmented positive and negative samples.
6. Apply a `32 x 32` sliding window to produce cell-level mineralization probabilities.

The paper reports an Adam learning rate of `1e-5`, 1,200 CNN epochs, 99.7% training
accuracy, 98.9% validation accuracy, and a final prospective area covering 2.36% of the
study area.

## Paper-to-code comparison

| Concern | Paper | Legacy implementation | Reproduction impact |
| --- | --- | --- | --- |
| CNN input | `32 x 32 x 28` | `32 x 32 x (25 + channels)`; invoked with 27 channels | Architecture mismatch |
| CNN filters | 64, 128, 256 with `3 x 3` kernels | 32, 32, 64; final kernel is `2 x 2` | Architecture mismatch |
| Dense layers | 512 then 1024 before output | One 256-unit layer before output | Architecture mismatch |
| CNN epochs | 1,200 | 500 | Training mismatch |
| Augmentation count | 200 per original sample | Classifier call requests 10 | Dataset mismatch |
| GAN channels | Published output described as 28 channels | Loader reshapes each sample to 3 channels | Undocumented reduction |
| Data split | 80/20 after augmentation | Keras `validation_split=0.2` | Potential group leakage |
| Optimizer/loss | Only partly specified | Two GAN files disagree | Ambiguous experiment |
| Runtime | Python 3.8, TensorFlow 2.1.0 | No environment lock | Environment not reproducible |

## Missing or conflicting implementation details

- `CNNforSouthgan.py` imports a missing `utils.py`.
- GAN scripts import `data_loader.py`, but the provided file was named
  `data_loader(1).py`.
- Source and prediction paths are hard-coded to one workstation.
- Random seeds and deterministic runtime settings are absent.
- Scaling, missing-value handling, VGG preprocessing, and inverse transformations are
  not defined end to end.
- The two GAN scripts disagree on discriminator loss, adversarial weight, batch
  normalization momentum, checkpoint epoch, and training behavior.
- The paper describes the eastern half as the training area in one section, while a later
  results passage describes eastern deposits as unseen test data. The intended spatial
  split requires confirmation from original records.
- The origin and role of the apparent three-channel PCA representation are not documented.
- The negative-site random seed and the exact ArcGIS preprocessing operations are not
  captured.

## Validation risks

Splitting after augmentation can place closely related synthetic descendants of the same
original sample in both training and validation sets. This can inflate validation accuracy.
The maintained project will therefore preserve a paper-faithful mode for historical
comparison and add a grouped, spatially independent evaluation mode for credible model
assessment.

Accuracy alone is insufficient for a rare-event spatial prediction problem. Later
milestones should add ROC AUC, precision-recall AUC, calibration, confusion matrices,
spatial hit rate, prospective-area coverage, and uncertainty summaries.

## Reconstruction milestones

1. Establish a public/private repository boundary.
2. Approve the framework, runtime, and paper-faithful architecture specification.
3. Implement deterministic data contracts and a synthetic fixture generator.
4. Implement and unit-test random downscaling, augmentation, and classifier models.
5. Reconstruct the authorized local preprocessing pipeline.
6. Run paper-faithful and leakage-safe experiments with machine-readable tracking.
7. Publish reviewed aggregate results, documentation, and a model card.
