# GAN-Augmented Mineral Prospectivity Mapping

Reproducible research implementation of a GAN-based augmentation workflow and a
convolutional neural network for regolith-hosted rare-earth-element prospectivity
mapping.

This repository accompanies the following peer-reviewed study:

> Li, T., Zuo, R., Zhao, X., & Zhao, K. (2022). Mapping prospectivity for
> regolith-hosted REE deposits via convolutional neural network with generative
> adversarial network augmented data. *Ore Geology Reviews, 142*, 104693.
> [https://doi.org/10.1016/j.oregeorev.2022.104693](https://doi.org/10.1016/j.oregeorev.2022.104693)

## Demonstration

![GAN-based geoscience data augmentation](docs/assets/demo.gif)

The animation compares representative multichannel geoscience samples during
super-resolution generation:

- **Generated** (left): the generator output reconstructed from the low-resolution input;
- **Original** (center): the corresponding high-resolution reference window;
- **Low** (right): the randomly downscaled `8 x 8` input constructed from the original
  `32 x 32` sample.

Each row represents a different sample. The epoch label follows the saved training frame,
making it possible to inspect how generated spatial patterns evolve during GAN training.
The objective is to recover coherent local structures while keeping generated values within
the learned distribution of the original geological, geochemical, and geomorphological
windows.

This animation is a qualitative diagnostic rather than proof of geological validity.
Quantitative assessment is handled separately with PSNR and downstream prospectivity
classification metrics.

## Project status

The repository is being reconstructed from the original research scripts. The current
milestone establishes the public/private data boundary and records the reproducibility
gaps that must be resolved before model results can be claimed.

- Original scripts are preserved in `legacy/` for traceability.
- Confidential research data and the local paper PDF are excluded from Git.
- CI and a repository guard reject common geospatial data, tabular data, model weights,
  and oversized files.
- Deterministic downscaling, synthetic fixtures, and PyTorch model components are under
  active reconstruction.

## Reproducibility principles

1. No confidential observations, deposit coordinates, raster layers, or trained weights
   are committed.
2. Public tests use generated synthetic fixtures only.
3. Every experiment will be configuration-driven and seed-controlled.
4. Paper-faithful results and leakage-safe validation results will be reported separately.
5. Claims will be linked to machine-readable metrics and environment metadata.

See [the reproducibility audit](docs/reproducibility_audit.md) and
[the data contract](docs/data_contract.md) for the current specification. The complete
stage graph and confidentiality boundary are documented in
[the workflow guide](docs/workflow.md).

## Repository layout

```text
configs/                 Experiment configuration documentation
data/                    Documentation only; private data are ignored
docs/                    Reproducibility and design records
legacy/                  Original, non-runnable research scripts
scripts/                 Repository safety and automation tools
src/ree_prospectivity/   Maintained Python package
tests/                   Public tests using synthetic inputs
```

The maintained package now separates:

- data contracts, preprocessing, splitting, and seeded random downscaling;
- generator, discriminator, GAN training, and augmentation-quality assessment;
- prospectivity CNN training and aggregate classification evaluation;
- memory-bounded full-area sliding-window inference;
- policy-aware workflow planning, resumable manifests, and redacted export metadata.

## Implementation status

The end-to-end workflow and model components are currently represented in code, while the
runtime environment and dependency lock are intentionally deferred. Repository automation
currently checks only the public/private boundary and Python syntax.

The maintained design uses PyTorch-style model components. The historical TensorFlow/Keras
scripts remain available only as traceability artifacts in `legacy/`.

## Data availability

The research data are confidential and are not distributed with this repository.
Future releases will provide a schema-compatible synthetic dataset and instructions for
mapping authorized local data into the same interface.

## Citation

If this repository supports your work, please cite the associated paper:

```bibtex
@article{li2022gan_ree_prospectivity,
  title   = {Mapping prospectivity for regolith-hosted REE deposits via convolutional
             neural network with generative adversarial network augmented data},
  author  = {Li, Tong and Zuo, Renguang and Zhao, Xinfu and Zhao, Kuidong},
  journal = {Ore Geology Reviews},
  volume  = {142},
  pages   = {104693},
  year    = {2022},
  doi     = {10.1016/j.oregeorev.2022.104693}
}
```

## License

This project is released under the [MIT License](LICENSE).
