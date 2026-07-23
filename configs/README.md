# Experiment configurations

This directory will contain versioned, human-readable experiment configurations.

Planned configuration groups:

- `data`: local paths, channel ordering, normalization, and split policy;
- `augmentation`: downscaling procedure, generator architecture, and sample count;
- `classifier`: CNN architecture and optimization settings;
- `evaluation`: metrics, thresholds, spatial holdout, and uncertainty estimates;
- `runtime`: device selection, deterministic settings, and output locations.

Local paths and credentials must be supplied outside tracked configuration files.

`paper_figure.toml` reconstructs the architecture order shown in Figure 6 and retains the
publication's post-augmentation split. `leakage_safe.toml` assigns original groups before
augmentation and reserves a spatially independent test partition.
