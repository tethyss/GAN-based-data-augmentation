# Data contract

## Paper-level sample definition

Each high-resolution sample is a `32 x 32 x 28` tensor covering a `32 km x 32 km`
window at 1 km cell resolution:

- 26 geochemical channels;
- 1 geological channel representing proximity to granite;
- 1 geomorphological channel.

The augmentation model receives a random downscaled `8 x 8 x C` tensor. Each low-resolution
cell is the average of two randomly selected values from the corresponding `4 x 4` block
of the high-resolution sample.

## Proposed maintained interface

The maintained loader will return:

| Field | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `high_resolution` | `float32` | `(N, 32, 32, C)` | Authorized source windows |
| `low_resolution` | `float32` | `(N, 8, 8, C)` | Seeded random downscales |
| `label` | integer | `(N,)` | Deposit/non-deposit class |
| `sample_id` | string | `(N,)` | Stable non-spatial identifier |
| `group_id` | string | `(N,)` | Original site used for grouped splits |
| `split` | category | `(N,)` | Train, validation, or spatial test |

Channel names, units, missing-value rules, transformations, and normalization statistics
will be represented in a separate metadata object rather than inferred from array position.

## Local inventory

The local workbook has one worksheet named `dataclip.shp` with 27,557 rows and 29 columns.
This observation records dimensions only; values and coordinates have not been copied into
the repository. The relationship between its 29 columns and the paper's 28 model channels
must be verified before implementing the loader.

The legacy deposit workbook has not yet been converted or exposed.

## Split policy

Two policies will be supported and clearly labeled:

1. **Paper-faithful**: reconstruct the published training and validation procedure as
   closely as the available records permit.
2. **Leakage-safe**: assign original sites to splits before augmentation and preserve
   spatially independent test deposits.

Synthetic descendants of the same original site must share a `group_id`.

## Confidentiality requirements

- Raw paths are supplied only through ignored local configuration.
- Logs must not include coordinates or raw records.
- Tests and CI use synthetic inputs.
- Metrics intended for publication must be aggregated and reviewed.
- No derived artifact is assumed safe merely because direct identifiers were removed.
