# Workflow-to-code map

This map separates complete process ownership from the private adapters that will be
implemented only after the public architecture is approved.

| Workflow responsibility | Maintained implementation |
| --- | --- |
| Typed experiment configuration | `ree_prospectivity.config` |
| Feature-cube and site contracts | `ree_prospectivity.preprocessing` |
| Confidential sample contract | `ree_prospectivity.data.contracts` |
| Seeded paper/legacy downscaling | `ree_prospectivity.data.downscaling` |
| Leakage-safe and paper-faithful splits | `ree_prospectivity.splitting` |
| Super-resolution generator | `ree_prospectivity.models.generator` |
| Patch discriminator | `ree_prospectivity.models.discriminator` |
| Alternating GAN optimization | `ree_prospectivity.training.gan` |
| Synthetic descendant generation | `ree_prospectivity.augmentation` |
| PSNR calculation | `ree_prospectivity.metrics` |
| Mineral prospectivity CNN | `ree_prospectivity.models.classifier` |
| CNN optimization | `ree_prospectivity.training.classifier` |
| Aggregate test metrics | `ree_prospectivity.evaluation` |
| Full-area sliding-window prediction | `ree_prospectivity.inference` |
| Stage dependency graphs | `ree_prospectivity.pipeline.plan` |
| Resumable and redacted run manifests | `ree_prospectivity.pipeline.manifest` |
| Backend injection and execution | `ree_prospectivity.pipeline.runner` |
| End-to-end in-memory reference backend | `ree_prospectivity.pipeline.in_memory_backend` |
| CLI planning and execution | `ree_prospectivity.cli` |

## Deliberately external components

The public repository does not implement these local concerns yet:

- reading the confidential Excel workbooks;
- reproducing ArcGIS IDW and granite-distance preprocessing;
- mapping private columns into the reviewed 28-channel order;
- resolving the paper's east/west training-region ambiguity;
- selecting approved output rasters or aggregate figures for publication.

Those operations will be supplied by an ignored local backend implementing
`WorkflowBackend`. The public runner remains unaware of raw paths and values.
