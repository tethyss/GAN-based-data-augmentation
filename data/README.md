# Data boundary

No research data belong in Git.

The following material stays local:

- geochemical, geological, and geomorphological grids;
- deposit locations and negative-site coordinates;
- intermediate ArcGIS exports;
- augmented samples;
- checkpoints, predictions, and prospectivity rasters.

The repository guard rejects common tabular, geospatial, array, and model formats.
Authorized local data will later be referenced through an ignored configuration file or
environment variable.

Only documentation and explicitly generated synthetic fixtures may be published.
