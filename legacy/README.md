# Legacy research implementation

These files are preserved without functional refactoring so that later changes remain
traceable:

- `CNNforSouthgan.py`: prospectivity classifier and full-grid inference script;
- `data_loader(1).py`: CSV sample loader and random downscaling routine;
- `srgan(1).py`: GAN training and visualization variant;
- `srgan_1.py`: GAN inference/export variant;
- `read_data.ipynb`: exploratory preprocessing notebook.

The legacy scripts are not expected to run:

- imports and filenames do not match;
- required modules and directories are missing;
- local paths are hard-coded;
- the historical TensorFlow/Keras environment is not locked;
- behavior differs from the published method.

They must not be used as a production package or treated as an exact executable record of
the paper.
