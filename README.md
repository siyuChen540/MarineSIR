# MarineSIR: Marine Satellite Image Reconstruction Workbench

MarineSIR is a Python application for cloud-gap reconstruction in marine satellite image sequences. The current application wraps the refactored FTC/ConvLSTM cloud-removal backend in a minimal scientific GUI. The GUI is intentionally separated from the training and prediction process: the interface remains a lightweight PyQt5 renderer, while model training and evaluation run in an external Python process.

## Current Architecture

1. **Data preparation**: Reads ordered image sequences from folders. `.npy` is fully supported; `.nc/.nc4` and `.tif/.tiff` are supported for basic 2D-per-file sequences. The data inspector reports file count, sliding-window count, shape, missing ratio, and value range.
2. **FTC-LSTM reconstruction backend**: The real backend lives in `core/cloudRemoval`. It provides configurable Fourier ConvLSTM modes (`fft_add`, `fft_concat`, `none`), hybrid pixel/SSIM loss, mask-aware metrics, checkpointing, CSV metrics, TensorBoard output, and saved visual samples.
3. **GUI workflow**: `main.py` launches a PyQt5 workbench for dataset inspection, training, metric visualization, sample visualization, parameter review, structured logs, and prediction export. Training and prediction are launched via `QProcess`, so the algorithm process is isolated from the rendering process.
4. **Classical baselines**: `core/classical_cli.py` currently provides a lightweight DINEOF/SVD gap-filling baseline for quick comparison. DINCAE is exposed as a planned integration point but is not bundled yet.

## Important Runtime Note

The backend Python defaults to:

```text
anaconda3\envs\torch_env\python.exe
```

That environment has PyTorch/CUDA and the scientific backend packages, but it may not include PyQt5. You can run the GUI from any environment with PyQt5 and point the backend Python field to the Torch environment above.

## Install Dependencies

GUI environment:

```bash
pip install -r requirements-gui.txt
```

Backend environment:

```bash
pip install -r requirements-backend.txt
```

For CUDA-enabled PyTorch, prefer the install command recommended by the PyTorch project for your GPU/CUDA stack instead of blindly installing the CPU wheel.

## Run the GUI

```bash
python main.py
```

Default GUI fields point to the included `exampleDS` folder and `core/cloudRemoval/configs/fast_debug.yaml` for a quick smoke test. Use the **Inspect Data** button first, then **Run / Train**. The Runtime section lets you choose `auto`, `cpu`, `cuda`, `cuda:0`, or `cuda:1`. During FTC-LSTM training, the Training tab plots loss, reconstruction metrics, samples/second, epoch duration, and CUDA memory when available. After training, **Export Predictions** writes NetCDF or NPZ products from the selected checkpoint.

The Log tab uses a structured log panel with level filtering, clear, and auto-scroll controls. The Parameters tab summarizes current runtime, data, algorithm, and training settings before each run.

## Command-Line Training

The root `train.py` is a compatibility wrapper around the real backend:

```bash
python train.py --config core/cloudRemoval/configs/default.yaml
```

Quick smoke test with the included sample data:

```bash
python core/cloudRemoval/tools/inspect_batch.py --config core/cloudRemoval/configs/fast_debug.yaml --set data.root_dir=exampleDS --set data.mask_dir=null
python train.py --config core/cloudRemoval/configs/fast_debug.yaml --set data.root_dir=exampleDS --set data.mask_dir=null --set training.epochs=1
```


## Classical Baseline

Run DINEOF directly from the command line:

```bash
python core/classical_cli.py dineof \
  --data-root exampleDS \
  --suffix .npy \
  --output-dir record/dineof \
  --rank 8 \
  --max-iter 50 \
  --output-format netcdf
```

The GUI can also run DINEOF by selecting `DINEOF` in the Algorithm field and pressing **Run / Train**.

## Prediction Export

```bash
python core/cloudRemoval/evaluate.py \
  --config core/cloudRemoval/configs/fast_debug.yaml \
  --checkpoint record/<run>/checkpoints/best.pt \
  --split all \
  --output-format netcdf
```

NetCDF export writes variables `reconstruction`, `input`, `target`, and `observed_mask` with dimensions `time`, `y`, and `x`. For Windows paths containing non-ASCII characters, MarineSIR writes NetCDF through xarray's scipy backend for better path compatibility.

## Known Limitations

- NetCDF and TIFF support currently assumes one 2D image per file. Multi-time-step NetCDF files and georeferenced GeoTIFF metadata preservation should be added before public release.
- A real pretrained FTC-LSTM checkpoint is not bundled yet. The GUI can train a checkpoint or export predictions from a user-selected checkpoint.
- DINCAE is listed as a planned integration point, but a true DINCAE backend is not yet included.
- The included `exampleDS` folder is a minimal NPY smoke-test dataset, not a full benchmark dataset.
