# CT windowing grid search for stroke classification on NCCT

Code for the paper

> A. Kaban, M. Karakaya. *Is the Clinical Brain Window Optimal for AI? A
> Systematic Grid Search of CT Windowing for Deep-Learning Stroke
> Classification on Non-Contrast CT.*

The study trains four architectures (ResNet-50, DenseNet-121,
EfficientNet-B0, MobileViTv2-175) on two binary tasks (normal vs. ischemic,
normal vs. hemorrhagic) and compares windowed inputs:

* **Experiment A**: raw HU input (split-wide min-max scaling) vs. the
  clinical brain window (WW = 80, WC = 40); 80 runs.
* **Experiment B**: every window of a 22 x 16 grid (WW 10-220 HU, WC 0-150 HU,
  steps of 10 HU) against the clinical window; 14,080 runs.

Each configuration is trained with five seeds (42, 123, 999, 2026, 7).

The repository contains the code only. No data, result files or model weights
are included.

## Installation

Python 3.10 or newer. Install the PyTorch build for your CUDA version first
(https://pytorch.org), then

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions stated in the paper (PyTorch 2.13.0,
torchvision 0.28.0, timm 1.0.28, SciPy 1.13.1, statsmodels 0.14.2); the
versions of the other packages used in the original experiments were not
recorded. Without `pip install -e .`, set `PYTHONPATH=src` before running the
scripts. Run the tests with `pytest`.

## Repository layout

```
src/ctwin/            library code
  preprocessing.py    DICOM -> HU -> 256x256 int16 curation pipeline
  slice_selection.py  lesion-coverage slice selection (AISD, ISLES 2024)
  windowing.py        windowing and RAW min-max normalization
  data.py, models.py  data loading, the four timm architectures
  training.py         training and evaluation of one run
  configs.py          hyperparameters of every experiment (paper Table 2)
  results_io.py       per-run result files
  analysis.py, stats.py   collecting results, paired tests
scripts/data/         dataset preparation (paper Sec. 3.1)
scripts/train/        Experiment A and the grid search
scripts/analysis/     figures 6-9, Experiment A summary, statistics
scripts/figures/      schematic figures 4 and 5
tests/
```

## 1. Data preparation

All sources are converted to the same representation: 2D axial slices in
Hounsfield Units, resized to 256 x 256 with area interpolation, rounded and
stored as int16 `.npy` files under

```
<data_root>/{train,val,test}/{0,1,2}/<source prefix><id>.npy
```

with class 0 = normal or non-acute findings, 1 = acute ischemic stroke,
2 = hemorrhagic stroke. No windowing is applied at this stage.

| Script | Source | Classes |
|---|---|---|
| `prepare_aisd.py` | AISD | 1 |
| `prepare_isles2024.py` | ISLES 2024 | 1 |
| `prepare_teknofest2021.py` | TEKNOFEST 2021 Stroke Dataset | 0, 1, 2 |
| `prepare_rsna.py` | RSNA 2019 Brain CT Hemorrhage | 0, 2 |
| `prepare_private_clinical_dataset.py` | private clinical dataset (validation and test only) | 0, 1, 2 |

Each script documents its inputs and selection rules (`--help`). The public
datasets must be obtained from their providers. The private clinical dataset
is not available; its script documents how it was split.

`prepare_rsna.py` reads a per-patient metadata JSON built from the RSNA
labels and DICOM headers. The script that built this file was not kept, so
the file is not part of the repository; its expected format is described in
the script's docstring.

After preparation, `scripts/data/count_dataset.py --data-root <root>
--check-paper` compares the slice counts with Table 1 of the paper
(13,636 slices in total):

| Split | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| train | 3,600 | 2,176 | 3,500 |
| val | 534 | 414 | 541 |
| test | 1,067 | 763 | 1,041 |

## 2. Training

```bash
# Experiment A (both conditions, four architectures, five seeds) for one task
python scripts/train/run_experiment_a.py --task ischemic --data-root <root> --results-dir results/experiment_a

# Experiment B: the full grid for one task and architecture
python scripts/train/run_grid.py --task hemorrhagic --arch resnet50 --data-root <root> --results-dir results/grid
```

`--task` is `ischemic` or `hemorrhagic`; `--arch` is `resnet50`,
`densenet121`, `efficientnet_b0` or `mobilevitv2_175`. `--seeds`, `--ww` and
`--wc` restrict a run to part of the grid. Hyperparameters come from
`src/ctwin/configs.py`.

Every run writes one text file with its configuration and the TRAIN / VAL /
TEST metrics of the checkpoint with the lowest validation loss:

```
results/grid/<task>/<arch>/seed_<seed>/<WW>_<WC>_windowing.txt
results/experiment_a/<task>/<arch>/<RAW|BRAIN>_seed_<seed>.txt
```

A run whose result file is complete is skipped, so an interrupted script can
be restarted.

## 3. Analysis and figures

```bash
# Figures 6-9 (heatmaps and marginal plots) and the descriptive numbers of Sec. 4.2
python scripts/analysis/grid_figures.py --ischemic-dir results/grid/ischemic \
    --hemorrhagic-dir results/grid/hemorrhagic --out-dir figures

# Experiment A summary and the Experiment A table
python scripts/analysis/summarize_experiment_a.py --ischemic-dir results/experiment_a/ischemic \
    --hemorrhagic-dir results/experiment_a/hemorrhagic --out-dir analysis/experiment_a

# paired tests, multiple-comparison corrections, transfer and surface agreement
python scripts/analysis/compute_statistics.py \
    --grid-ischemic-dir results/grid/ischemic --grid-hemorrhagic-dir results/grid/hemorrhagic \
    --expa-ischemic-dir results/experiment_a/ischemic \
    --expa-hemorrhagic-dir results/experiment_a/hemorrhagic --out-dir analysis/statistics

# schematic figures 4 and 5, drawn around one example NCCT slice
python scripts/figures/make_fig4_experiment_a.py --dicom <slice.dcm> --out figures/experiment_A.png
python scripts/figures/make_fig5_experiment_b.py --dicom <slice.dcm> --out figures/experiment_B.png
```

The analysis scripts also read the result folders of the original
experiments, whose architecture folders had other names (see
`src/ctwin/analysis.py`), and the original result files, whose labels were in
Turkish.

`compute_statistics.py` was written for this repository: the statistics in the
paper were computed with ad hoc scripts that were not kept. On the original
result files it reproduces the statistical tables of the paper.

## Reproducibility notes

* The scripts reproduce the behaviour of the original experiment notebooks:
  sample order, seeding, augmentation, AMP, gradient clipping, scheduler,
  early stopping, checkpoint selection and metrics. On CPU, short runs of the
  original code and of this code give identical results. GPU training with
  AMP and cuDNN autotuning is not bit-for-bit deterministic, so retraining will
  not reproduce the published numbers exactly.
* Some original training scripts created a new AMP gradient scaler for every
  run, others kept one scaler for the whole process; `configs.py` records
  which was used for each experiment (`fresh_grad_scaler`). With a shared
  scaler, a run depends slightly on the runs before it in the same process,
  and restarting a script does not continue such a process bit for bit.
* The hemorrhagic grid had epoch ceilings of 17 (ResNet-50), 15
  (DenseNet-121), 35 (EfficientNet-B0) and 15 (MobileViTv2-175); 175 of its
  7,040 runs ended at the ceiling.
* Deliberate differences from the original scripts that do not change the
  results: split folders are named `train`, `val` and `test`; files are listed
  in sorted order instead of file-system order; result files are written in
  English and atomically; name collisions, incomplete result files and
  inconsistent inputs raise errors instead of being skipped or overwritten.
* Files of the private dataset carry the prefix `pcd_`, which sorts
  differently from the prefix used originally. This changes the order of the
  samples (and therefore the batches) in retraining, not the data.

## License

MIT, see [LICENSE](LICENSE).
