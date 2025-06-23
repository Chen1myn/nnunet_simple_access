# nnUNet Batch Runner with Auto Notebook Visualization

> ⚠️ This project requires **nnUNet v2**.  
> 📦 Official repository: https://github.com/MIC-DKFZ/nnUNet

This repository provides a customized `run_all.py` script to fully automate the segmentation pipeline for a single-organ nnUNet task—from **MSD conversion**, **planning**, **training**, and **inference**, all the way to **automatic notebook-based visualization** in the browser.

---

## 📦 Key Features

- [x] Centralized parameter and environment setup
- [x] Automatically skips already completed steps
- [x] Integrates nnUNet CLI + internal API for streamlined calls
- [x] Automatically launches Jupyter Notebook to view predictions
- [x] Configurable organ name, dataset ID, input file, and all paths

---

## 📁 Recommended Directory Structure

```
project_root/
├── run_all.py                  # Main script
├── show.ipynb                  # Jupyter notebook for visualization
├── config.json                 # Auto-generated config for notebook use
└── DATASET/
    ├── nnUNet_raw/
    ├── nnUNet_preprocessed/
    └── nnUNet_results/
```

---

## ⚙️ Parameter Explanation (Edit at Top of `main()`)

```python
DATASET_ID = "99"
```
> ✳️ Dataset ID for nnUNet (e.g., will become `Dataset099_Heart`)

---

```python
PLAN_CONFIG = "3d_fullres"
```
> ✳️ nnUNet training plan  
Options: `2d`, `3d_lowres`, `3d_fullres`, `3d_cascade_fullres`

---

```python
FOLD = "0"
```
> ✳️ Fold index for cross-validation

---

```python
NUM_PROC = "8"
```
> ✳️ Number of CPU threads used in preprocessing

---

```python
EPOCHS = "5"
```
> ✳️ Number of training epochs (currently only stored as environment variable)

---

```python
NAME = "Heart"
```
> ✳️ Organ name, used to construct dataset path: `Dataset099_Heart`

---

```python
EXAMPLE_RAW = "la_003_0000.nii.gz"
```
> ✳️ A sample raw CT file used for notebook demo

---

```python
# Edit these to your local nnUNet paths:
'MSD_raw':             r"...",
'nnUNet_raw':          r"...",
'nnUNet_preprocessed': r"...",
'nnUNet_results':      r"..."
```
> ✅ These must point to your actual local directories

---

## 🚀 How to Use

1. Edit paths and parameters at the top of `run_all.py`
2. Place your `show.ipynb` notebook in the same directory
3. Run:

```bash
python run_all.py
```

The script will:
1. Check if preprocessing & model checkpoint already exist
2. If not, it will:
   - Convert MSD dataset
   - Plan and preprocess
   - Train nnUNet model
   - Run inference on `imagesTr` → `imagesTs_predlowres`
3. Generate `config.json` for notebook
4. Auto-execute `show.ipynb` and launch it in browser

---

## 📤 Output Structure Example

```
nnUNet_raw/
└── Dataset099_Heart/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── imagesTs_predlowres/       ← inference output
```

```
nnUNet_results/
└── Dataset099_Heart/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        └── fold_0/
            └── checkpoint_best.pth
```

---

## 🧠 Notebook Integration

- `config.json` is automatically generated for the notebook:

```json
{
  "nnUNet_raw": "your_path_here",
  "datasetid": "99",
  "NAME": "Heart"
}
```

- `show.ipynb` is automatically executed using `nbconvert`
- A browser tab will open with the rendered `show_executed.ipynb`

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: nnunetv2` | Ensure nnUNet v2 is properly installed |
| Notebook not opening | Try running `jupyter notebook` manually |
| Inference results missing | Make sure `checkpoint_best.pth` was trained successfully |

---

## 📩 Contact

If you have questions or suggestions, feel free to reach out:  
📧 **122090859@link.cuhk.edu.cn**

---
