# Amazon_ml_25

A small machine learning project for multimodal (text+image) feature extraction and modelling. The repository contains data samples, notebooks, and Python source for building baseline and advanced models.

## Repository layout

- `dataset/` — CSV datasets used for training and testing (some files are large).
- `notebooks/` — Jupyter notebooks for exploration and feature engineering.
- `src/` — Python source code and utilities (models, feature extraction, helpers).
- `images/`, `models/`, `outputs/` — assets, saved models and generated outputs.

## Quick start

Prerequisites
- Python 3.8+ (recommended)
- pip

Create a virtual environment and install typical dependencies (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(There is no `requirements.txt` in the repo by default; consider creating one capturing packages you use such as numpy, pandas, scikit-learn, catboost, pillow, torchvision, etc.)

Run a script (example)

```powershell
# Run the baseline model training script
python src/baseline_model.py

# Extract image features (if images are present)
python src/add_image_features.py
```

## Notebooks
Open the notebooks in `notebooks/` with Jupyter or VS Code Notebook support for interactive exploration.

## Data and large files
This repository currently contains two relatively large files in `dataset/` (~70 MB each): `train.csv` and `test.csv`.

GitHub recommends using Git LFS for files larger than ~50 MB. To enable LFS (recommended for this project):

```powershell
# Install Git LFS (follow https://git-lfs.github.com), then:
git lfs install
git lfs track "dataset/*.csv"
git add .gitattributes
git commit -m "Track dataset CSVs with Git LFS"
# Re-add or migrate large files if needed
```

Alternatively, consider storing raw datasets in cloud storage (S3, GCS) and keeping only small samples in the repo.

## Contribution
- Add issues for bugs or feature requests.
- Use feature branches and open pull requests against `main`.

## License
Add a LICENSE file as appropriate for your project.

## Contact
If you want me to add `requirements.txt`, enable Git LFS and migrate existing large files, or add a CI workflow, say which you'd like done next.