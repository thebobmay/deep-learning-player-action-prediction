# Deep Learning for Player Action Prediction from Atari Gameplay Frames

Udacity AI Masters Capstone - Deep Learning Systems

**Student:** Robert Mayfield

---

## Project Description

This project builds a PyTorch deep learning experiment that predicts player actions from Atari Montezuma's Revenge gameplay frames. The analysis trains a CNN on real human gameplay recordings and compares a single frame baseline against a four frame stacked experimental model to test whether temporal context improves action prediction. Montezuma's Revenge is a side-scrolling platformer involving jumping, ladder climbing, and hazard avoidance, making it directly relevant to the AI Game Director Studio's focus on platformer behavior modeling.

---

## Dataset

**Name:** Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset
**Source:** Zhang et al. (2020). AAAI Conference on Artificial Intelligence.
**License:** CC BY 4.0
**Sessions:** 27 (20 standard, 7 highscore expert sessions)
**Total frames:** 447,300
**Action classes:** 5 (NOOP, RIGHT, LEFT, UP, DOWN)

See `dataset_access_instructions.md` for full download instructions.

---

## Files Included

```
notebooks/deep_learning.ipynb                       main project notebook
reports/Deep_Learning_Systems_Analysis_Report.pdf   full written report
reports/module_summary.pdf                          identical copy of report
environment.yml                                     conda environment (authoritative GPU spec)
requirements.txt                                    pip package dependencies
dataset_access_instructions.md                      dataset source and download steps
data/sample/                                        reserved for sample frames
outputs/figures/                                    saved chart files
outputs/tables/                                     dl_metrics.csv and other tables
outputs/model_artifacts/                            saved PyTorch model weights
```

---

## How to Run

1. Create the conda environment from `environment.yml`:
   ```
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```
   conda activate deep-learning-capstone
   ```
3. Download the dataset from Zenodo (see `dataset_access_instructions.md`) and place `montezuma_revenge.zip` into `data/raw/`. The notebook reads frames directly from the zip archive.
4. Open the notebook in Jupyter:
   ```
   jupyter notebook notebooks/deep_learning.ipynb
   ```
5. Run all cells from top to bottom.

**Prerequisites:** Anaconda or Miniconda with Python 3.11, and conda on your PATH.

**Note:** GPU execution is recommended. The notebook detects CUDA automatically and falls back to CPU if unavailable. Training all three models end to end takes approximately 75 to 90 minutes. The two primary comparison models take approximately 55 to 60 minutes. The `environment.yml` installs `torch==2.6.0+cu124` (CUDA 12.4 build). CPU-only users should install PyTorch separately after activating the environment: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`.

---

## Bias and Responsible Data Handling

All gameplay recordings come from a single player across 27 sessions. The model learns to predict that player's behavioral patterns and should not be treated as a universal model of human Montezuma's Revenge play. The dataset was collected under informed consent as part of an academic study. Any system applying this model to infer player intent from live gameplay should disclose that inference is occurring and provide mechanisms for players to opt out.

---

## Future Integration Reflection

### How this classifier could support the AI Game Director Studio

A model that predicts player actions from game frames can serve as a behavioral signal layer in the AI Game Director: detecting when a player is stuck (repeated NOOP frames), recognizing movement patterns that indicate difficulty navigating a level section, or flagging sequences where directional actions spike and then collapse. These signals can inform level difficulty adjustment or trigger director interventions.

### How this dataset and model would need to evolve for deeper integration

A production system would require data from multiple players, multiple skill levels, and multiple game titles. The current model is specific to one player's Montezuma's Revenge patterns. Integrating with a live game would also require a real-time frame inference pipeline rather than the batch training setup used here.

### How agentic automation could assist this workflow

An agentic pipeline could monitor live session frames, run rolling inference using the trained model, flag behavioral state changes (prolonged idle, direction reversals, apparent confusion), and surface those signals to the game director agent without requiring manual analysis per session.

---

## Requirements

See `environment.yml` for the full conda environment including CUDA dependencies.
See `requirements.txt` for pip package dependencies.

Key libraries: Python 3.x, PyTorch 2.6.0 (CUDA 12.4), NumPy, Pillow, scikit-learn, matplotlib, seaborn, tqdm, Jupyter

**Note on requirements.txt format:** This file was generated with `pip freeze` from a conda environment. Many entries use the `package @ file:///...` format, which is standard output for conda-managed packages when captured via pip. The critical packages (torch, numpy, scikit-learn, matplotlib, etc.) include version numbers. To reproduce the environment, use `conda env create -f environment.yml` rather than `pip install -r requirements.txt`.