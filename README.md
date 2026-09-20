# dsa-ml-lab

Course: Machine Learning Algorithms Lab

The goal of the course project is to study and upgrade four core machine learning algorithms implemented in this repo:
- Naive Bayes
- Linear Regression
- K-means
- AdaBoost

Each algorithm exists in two versions:
- **Original** — the baseline code from the lab sessions, written as a top-to-bottom script. Kept unchanged in `original_code/` for reference.
- **Improved** — the same algorithm reorganized into `learn()` / `predict()` (or `transform()`) functions, with the assigned extensions added on top. These are the scripts in the repo root.

## Contents
1. [Project Structure](#project-structure)
2. [Quick usage example](#quick-usage-example)
3. [Original vs improved](#original-vs-improved)
4. [Not yet implemented](#not-yet-implemented)
5. [Experiments & Reproducibility](#experiments--reproducibility)

## Project Structure
- **Improved implementations** (repo root):
  - [naive_bayes.py](naive_bayes.py): Naive Bayes.
  - [linear_regression.py](linear_regression.py): Linear Regression.
  - [kmeans.py](kmeans.py): K-means.
  - [adaboost.py](adaboost.py): AdaBoost.
  - [requirements.txt](requirements.txt): Python dependencies.
- **Original implementations**:
  - See the `original_code/` folder for the unmodified lab versions of all four algorithms.
- **Data**:
  - See the `data/` folder for the CSV datasets. The improved scripts run on the datasets required by the assignments (`boston.csv`, `drug.csv`, `drugY.csv`); the original scripts use the smaller lab datasets (`house.csv`, `cold.csv`, `life.csv`).

## Quick usage example
- **Setup**:
  - Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

  - Run a script directly from the repo root (each script contains examples or simple runners, and dataset paths are relative to the root):
```bash
python naive_bayes.py
python linear_regression.py
python kmeans.py
python adaboost.py
```

  - Adjust or extend scripts to run experiments with different datasets or parameters.

## Original vs improved

Every algorithm keeps the original logic at its core; the changes below are what was added on top.

- **Naive Bayes** — `original_code/naive_bayes.py` → [naive_bayes.py](naive_bayes.py)
  - Original: categorical attributes only, probabilities multiplied directly, no smoothing. Runs on `cold.csv`.
  - Added: log-space probabilities to avoid underflow; additive smoothing with user-controlled strength (`smoothing`); numeric attributes supported by modelling the conditional distribution as a normal distribution. Runs on `drug.csv` with a train/test split.

- **Linear Regression** — `original_code/linear_regression.py` → [linear_regression.py](linear_regression.py)
  - Original: plain gradient descent with a fixed learning rate. Runs on `house.csv`.
  - Added: L2 regularization with a tunable `lam` (the intercept `w0` is excluded from the penalty via a mask); online/stochastic gradient descent on a single random instance (`stochastic`); automatic learning-rate selection (`auto_alpha`) — step size grows while the cost falls, and the step is undone and halved when it rises. Runs on `boston.csv` with a train/test split.

- **K-means** — `original_code/kmeans.py` → [kmeans.py](kmeans.py)
  - Original: random centroid initialization, unweighted distances, single run. Runs on `life.csv`.
  - Added: per-attribute weights so similarity respects attribute importance (`w`); automatic restart of the whole procedure `N` times, reporting the best model by cluster quality; farthest-point initialization instead of random sampling; diagnostics in `transform()` that warn when a cluster is poorly represented by its centroid (`max_radius`) or when two clusters are too similar (`min_separation`). Runs on `boston.csv`.

- **AdaBoost** — `original_code/adaboost.py` → [adaboost.py](adaboost.py)
  - Original: hardcoded `GaussianNB` weak learner, fixed ensemble size, prediction only. Runs on `drugY.csv`.
  - Added: any base algorithm can be passed in (`base_alg`); a `learning_rate` parameter controls how strongly each model adapts to the previous model's errors; a list of base algorithms can be supplied and one is picked at random each iteration (seeded by `rng`); predictions come with a confidence score alongside the ensemble vote. Still runs on `drugY.csv`, now with a train/test split.

## Not yet implemented
- K-means bonus: automatic selection of the optimal number of clusters using the silhouette index.

## Experiments & Reproducibility
- Train/test splits use `random_state=42`, and AdaBoost's random choice of base algorithm is seeded with `rng=42`, so those runs repeat exactly.
- Two parts are intentionally left unseeded and will vary slightly between runs: the initial weights in Linear Regression (`np.random.random`) and the centroid initialization in K-means. K-means compensates by restarting `N` times and keeping the best model; pass a larger `N` for more stable clusters.
- All dataset paths are relative to the repo root, so run the scripts from there.
