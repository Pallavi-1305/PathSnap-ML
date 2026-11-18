# PathSnap

PathSnap matches noisy vehicle GPS observations to directed road arcs. This repository contains candidate generation, five candidate classifiers, training-only radius/model selection, Optuna tuning, Viterbi decoding, route evaluation, EDA, Folium maps, and a frozen-model API with a small browser UI.

## Current evaluation snapshot

- Dataset available locally: 80 training trips and 10 student-test trips. No evaluator trip folder was present or accessed.
- Candidate setting: **150 m radius, cap 30**, selected by five-fold training-trip GroupKFold macro F1 among the required 150/175/200 m experiments.
- Primary model: **Logistic Regression**, selected by mean training GroupKFold macro F1 of **0.8013**.
- Optuna: 12 five-fold training-only trials; best `C=56.6985`, macro F1 **0.80135**. Tuned and untuned student macro F1 were both **0.8765**.
- HMM/Viterbi: evaluated after classifier predictions. Mean student route F1 was **0.5490** classifier-only and **0.5440** with HMM; the HMM improved precision and transition feasibility but reduced recall and sequence similarity.
- Full results and metric definitions: [report/final_report.md](report/final_report.md).

## Data and setup

The raw files are not stored in Git. Each trip directory should contain `.track`, `.nodes`, `.arcs`, and `.route` files. `config.yaml` points to the expected dataset root using a relative path; change `data.raw_root` if your local layout differs. The 80/10 partition is read from `data/splits.csv`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce the pipeline

Run these commands from the repository root, in order:

```powershell
python scripts/1_eda.py
python scripts/1b_baseline_radius.py
python run_preprocessing.py
python scripts/2_compare_models.py
python scripts/3_tune.py
python scripts/4_export_final.py
python scripts/5_decode_hmm.py
python scripts/6_training_missing_candidate_map.py
python scripts/7_api_dry_run.py
python -m pytest -q
python -m pytest tests/test_api.py -m slow -q -s
```

`scripts/1_eda.py` uses training trips only to compare candidate settings. `scripts/1b_baseline_radius.py` adds the old 50 m/cap 10 setting as a reference row; it does not participate in the 150–200 m radius selection. The model comparison uses the held-out student set for reporting only. Optuna loads only training trips. The tuned-model export evaluates the student set after tuning is complete. HMM settings stay fixed in `config.yaml`.

Important outputs:

- Radius, recall and miss results: `results/candidate_radius_comparison.csv`, `results/candidate_radius_fold_results.csv`, and `results/candidate_radius_selection.json`.
- Five-model results and class metrics: `results/model_comparison.csv`, `results/model_classification_report.csv`, `results/model_details.csv`.
- Per-model student confusion matrices: `results/confusion_matrices/` (CSV and PNG).
- Optuna trial history and best settings: `results/tuning_trials.csv`, `results/best_params.json`.
- Tuned/untuned comparison and frozen artifact: `results/tuned_vs_untuned.csv`, `artifacts/final_model.joblib`, `artifacts/model_metadata.json`.
- Label-free candidate probabilities: `results/candidate_probabilities.csv`; full observation accounting, including points with no candidates, is in `data/processed/observation_accounting.csv`.
- HMM sequence output and classifier/HMM route metrics: `results/hmm_predictions.csv`, `results/decoding_comparison.csv`.
- EDA tables/plots: `results/eda/`; interactive maps: `results/figures/maps/`.

The probability export is keyed by `trip_id`, `gps_point_id`, and `candidate_arc_id`. Reference labels are stored separately. Missing observations are retained in the accounting file and are never assigned an invented arc.

## API and UI

Laptop 1 (API; console output includes request and completion logs):

```powershell
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Laptop 2 (static UI):

```powershell
python -m http.server 8080 --directory app/ui
```

Open `http://localhost:8080` on Laptop 2 and enter Laptop 1's reachable address, such as `http://192.168.1.20:8000`. The API exposes `GET /api/trips` and `POST /api/decode/{trip_id}`. It allows only IDs in the `student_test` split. It generates predictions before opening the corresponding reference route, returns metrics and predicted geometry, and never returns reference-route geometry. All 10 student-test API requests passed the automated dry run.

## Evaluation and labels

Candidate labels are strict: for each GPS observation, at most the closest candidate arc that appears in that trip's reference route is positive; all remaining candidates are negative. If no correct arc survives the radius and cap, no candidate is labeled positive. This avoids turning every route arc near a GPS point into a positive, but it can miss legitimate parallel or ambiguous route arcs.

The per-point features `dist_rank` and `dist_gap_m` describe a candidate's distance rank and separation from neighboring candidates. They help distinguish similarly close road options without using the reference route.

All classifier and HMM outputs preserve the full GPS observation sequence. A point with no candidates remains missing and contributes a false negative to the confusion accounting, recall/F1 and observation-level accuracy. Standard precision is `TP/(TP+FP)`; an absent prediction creates a false negative rather than a false positive, so precision is mathematically unchanged by that row. Candidate-miss rate and the full-point denominator make coverage visible. HMM route precision/recall/F1 and sequence penalties use the same explicit miss count.

## Git and AI disclosure

Work is on `member3-hmm-viterbi`. No commit or push was made. AI assistance (Codex) was used for implementation, debugging, and report drafting; the team remains responsible for reviewing and presenting the work.
