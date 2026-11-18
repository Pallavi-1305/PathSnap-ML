# PathSnap: final implementation and evaluation report

**Branch:** `member3-hmm-viterbi`  
**Data available for this run:** 80 training trips and 10 student-test trips. The private evaluator dataset was not present and was not accessed.  
**Evaluation date:** 2026-11-18

## 1. Problem and pipeline

PathSnap assigns each noisy GPS observation to a directed road arc. For each point, candidate generation finds nearby arcs, a classifier scores each candidate, and a directed-road HMM with Viterbi decoding uses those probabilities to produce a continuous route. Reference routes are used for training labels and evaluation only.

The pipeline now preserves the original observation sequence and accounts for every GPS point, including observations for which candidate generation returns no arc. Student-test reference routes are opened only after classifier and HMM predictions have been generated. The API returns predicted geometry and metrics, never the reference route geometry.

## 2. Candidate labels and features

For each GPS point, the strict labeling rule marks at most one candidate positive: the closest retained candidate whose directed arc appears in that trip's reference route. Every other candidate is negative. If the route contains no candidate surviving the radius and cap, the point has no positive candidate. This avoids labeling every nearby parallel road as correct. It can under-label legitimate alternatives when the reference route is incomplete, when a correct arc is outside the search set, or when several route arcs are equally plausible.

The two per-point relative features are retained:

- `dist_rank`: candidate distance rank within its GPS observation.
- `dist_gap_m`: distance separation from neighboring candidates, helping distinguish close alternatives.

They use candidate geometry only and do not use reference labels.

## 3. Candidate-radius experiments

All radius and cap selection used training trips only. The 50 m/cap 10 baseline is a reference; selection was among 150/175/200 m at cap 30, using five-fold GroupKFold by trip. CV macro F1 is the mean of validation-fold scores. Candidate recall and miss rate below are also fold means; pooled values are saved separately where available.

| Radius / cap | CV macro F1 (mean ± SD) | Fold candidate recall | Fold miss rate | Mean candidates / observation | Candidate rows | Mean RF fit time / fold (s) |
|---|---:|---:|---:|---:|---:|---:|
| 50 m / 10 (baseline) | 0.7502 ± 0.0627 | 90.05% | 9.95% | 5.42 | 1,130,397 | 8.60 |
| 150 m / 30 | **0.6017 ± 0.0223** | 93.56% | 6.44% | 18.20 | 3,794,119 | 17.56 |
| 175 m / 30 | 0.5920 ± 0.0582 | 93.74% | 6.26% | 20.28 | 4,226,252 | 18.48 |
| 200 m / 30 | 0.5801 ± 0.0375 | **95.57%** | **4.43%** | 21.95 | 4,574,751 | 16.78 |

The 150 m/cap 30 setting was selected because it had the best RF CV macro F1 among the requested expanded-radius configurations, improved fold candidate recall by 3.51 percentage points over the 50 m baseline, and produced fewer candidates than 175 or 200 m. The 200 m setting adds 0.16 million candidate rows over 150 m while lowering CV macro F1 by 0.0215. The baseline has a higher CV macro F1 but lower candidate recall; it is not hidden or represented as a selected expanded-radius configuration. The full trade-off is in `results/candidate_radius_comparison.csv` and fold scores are in `results/candidate_radius_fold_results.csv` and `results/candidate_radius_baseline_folds.csv`.

The team brief's local estimates (27% correct-candidate coverage and 103 m median nearest-correct-road distance at 50 m) did not reproduce on the data available in this workspace. With the current files and CRS `EPSG:32640`, pooled training candidate recall is 88.36% at 50 m/cap 10 and 92.42% at 150 m/cap 30. The mean nearest correct route-arc distance across training observations has median 6.75 m, 75th percentile 13.20 m, and 95th percentile 157.91 m. Results are produced from this repository's current raw trips and should be checked against any different dataset/version behind the teammate estimates.

At the selected setting, 208,417 training observations produced 3,794,119 candidate rows. The pooled training miss rate (no correct retained candidate) was 7.58%; the zero-candidate rate was 0.113%. All 19,160 student-test observations had at least one correct candidate at this radius. Candidate generation for the shared 200 m superset across 80 training trips took 359 seconds; the 150 m filtering/relabeling pass took 7.11 seconds. Per-configuration model fit/inference timings are in the CSV output.

## 4. Five-model comparison

Models use identical candidate features, strict labels, per-point top-1 decisions, and observation accounting. GroupKFold partitions by trip. Training subsampling keeps all positives and up to ten nearest negatives per GPS point. The student set was scored for reporting after training; it did not select the primary model. Route continuity is N/A in this comparison and is measured in the HMM stage.

| Model | CV macro F1 (mean ± SD) | Student candidate-row accuracy | Precision | Recall | F1 | Macro F1 | Log loss | Observation Top-1 | Candidate miss | Fit time (s) | Inference (µs/candidate) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8013 ± 0.0390** | 0.9785 | 0.7642 | 0.7642 | 0.7642 | **0.8765** | **0.0676** | **0.7642** | 0.00% | 3.79 | 0.097 |
| Gaussian Naive Bayes | 0.7949 ± 0.0451 | 0.9765 | 0.7416 | 0.7416 | 0.7416 | 0.8647 | 0.2209 | 0.7416 | 0.00% | 3.04 | 0.233 |
| Random Forest | 0.6050 ± 0.0235 | 0.9331 | 0.2656 | 0.2656 | 0.2656 | 0.6152 | 1.4016 | 0.2656 | 0.00% | 24.08 | 0.706 |
| Histogram Gradient Boosting | 0.5825 ± 0.0142 | 0.9304 | 0.2359 | 0.2359 | 0.2359 | 0.5997 | 2.1820 | 0.2359 | 0.00% | 11.64 | 1.583 |
| Decision Tree | 0.5755 ± 0.0195 | 0.9305 | 0.2377 | 0.2377 | 0.2377 | 0.6006 | 19.7521 | 0.2377 | 0.00% | 34.15 | 0.121 |

Candidate-row accuracy is high partly because incorrect candidates are common; observation Top-1 accuracy is the more direct measure of whether the selected road arc is correct. Student class-wise metrics are:

| Model | Class 0 P/R/F1 | Class 1 P/R/F1 |
|---|---:|---:|
| Logistic Regression | 0.9887 / 0.9887 / 0.9887 | 0.7642 / 0.7642 / 0.7642 |
| Gaussian Naive Bayes | 0.9877 / 0.9877 / 0.9877 | 0.7416 / 0.7416 / 0.7416 |
| Random Forest | 0.9649 / 0.9649 / 0.9649 | 0.2656 / 0.2656 / 0.2656 |
| Histogram Gradient Boosting | 0.9635 / 0.9635 / 0.9635 | 0.2359 / 0.2359 / 0.2359 |
| Decision Tree | 0.9636 / 0.9636 / 0.9636 | 0.2377 / 0.2377 / 0.2377 |

Each confusion matrix is saved under `results/confusion_matrices/` as CSV and PNG, with counts also in `results/model_comparison.csv`. In `TN, FP, FN, TP` order:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 396,833 | 4,517 | 4,517 | 14,643 |
| Gaussian Naive Bayes | 396,400 | 4,950 | 4,950 | 14,210 |
| Random Forest | 387,278 | 14,072 | 14,072 | 5,088 |
| Histogram Gradient Boosting | 386,709 | 14,641 | 14,641 | 4,519 |
| Decision Tree | 386,744 | 14,606 | 14,606 | 4,554 |

### Model choices, objectives, and generalization

| Model | Default parameters and objective | Strength and limitation |
|---|---|---|
| Logistic Regression | `C=1`, `max_iter=200`; L2-regularized binary log loss | Fast, calibrated probability baseline with a stable linear boundary; can miss nonlinear interactions. Selected because it had the highest training-fold macro F1. |
| Gaussian Naive Bayes | `var_smoothing=1e-9`; Gaussian class-conditional likelihood | Fast generative model; conditional feature-independence and Gaussian assumptions can be restrictive. It was a close second in CV. |
| Decision Tree | Gini, `max_depth=12`, `min_samples_leaf=50`; impurity minimization | Captures nonlinear rules and is interpretable; the single constrained tree was weak on class 1 here. |
| Random Forest | 50 trees, depth 12, `max_samples=0.2`; bagged Gini trees | Captures interactions and reduces single-tree variance; higher training/inference cost and weaker probability calibration. Used for consistent radius selection. |
| Histogram Gradient Boosting | learning rate 0.1, 200 iterations, early stopping; binomial log loss | Efficient nonlinear boosting; performance depends on depth/iteration regularization and was weak on class 1 in this run. |

The selected Logistic Regression has training macro F1 0.7975 versus mean CV 0.8013, so there is no training-to-fold gap suggesting overfit. Student macro F1 is higher (0.8765), but it was not used for selection. Naive Bayes shows a similarly small training/CV gap. The tree models have much lower absolute macro F1; their train/CV gaps are modest, which is more consistent with limited fit under the chosen settings than severe overfit. Random Forest has a small train/CV gap (0.6243 versus 0.6050). Student results are reported only as a final holdout comparison and are not evidence used to tune.

## 5. Optuna and final model

Optuna's seeded TPE sampler ran 12 trials for the already CV-selected Logistic Regression. Each trial used five-fold GroupKFold by trip and mean fold macro F1 as the objective. Only the 80 training trips were loaded. All trial values, states, parameters, durations, and fold scores are in `results/tuning_trials.csv`.

Best trial 1 set `C=56.6984951` with CV macro F1 0.8013504. Several larger-C trials had the same value at the displayed precision, so the best trial is the first maximum. The selected model was refit using training data only and saved to `artifacts/final_model.joblib`; metadata and frozen radius, cap, features, and HMM settings are in `artifacts/model_metadata.json`.

| Student metric | Untuned | Tuned | Change |
|---|---:|---:|---:|
| Accuracy | 0.978517 | 0.978517 | ~0 |
| Precision / recall / F1 | 0.764248 | 0.764248 | 0 |
| Macro F1 | 0.876497 | 0.876497 | 0 |
| Log loss | 0.067592 | 0.067591 | -0.0000008 |
| Observation Top-1 | 0.764248 | 0.764248 | 0 |

Tuning made no meaningful student-test improvement. This comparison was produced after the training-only search finished; no test result was fed back to the tuner.

## 6. Candidate probabilities and missing observations

`results/candidate_probabilities.csv` contains all 420,510 selected-model probabilities for the 10 student-test trips, keyed by `trip_id`, `gps_point_id`, and `candidate_arc_id`, with `selected_model`. It has no reference label. `data/processed/observation_accounting.csv` contains one row for every original GPS point, including candidate count, whether any candidate exists, whether a correct candidate exists, and candidate-miss status.

The shared accounting helper groups by both trip ID and GPS point ID. A missing correct candidate is added as an explicit false negative to candidate confusion metrics and remains in the observation Top-1 denominator. Such a miss cannot mathematically reduce standard precision (`TP/(TP+FP)`) without fabricating a positive prediction; it reduces recall/F1 and accuracy, while candidate-miss rate reports coverage directly. An empty HMM layer stays empty and decodes to a missing state. Disconnected transition layers restart from available emissions rather than inventing a connector.

## 7. EDA

EDA is based on the 80 training trips:

- 208,417 GPS observations; median 1,529 observations per trip (range 361–19,313).
- No missing or invalid coordinates, no nonpositive timestamp gaps, and no implausible-speed flags. Median timestamp gap is 1 second; the mean per-trip 95th-percentile gap is 1.82 seconds (maximum 17 seconds).
- Median trip length 24.07 km (range 6.33–169.25 km). The median per-trip median step distance is 13.18 m; the median per-trip 95th-percentile step distance is 26.86 m.
- At 150 m/cap 30, mean candidate count is 18.20 and the 95th percentile reaches the cap of 30. Pooled zero-candidate rate is 0.113%.
- Road networks have median 9,608 nodes and 19,526 directed arcs. No invalid arcs or isolated nodes were found. Weakly connected component count has median 20 (range 1–421), so local road graphs can still contain disconnected areas.

Saved trip/network tables and plots are in `results/eda/`: GPS trip and road network summaries, trip-length and step-distance histograms, candidate trade-off, nearest route-arc distance, and weak-component plots. Network statistics are weak connectivity because that is the appropriate undirected connectivity summary for these directed arc tables; HMM transitions themselves remain directed.

## 8. HMM/Viterbi route evaluation

The HMM uses only the selected Logistic Regression probabilities as emissions, scored as `log(max(probability_floor, p))`. Directed transition scores compare approximate midpoint-to-midpoint network distance with GPS displacement, normalize over reachable next candidates, obey the configured 5 km network limit and 70 m/s speed cap, and are accumulated in log space by Viterbi. Consecutive repeated predicted arc IDs are collapsed for route-level scoring; `None` states remain in the per-observation prediction file, are skipped when forming the collapsed arc list, and are added as explicit miss penalties to route metrics. The reference route keeps its supplied order and repeated arcs for sequence comparison. Candidate labeling uses route arc set membership, so duplicate reference IDs do not create duplicate positive labels; at most one candidate per observation is positive. Route precision/recall use multiset arc overlap; edit distance is Levenshtein distance; normalized similarity is one minus edit distance over the longer sequence. Exact route match is reported separately.

| Student route metric (mean per trip) | Classifier only | Classifier + HMM/Viterbi |
|---|---:|---:|
| Arc precision | 0.4294 | 0.4928 |
| Arc recall | **0.8411** | 0.6635 |
| Arc F1 | **0.5490** | 0.5440 |
| Edit distance (lower is better) | 517.4 | 366.6 |
| Normalized sequence similarity | **0.3861** | 0.3740 |
| Transition feasibility | 0.9792 | **0.9942** |
| Candidate-miss rate | 0.0000 | 0.0000 |
| Exact route match | 0.0000 | 0.0000 |
| Nearest-reference-arc observation accuracy | **0.7834** | 0.4809 |

HMM did **not** improve aggregate route F1 or normalized sequence similarity. It increased precision and directed transition feasibility, but reduced recall and point-level nearest-reference-arc accuracy. Four of ten student trips improved in route F1; six did not. This supports reporting mixed effects rather than claiming improvement. No student trip had a zero-candidate observation at the frozen setting. Training trip `00000002` provides a diagnostic missing-state example: 218 of 7,243 GPS observations had no candidate, and the Folium map preserves those as red GPS points without assigning an arc.

The exact per-trip classifier/HMM output and metrics are in `results/hmm_predictions.csv` and `results/decoding_comparison.csv`. Plots are in `results/figures/`. Folium maps:

- Successful per-trip HMM change: `results/figures/maps/hmm_improvement_00000007.html`.
- No improvement: `results/figures/maps/hmm_no_improvement_00000000.html`.
- Empty candidate layers (training diagnostic, not a held-out score): `results/figures/maps/missing_candidates_training_00000002.html`.

Each Folium map has toggleable GPS, candidates, classifier, HMM, and reference layers; the reference layer is hidden by default. The API does not return reference geometry.

## 9. API, UI, and dry run

`app/api.py` exposes `GET /api/trips` and `POST /api/decode/{trip_id}` for the student-test allowlist. It generates candidates and classifier/HMM predictions before opening `.route` for evaluation. Request middleware and trip-level logs record incoming IDs, completion status, HTTP status, and elapsed time. A live local request for trip `00000007` returned HTTP 200 with 1,354 observations and `reference_route_exposed=false`; the server console showed both `request.start` and `request.complete` entries.

The saved all-trip run in `results/api_dry_run.csv` passed 10/10 requests. Each response had the full observation count, classifier/HMM metrics, and no reference geometry. The independent slow API test also passed all ten routes. The static UI served HTTP 200 with the trip selector and prediction map elements; its inline JavaScript passed `node --check`. The available browser-control host was unavailable in this environment, so visual interaction in a real browser was not confirmed.

Run the API on Laptop 1:

```powershell
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Run the UI on Laptop 2:

```powershell
python -m http.server 8080 --directory app/ui
```

Open `http://localhost:8080`, enter Laptop 1's reachable address, load trips, and choose a student-test ID. The private evaluator dataset is not available here, so no evaluator-side request or accuracy result is claimed.

## 10. Tests and reproducibility

The default regression suite finished with **43 passed, 1 slow test deselected**. The separate all-10 API test passed; `scripts/7_api_dry_run.py` reproduces it and saves the response summary. Some warnings are from scikit-learn's tiny-fixture Random Forest test and Starlette/httpx deprecation notices; they did not fail tests.

From the repository root:

```powershell
python -m pip install -r requirements.txt
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

The API dry run is a final verification only; it must not be used to change radius, features, model, or HMM settings. Raw datasets, private evaluator data, and machine-specific absolute paths are not committed. Codex assisted with implementation, debugging, and report drafting; the team is responsible for reviewing and presenting the project.
