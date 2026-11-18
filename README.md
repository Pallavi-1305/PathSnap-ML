# PathSnap – Intelligent GPS-to-Road Map Matching

PathSnap is a machine learning-based road map-matching system that identifies the most likely directed road arcs corresponding to noisy vehicle GPS observations.

The system combines candidate generation, supervised machine learning, hyperparameter optimization, and Hidden Markov Model (HMM) decoding to improve the accuracy and continuity of reconstructed vehicle routes.

It also provides a REST API and a lightweight browser-based interface for interacting with the trained model and visualizing predicted routes.

## Key Features

- **GPS Candidate Generation:** Generates nearby directed road arcs for each GPS observation using configurable search radii and candidate limits.
- **Machine Learning Classification:** Implements and compares five classification algorithms to identify the most likely road candidates.
- **Candidate Selection Optimization:** Uses training-only cross-validation to select suitable candidate-generation parameters.
- **Hyperparameter Tuning:** Uses Optuna to optimize the selected machine learning model.
- **HMM-Based Route Decoding:** Applies Viterbi decoding to improve sequential consistency between predicted road arcs.
- **Route Evaluation:** Evaluates candidate-level and route-level performance using precision, recall, F1-score, and sequence metrics.
- **Exploratory Data Analysis:** Generates statistical summaries, evaluation plots, and interactive Folium maps.
- **REST API:** Provides prediction endpoints using a frozen trained model.
- **Browser Interface:** Offers a lightweight interface for submitting trip requests and viewing predicted routes.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Sequence Modeling | HMM, Viterbi Algorithm |
| Hyperparameter Optimization | Optuna |
| Data Processing | Pandas, NumPy |
| API Development | FastAPI, Uvicorn |
| Visualization | Matplotlib, Folium |
| Testing | Pytest |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib |

## System Workflow

The PathSnap pipeline follows these stages:

1. **Data Loading:** Loads GPS observations, road network information, and reference routes.
2. **Exploratory Data Analysis:** Examines GPS characteristics, road network properties, and candidate-generation performance.
3. **Candidate Generation:** Identifies nearby directed road arcs for each GPS observation.
4. **Feature Extraction:** Calculates distance-based and candidate-ranking features.
5. **Model Training:** Trains and compares five machine learning classifiers.
6. **Hyperparameter Optimization:** Uses Optuna to tune the selected classifier using training data.
7. **HMM Decoding:** Applies the Viterbi algorithm to obtain sequentially consistent route predictions.
8. **Evaluation:** Compares classifier-only and HMM-based route predictions.
9. **API Integration:** Loads the frozen model and exposes prediction functionality through a REST API.

## Evaluation Results

The following results represent the current experimental evaluation.

### Dataset

- 80 training trips.
- 10 student-test trips.
- No evaluator trip folder was present or accessed.

### Candidate Generation

- Selected search radius: **150 metres**.
- Maximum candidates per observation: **30**.
- Selection based on five-fold GroupKFold macro F1 using training trips.

### Model Selection

- Selected classifier: **Logistic Regression**.
- Mean training GroupKFold macro F1: **0.8013**.

### Hyperparameter Optimization

- Optimization framework: Optuna.
- Number of trials: 12.
- Best parameter: `C = 56.6985`.
- Best macro F1: **0.80135**.
- Tuned and untuned student-test macro F1: **0.8765** for both.

### HMM Evaluation

| Evaluation Metric | Classifier Only | With HMM |
|---|---:|---:|
| Mean Student Route F1 | 0.5490 | 0.5440 |

The HMM improved precision and transition feasibility but reduced recall and sequence similarity. Therefore, HMM decoding did not produce an overall improvement in route F1 in the current evaluation.

Detailed results and metric definitions are available in [`report/final_report.md`](report/final_report.md).

## Project Structure

```text
PathSnap-ML/
│
├── app/
│   ├── api.py
│   └── ui/
│       └── index.html
│
├── artifacts/
│   ├── final_model.joblib
│   └── model_metadata.json
│
├── data/
│   └── splits.csv
│
├── pathsnap/
│   ├── data.py
│   ├── features.py
│   ├── hmm.py
│   ├── maps.py
│   ├── models.py
│   ├── pipeline.py
│   ├── probabilities.py
│   ├── tune.py
│   └── viz.py
│
├── preprocessing/
│
├── report/
│   └── final_report.md
│
├── results/
│   ├── confusion_matrices/
│   ├── eda/
│   └── figures/
│
├── scripts/
│
├── tests/
│
├── config.yaml
├── requirements.txt
├── pytest.ini
├── run_preprocessing.py
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pallavi-1305/PathSnap-ML.git
cd PathSnap-ML
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Dataset Configuration

Raw dataset files are not included in this repository.

Each trip directory is expected to contain:

- `.track`
- `.nodes`
- `.arcs`
- `.route`

The dataset root is configured through `config.yaml`.

Update `data.raw_root` according to your local dataset directory.

The training and student-test partition is defined in `data/splits.csv`.

## Reproducing the Pipeline

Execute the following commands from the project root directory in the specified order.

**1. Exploratory Data Analysis**

```bash
python scripts/1_eda.py
```

**2. Baseline Candidate Radius Evaluation**

```bash
python scripts/1b_baseline_radius.py
```

**3. Data Preprocessing**

```bash
python run_preprocessing.py
```

**4. Model Comparison**

```bash
python scripts/2_compare_models.py
```

**5. Hyperparameter Tuning**

```bash
python scripts/3_tune.py
```

**6. Export Final Model**

```bash
python scripts/4_export_final.py
```

**7. HMM/Viterbi Decoding**

```bash
python scripts/5_decode_hmm.py
```

**8. Training Missing-Candidate Analysis**

```bash
python scripts/6_training_missing_candidate_map.py
```

**9. API Dry Run**

```bash
python scripts/7_api_dry_run.py
```

**10. Run Tests**

```bash
python -m pytest -q
```

To run the slow API tests separately:

```bash
python -m pytest tests/test_api.py -m slow -q -s
```

## Important Outputs

| Output | Description |
|---|---|
| `results/candidate_radius_comparison.csv` | Candidate radius comparison |
| `results/model_comparison.csv` | Classifier performance comparison |
| `results/model_classification_report.csv` | Classification metrics |
| `results/confusion_matrices/` | Confusion matrices for individual models |
| `results/tuning_trials.csv` | Optuna optimization history |
| `results/best_params.json` | Best hyperparameters |
| `results/tuned_vs_untuned.csv` | Tuned and untuned model comparison |
| `artifacts/final_model.joblib` | Frozen trained model |
| `results/hmm_predictions.csv` | HMM prediction results |
| `results/decoding_comparison.csv` | Classifier and HMM evaluation |
| `results/figures/` | Evaluation visualizations |
| `results/figures/maps/` | Interactive route maps |

## Running the API and Frontend

PathSnap provides a FastAPI backend and a lightweight browser interface.

### 1. Start the API server

By default, the API is bound to this laptop only:

```bash
python -m app.api
```

### 2. Start the frontend

Run on Laptop 2:

```bash
python -m http.server 8080 --directory app/ui
```

### 3. Access the interface

Open the following URL in your browser:

```text
http://localhost:8080
```

On one laptop, open:

```text
http://localhost:8000
```

For two-laptop access, keep `api.host` bound to loopback unless HTTPS is configured.
Set `api.host` to the laptop's reachable address, provide a certificate and private key
through `api.ssl_certfile` and `api.ssl_keyfile`, then start the API with
`python -m app.api`. The frontend must then use `https://` with the matching address
(for example, `https://192.168.1.20:8000`) and trust the configured certificate.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/trips` | Retrieves available student-test trip IDs |
| POST | `/api/decode/{trip_id}` | Generates route predictions for a trip |

The API restricts requests to trip IDs belonging to the `student_test` split.

Predictions are generated before the corresponding reference route is opened for evaluation. The API returns predicted geometry and metrics without exposing reference-route geometry.

All 10 student-test API requests passed the automated dry run.

## Evaluation Methodology

Candidate labels follow a strict labeling strategy.

For each GPS observation, at most the closest candidate arc appearing in the corresponding reference route is marked positive. All other candidates are marked negative.

If no correct road arc survives candidate generation, the observation has no positive candidate. Missing observations are retained rather than assigning an artificial road arc.

The features `dist_rank` and `dist_gap_m` describe candidate distance ranking and separation from neighboring candidates.

Both classifier and HMM evaluations preserve the complete GPS observation sequence. Candidate misses are accounted for in observation-level evaluation, recall, F1-score, and candidate coverage analysis.

Detailed methodology, limitations, and metric definitions are documented in the final report.

## Team Members

This project was developed collaboratively by:

1. **Pallavi Rajkumar Bubanale**
2. **Snehita**
3. **Ramith**
4. **Ramitha**

## AI Assistance Disclosure

AI-assisted tools, including OpenAI Codex, were used to support implementation, debugging, and technical report drafting.

The project team is responsible for reviewing, validating, understanding, and presenting the implementation and experimental results.

---

**PathSnap – Intelligent GPS-to-Road Map Matching**

*Machine Learning | GPS Processing | Road Networks | Route Optimization*
