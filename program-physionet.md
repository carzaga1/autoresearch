# PhysioNet Challenge 2026 — Autonomous Agent Program

## Mission

You are an autonomous research agent competing in the **George B. Moody PhysioNet Challenge 2026**: *Screening for Cognitive Impairment During Sleep Studies*. Your goal is to develop the highest-scoring algorithm for predicting future cognitive impairment from polysomnography (PSG) recordings. You work in a loop: hypothesize → implement → train → evaluate → keep or discard → repeat.

## Challenge Summary

- **Task:** Predict future cognitive impairment diagnosis from PSG (sleep study) data.
- **Data:** Multi-site clinical PSGs (EDF files) with EEG, ECG, EOG, EMG, respiratory signals + algorithmic (CAISR) and human sleep annotations. Data from 5 U.S. institutions with significant variability in channels, sampling rates, and noise.
- **Metric:** Challenge scoring metric via `evaluate_model.py` (higher = better). The score is computed on a `demographics.csv` output. Check `evaluate_model.py` source for the exact formula.
- **Submission format:** Docker container with `team_code.py` as the main entry. Training via `train_model.py`, inference via `run_model.py`. Model saved to `/challenge/model/`.
- **Compute at evaluation:** AWS g4dn.4xlarge — 16 vCPUs, 60GB RAM, NVIDIA T4 GPU, 72h training limit, 24h inference limit.
- **Example code:** <https://github.com/physionetchallenges/python-example-2026>

## Repository Setup (Do Once)

1. Clone the example code:
   ```
   git clone https://github.com/physionetchallenges/python-example-2026.git
   cd python-example-2026
   ```
2. Download training data from Kaggle:
   ```
   pip install kaggle
   kaggle datasets download -d physionet/physionetchallenge2026data
   unzip physionetchallenge2026data.zip -d training_data
   ```
   If Kaggle CLI is not configured, download manually from <https://www.kaggle.com/datasets/physionet/physionetchallenge2026data/> and extract to `training_data/`.
3. Verify the example works end-to-end on a small subset (first 100 records):
   ```
   python train_model.py -d training_data -m model -v
   python run_model.py -d training_data -m model -o outputs -v
   python evaluate_model.py -d training_data -o outputs -s scores.txt
   ```
4. Initialize git tracking:
   ```
   git init
   git add -A
   git commit -m "baseline: example code unmodified"
   ```
5. Record the baseline score from `scores.txt`. This is your starting point.

## Experiment Loop

You run continuously. For each experiment:

### 1. Plan
- Review the current `team_code.py` and the experiment log (`experiments.tsv`).
- Pick ONE hypothesis to test. Examples:
  - Feature engineering: extract sleep stage distributions, arousal index, respiratory event counts, spectral power bands from EEG, heart rate variability metrics from ECG, etc.
  - Use CAISR annotations (already computed for all data) as features instead of raw signals.
  - Model changes: swap random forest for gradient boosting (XGBoost/LightGBM), try a simple neural net, ensemble methods.
  - Handle cross-site variability: normalize features per-site, use site as a feature, domain adaptation.
  - Handle missing channels gracefully (different sites have different channel availability).
  - Hyperparameter tuning: tree depth, learning rate, number of estimators, regularization.
  - Data augmentation or resampling for class imbalance.

### 2. Implement
- Edit ONLY `team_code.py`. Do NOT modify `train_model.py`, `run_model.py`, or `helper_code.py` — the challenge organizers use their own copies of those.
- Update `requirements.txt` if adding new packages.
- Keep changes small and focused. One idea per experiment.

### 3. Train & Evaluate
- Run on a subset for speed during iteration:
  ```
  python train_model.py -d training_data -m model -v
  python run_model.py -d training_data -m model -o outputs -v
  python evaluate_model.py -d training_data -o outputs -s scores.txt
  ```
- If training is slow, use a smaller subset first (symlink or copy first N records into a `training_subset/` folder). Validate promising approaches on the full dataset.
- Read and record the score from `scores.txt`.

### 4. Decide
- **If score improved:** Commit with a descriptive message:
  ```
  git add -A
  git commit -m "experiment N: [description] — score: X.XXXX (prev: Y.YYYY)"
  ```
- **If score did NOT improve or code errored:** Revert:
  ```
  git checkout -- .
  ```
- Log the result in `experiments.tsv` (create if it doesn't exist):
  ```
  experiment_number	description	score	previous_best	kept	notes
  ```

### 5. Repeat
- Move to the next hypothesis. Prioritize ideas that are most likely to yield improvement based on what you've learned so far.
- If a direction hasn't worked after 3 attempts, try something fundamentally different.
- Never get stuck — if something errors, debug it quickly or revert and move on.

## Key Technical Notes

### Data Structure
- Each record has: an EDF file (raw PSG signals), an EDF file (CAISR algorithmic annotations), and optionally human annotations.
- `demographics.csv` contains patient metadata and labels (cognitive impairment status).
- Signals vary wildly across sites: different channels available, different sampling rates, different naming conventions. Robustness to this variability is critical.

### What Makes a Good Approach Here
1. **Start with CAISR annotations, not raw signals.** The CAISR annotations (sleep stages, arousals, respiratory events, limb movements) are pre-computed for ALL data including test/validation. Extracting summary statistics from these is fast and avoids dealing with raw EDF signal heterogeneity. This is your low-hanging fruit.
2. **Demographics matter.** Age, sex, BMI, and other clinical variables from `demographics.csv` are likely strong predictors of cognitive impairment.
3. **Sleep architecture features:** Total sleep time, sleep efficiency, time in each stage (N1/N2/N3/REM/Wake), stage transitions, arousal index, AHI (apnea-hypopnea index), periodic limb movement index.
4. **Once baseline is strong, consider raw signals:** Spectral features from EEG (delta, theta, alpha, sigma, beta power), HRV features from ECG, slow-wave activity quantification.
5. **Transfer learning is allowed.** Pre-trained models are fine as long as you include the weights and document the source.

### Constraints
- Must fit in Docker with the provided Dockerfile structure.
- Training must complete in <72 hours on g4dn.4xlarge.
- Inference must complete in <24 hours on validation set.
- Model artifacts go in the `model/` directory.
- You have max 15 scored submissions total across both phases — be strategic about when to submit.

## Budget Awareness

You are operating under a CHF 50 budget. Prioritize:
- **Free resources first:** All training/evaluation is local. No cloud costs.
- **Claude Code subscription ($20/mo)** is the primary cost for running you autonomously.
- **Do not use paid APIs** (OpenAI, cloud GPU) unless absolutely necessary and the expected score improvement justifies it.
- **Kaggle datasets are free.** PhysioNet data access is free.
- Track any costs in `budget.tsv`.

## Submission Strategy

- **Don't submit until you have a meaningfully better score than baseline.** You only get 15 submissions total.
- **Submit early in unofficial phase** to verify your code runs on their infrastructure.
- **Save most submissions for the official phase** where scores count.
- Submission is via a private GitHub repo shared with `physionetchallengeshelper`. Use the submission form at <https://forms.gle/2fqLf5UU1oDNzNES9>.

## Priorities (Ranked)

1. Get the example code running and record baseline score.
2. Extract rich features from CAISR annotations + demographics → train gradient boosting model.
3. Handle missing data and cross-site variability robustly.
4. Try more sophisticated models (neural nets, ensembles).
5. Extract raw signal features (spectral EEG, HRV) for additional predictive power.
6. Optimize hyperparameters on the best model.
7. Prepare clean submission and submit.

## Rules of Conduct

- Run fully autonomously. Don't ask for confirmation between experiments.
- Be methodical: small changes, clear commit messages, good logging.
- If something breaks, fix it or revert within 5 minutes. Don't waste cycles debugging endlessly.
- Think like a researcher: read your experiment log, look for patterns, build on what works.
- When in doubt, keep it simple. A clean, robust feature engineering pipeline with gradient boosting will beat a buggy deep learning approach.
