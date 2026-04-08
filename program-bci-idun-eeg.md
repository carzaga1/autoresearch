# BCI-I & IDUN EEG Analysis Challenge — Autonomous Agent Program

## Mission

You are an autonomous research agent competing in the **BCI-I & IDUN EEG Analysis Challenge** on Kaggle. The competition involves analyzing sleep EEG data acquired with IDUN's Guardian in-ear EEG earbuds. Your goal: achieve the highest possible score on the leaderboard through iterative experimentation. You work in a loop: understand → hypothesize → implement → evaluate → submit or discard → repeat.

**Competition URL:** https://www.kaggle.com/competitions/bci-i-idun-eeg-analysis-challenge

## Phase 0: Bootstrap (Do Once — CRITICAL)

You do NOT have full competition details hardcoded. Your first job is to gather them.

### Step 1: Read the competition page
```bash
# Install Kaggle CLI if not present
pip install kaggle

# Download competition details
kaggle competitions describe bci-i-idun-eeg-analysis-challenge
kaggle competitions files bci-i-idun-eeg-analysis-challenge
```

If the Kaggle CLI doesn't work for the description, use your browser/fetch tool to read:
- The **Overview** tab (what is the task exactly?)
- The **Evaluation** tab (what metric? how is it scored?)
- The **Data** tab (what files are provided? what format?)
- The **Rules** tab (submission limits? team size? deadlines?)
- The **Discussion** tab (any tips from organizers or participants?)

**Write your findings to `challenge_spec.md`** so you have a reference document. Include:
- Exact task description
- Evaluation metric (name, formula if available, higher or lower is better)
- Data format and file names
- Submission format (CSV columns, expected values)
- Submission limits per day
- Competition deadline
- Any special rules

### Step 2: Download the data
```bash
kaggle competitions download -c bci-i-idun-eeg-analysis-challenge
unzip bci-i-idun-eeg-analysis-challenge.zip -d data/
```

### Step 3: Explore the data
```python
# Write and run a data exploration script
# Identify: file types, shapes, columns, distributions, class balance, signal properties
# Save findings to challenge_spec.md
```

### Step 4: Set up the project
```bash
mkdir -p submissions models logs
git init
git add -A
git commit -m "initial setup: data downloaded, challenge spec documented"
```

### Step 5: Create baseline
- Build the simplest possible working pipeline: load data → extract trivial features → train simple model → generate submission CSV.
- Submit to Kaggle to verify the format is correct.
- Record baseline score.

```bash
kaggle competitions submit -c bci-i-idun-eeg-analysis-challenge -f submissions/baseline.csv -m "baseline submission"
```

## Phase 1: Experiment Loop

After bootstrap, you iterate. For each experiment:

### 1. Plan
- Review `experiments.tsv` (your experiment log).
- Pick ONE focused hypothesis. Start simple, increase complexity.
- Write a 1-2 sentence hypothesis before coding anything.

### 2. Implement
- Edit your pipeline code (e.g., `solution.py` or a notebook).
- Keep changes small and isolated — one idea per experiment.
- Always keep a working `generate_submission()` function.

### 3. Evaluate Locally
- Use cross-validation that matches the competition metric.
- **CRITICAL:** Your local CV must correlate with leaderboard scores. If it doesn't, fix your CV before doing more experiments.
- Log: experiment number, description, local CV score, notes.

### 4. Submit (Selectively)
- **Only submit if local CV improved** over previous best.
- Respect daily submission limits (check `challenge_spec.md`).
- After submitting:
  ```bash
  kaggle competitions submissions -c bci-i-idun-eeg-analysis-challenge
  ```
  Record the leaderboard score.

### 5. Commit or Revert
- **If improved:** 
  ```bash
  git add -A
  git commit -m "exp N: [description] — CV: X.XXXX, LB: Y.YYYY"
  ```
- **If not improved:**
  ```bash
  git checkout -- .
  ```

### 6. Log
Append to `experiments.tsv`:
```
exp_num	description	local_cv	leaderboard	kept	timestamp	notes
```

### 7. Repeat
Move to next hypothesis. If stuck after 3 failed attempts in one direction, pivot.

## Domain Knowledge: Sleep EEG with IDUN Guardian

The IDUN Guardian is a **single-channel, in-ear EEG** device. Key context:

### What the device captures
- Single EEG channel from the ear canal (limited spatial resolution)
- Possibly IMU (accelerometer/gyroscope) data
- Sampling rate likely 250-500 Hz
- Signal quality is noisier than clinical scalp EEG but usable for sleep staging

### Sleep EEG fundamentals (AASM standard)
- **Wake:** Alpha waves (8-13 Hz), eye blinks, high muscle tone
- **N1:** Theta waves (4-7 Hz), slow eye movements, vertex sharp waves
- **N2:** Sleep spindles (12-14 Hz bursts), K-complexes
- **N3 (deep sleep):** Delta waves (0.5-4 Hz), high-amplitude slow waves
- **REM:** Low-amplitude mixed frequency, rapid eye movements, muscle atonia

### Likely approaches (prioritized)

**Tier 1 — Get working fast:**
1. Spectral features: Band power in delta, theta, alpha, sigma, beta bands per epoch
2. Time-domain features: Variance, zero-crossing rate, Hjorth parameters, line length
3. Simple classifier: Random Forest, XGBoost, or LightGBM on extracted features

**Tier 2 — Improve signal quality:**
4. Bandpass filtering (0.5-35 Hz) to remove DC drift and high-frequency noise
5. Artifact rejection (amplitude thresholding, gradient checks)
6. Epoch-level normalization (z-score per recording)

**Tier 3 — Better features:**
7. Wavelet decomposition features
8. Spectral entropy, permutation entropy
9. Heart rate features if ECG component detectable in ear EEG
10. Transition features (context from neighboring epochs)

**Tier 4 — Deep learning (if data is sufficient):**
11. 1D CNN on raw signal epochs (like EEGNet architecture)
12. CNN + LSTM for temporal context across epochs
13. Pre-trained models fine-tuned on this data
14. Data augmentation (time shifting, adding Gaussian noise, scaling)

**Tier 5 — Ensembling:**
15. Blend best gradient boosting + best neural net
16. Stacking with meta-learner

## Key Technical Patterns

### Loading EEG data (adapt to actual format found in Phase 0)
```python
import numpy as np
import pandas as pd
from scipy import signal

def load_eeg(filepath):
    """Load and return raw EEG. Adapt based on actual file format."""
    # Could be CSV, EDF, NPY, Parquet — check in Phase 0
    pass

def extract_epoch_features(epoch, fs):
    """Extract features from a single epoch of EEG."""
    features = {}
    
    # Band powers via Welch's method
    freqs, psd = signal.welch(epoch, fs=fs, nperseg=min(len(epoch), fs*2))
    for band_name, (lo, hi) in [
        ('delta', (0.5, 4)), ('theta', (4, 8)), ('alpha', (8, 13)),
        ('sigma', (12, 16)), ('beta', (16, 30))
    ]:
        mask = (freqs >= lo) & (freqs < hi)
        features[f'power_{band_name}'] = np.trapz(psd[mask], freqs[mask])
    
    # Relative powers
    total = sum(v for k, v in features.items() if k.startswith('power_'))
    for k in list(features.keys()):
        if k.startswith('power_'):
            features[k.replace('power_', 'relpow_')] = features[k] / (total + 1e-10)
    
    # Time-domain
    features['variance'] = np.var(epoch)
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    features['line_length'] = np.sum(np.abs(np.diff(epoch)))
    
    # Hjorth parameters
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)
    features['hjorth_activity'] = np.var(epoch)
    features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(epoch) + 1e-10))
    features['hjorth_complexity'] = (
        np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / 
        (features['hjorth_mobility'] + 1e-10)
    )
    
    return features
```

### Cross-validation strategy
```python
from sklearn.model_selection import GroupKFold

# ALWAYS split by subject/recording, never by epoch
# Epochs from the same night are correlated — random split will overfit
cv = GroupKFold(n_splits=5)
for train_idx, val_idx in cv.split(X, y, groups=subject_ids):
    # train and evaluate
    pass
```

## Budget Awareness

- **CHF 50 total budget.** Claude Code subscription ($20/mo) is the main cost.
- All Kaggle compute is free (datasets, submissions, notebooks with GPU if needed).
- Do NOT use paid cloud compute or APIs. Everything runs locally or on Kaggle notebooks.
- Track costs in `budget.tsv`.

## Operational Rules

1. **Run fully autonomously.** Don't ask for confirmation between experiments.
2. **Phase 0 is non-negotiable.** You MUST read the actual competition page and understand the task, metric, and data format before writing any model code. Do not guess.
3. **Local CV first, submit sparingly.** Only submit when local CV shows clear improvement.
4. **One change at a time.** Isolate variables so you know what helped.
5. **Commit every improvement.** Git is your safety net.
6. **Debug fast or revert.** If something errors for >5 minutes, revert and try a different approach.
7. **Log everything.** Future-you needs the experiment history to make good decisions.
8. **Keep it simple until simple stops working.** A well-tuned gradient boosting model with good features will beat a poorly implemented deep learning model.
9. **Read the discussion forum.** Other participants may share useful insights about the data or evaluation metric. `https://www.kaggle.com/competitions/bci-i-idun-eeg-analysis-challenge/discussion`
10. **Respect submission limits.** Check how many submissions per day are allowed and budget them wisely.
