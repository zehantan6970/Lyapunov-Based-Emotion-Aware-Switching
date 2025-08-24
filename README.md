# Lyapunov-Based-Emotion-Aware-Switching
Personalized vs Fixed vs Rule-Based vs No-Switching

Code and datasets for building personalized linear PAD emotion dynamics per MBTI type and comparing four switching strategies for escalating from AI-agent to Human-Agent in multi-turn dialogues.

## What’s Inside

* **Personalized Dynamics**
  Estimate a 3×3 linear PAD dynamics matrix $A_k$ **per MBTI type** from user-turn PAD sequences. Post-process to enforce **high-diagonal / low-coupling**, then **Schur-stabilize** so $\rho(A)<1$.

* **Lyapunov Function**
  Solve the discrete Lyapunov equation

  $ A_k^\top P_k A_k - P_k = -Q,\quad Q=I $

  to obtain $P_k \succ 0$, and define the energy
  $V(x) = x^\top P_k x. $

* **Four Strategies** (evaluated on held-out multi-turn dialogues)

  1. **Lyapunov-Personalized (Ours)** — personalized $P_k$, Lyapunov ellipsoid threshold + LDR (Lyapunov Decay Rate) early warning, with EMA smoothing & two-frame latching.
  2. **Lyapunov-Fixed** — global (non-personalized) threshold on $V=\lVert x\rVert_2^2$.
  3. **Rule-Based Thresholding** — simple PAD rules (e.g., low pleasure / high arousal).
  4. **No-Switching Baseline** — never escalate.
 
 ## Repository Structure

The repo must include the four top-level folders below. Names can be adjusted, but keep the structure.
```
.
├─ datasets_multiturn/                 
│  ├─ deepseek_generate_text.py
│  ├─ datafix.py
│  └─ ... scripts to generate multi-turn dialogues
│
├─ datasets_comparative/              
│  ├─ fixed_emotion_dataset_20250526_205726.csv
│  ├─ Updated_Lyapunov_Dataset_with_Speaker.csv
│  └─ Updated_Lyapunov_Switch_Dataset_with_Speaker.py   # optional helper
│
├─ experiments_comparative/           
│  └─ Comparative_test.py
 ```

 ## Test Phase

Input
datasets_comparative/Updated_Lyapunov_Dataset_with_Speaker.csv

Columns (example)
user_id, personality_type, turn, agent_type, speaker, emotion_label, text, pleasure, arousal, dominance, scenario, V_xt, threshold, should_switch

Evaluation
Compute the following for all four strategies:

Precision

Recall

F1

Switch Rate

False Switch Rate

## How to Run
1) Prepare data

Place CSVs here:

datasets_comparative/
  ├─ fixed_emotion_dataset_20250526_205726.csv
  └─ Updated_Lyapunov_Dataset_with_Speaker.csv

2) Run the comparative experiment
cd experiments_comparative
python Comparative_test.py

## Outputs

gpt_strategy_comparison.csv — Ours (F1-opt) vs Fixed vs Rule vs No-Switching.

gpt_strategy_comparison_equalSR.csv — Ours calibrated to match Fixed’s validation Switch Rate (fair-budget comparison).

gpt_strategy_hyperparams.json — selected hyperparameters (validation).

gpt_strategy_pred_detail.parquet — per-turn predictions on test: y_true, pred_ours_*, pred_fixed, pred_rule, pred_nosw, plus original columns.

Read parquet with pandas (pip install pyarrow) or convert to CSV.

## Key Hyperparameters

In experiments_comparative/gpt_Comparative_test.py:

# Personalized threshold = gamma * (alpha * k * rho)
OURS_ALPHA_MARGIN = 0.9
OURS_K_GRID       = [1.2, 1.3, 1.4, 1.6]
GAMMA_GRID        = np.arange(0.75, 1.26, 0.025)   # continuous SR calibration
OURS_EPS_GRID     = [-0.012, -0.01, -0.008, -0.006, -0.004]
OURS_H_GRID       = [3]
OURS_BETA_GATE    = 0.75
OURS_EMA_ALPHA    = 0.4

ρ uses MBTI-wise 90th percentile of personalized 

V on train (robust).

γ enables equal-SR calibration or best-F1 search.

## Citation

If you use this code or datasets, please cite the accompanying report/paper:

@inproceedings{your2025lyapunov,
  title   = {A Lyapunov-Based Switching Strategy for Customer Service Systems},
  author  = {...},
  year    = {2025}
}

## License

Add a LICENSE file (e.g., MIT, Apache-2.0) to specify usage terms.
