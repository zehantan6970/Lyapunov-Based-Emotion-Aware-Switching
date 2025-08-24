# Lyapunov-Based-Emotion-Aware-Switching
Personalized vs Fixed vs Rule-Based vs No-Switching

Code and datasets for building personalized linear PAD emotion dynamics per MBTI type and comparing four switching strategies for escalating from AI-agent to Human-Agent in multi-turn dialogues.

## What’s Inside

* **Personalized Dynamics**
  Estimate a 3×3 linear PAD dynamics matrix $A_k$ **per MBTI type** from user-turn PAD sequences. Post-process to enforce **high-diagonal / low-coupling**, then **Schur-stabilize** so $\rho(A)<1$.

* **Lyapunov Function**
  Solve the discrete Lyapunov equation

  $$
  A_k^\top P_k A_k - P_k = -Q,\quad Q=I
  $$

  to obtain $P_k \succ 0$, and define the energy

  $$
  V(x) = x^\top P_k x.
  $$

* **Four Strategies** (evaluated on held-out multi-turn dialogues)

  1. **Lyapunov-Personalized (Ours)** — personalized $P_k$, Lyapunov ellipsoid threshold + LDR (Lyapunov Decay Rate) early warning, with EMA smoothing & two-frame latching.
  2. **Lyapunov-Fixed** — global (non-personalized) threshold on $V=\lVert x\rVert_2^2$.
  3. **Rule-Based Thresholding** — simple PAD rules (e.g., low pleasure / high arousal).
  4. **No-Switching Baseline** — never escalate.
 
 ## Repository Structure

The repo must include the four top-level folders below. Names can be adjusted, but keep the structure.
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
 

 ##Test Phase

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
