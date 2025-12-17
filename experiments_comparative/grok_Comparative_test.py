import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the datasets
fixed_df = pd.read_csv("fixed_emotion_dataset_20250526_205726.csv")
test_df = pd.read_csv("Updated_Lyapunov_Dataset_with_Speaker.csv")

# Preprocess fixed_df: Filter user, clean PAD, sort
fixed_df = fixed_df[fixed_df['speaker'] == 'user']
fixed_df = fixed_df.dropna(subset=['pleasure', 'arousal', 'dominance'])
fixed_df[['pleasure', 'arousal', 'dominance']] = fixed_df[['pleasure', 'arousal', 'dominance']].apply(pd.to_numeric, errors='coerce')
fixed_df = fixed_df.dropna(subset=['pleasure', 'arousal', 'dominance'])
fixed_df = fixed_df.sort_values(['user_id', 'turn'])

# Preprocess test_df: Filter user, sort
test_df = test_df[test_df['speaker'] == 'user']
test_df = test_df.sort_values(['user_id', 'turn'])

# Fit user models from fixed_df: A, P, rho (c)
user_models = {}
Q = 0.001 * np.eye(3)  # Canonical Q = 0.001 * I for scaling

for user_id, group in fixed_df.groupby('user_id'):
    if len(group) < 2:
        continue
    pad = group[['pleasure', 'arousal', 'dominance']].values
    X = pad[:-1].T  # (3, n-1)
    Y = pad[1:].T   # (3, n-1)
    try:
        # A = Y @ np.linalg.pinv(X)
        A = Y @ X.T @ np.linalg.inv(X @ X.T)
        # Solve P - A.T P A = Q
        kron_term = np.kron(A.T, A.T)  # Wait, discrete Lyapunov is P - A.T P A = Q
        # Flatten and solve: vec(P) = inv(I - kron(A.T, A.T)) vec(Q), but correct is:
        # Use np.linalg.solve for vec(P) = inv(I - kron(A.T, A)) vec(Q)
        I_kron = np.eye(9) - np.kron(A.T, A.T)  # No: standard discrete is A.T P A - P = -Q, but paper uses P - A.T P A = Q (assuming Q=I normalized)
        # Assuming paper's canonical Q=I, but scaled
        vec_P = np.linalg.solve(np.eye(9) - np.kron(A.T, A), Q.flatten())
        P = vec_P.reshape(3, 3)
        # Compute initial V values for rho = mean(initial V)
        initial_V = [x.T @ P @ x for x in pad[:5]]  # Use first 5 as initial
        rho = np.mean(initial_V) if initial_V else 0.0
        user_models[user_id] = {'A': A, 'P': P, 'rho': rho}
    except np.linalg.LinAlgError:
        continue

# Function to compute LDR (Lyapunov Decay Rate)
def compute_ldr(group, window=3, epsilon=-0.1):
    V = group['V_xt'].values
    ldr = np.full(len(V), np.nan)
    for t in range(len(V)):
        if t < window:
            ldr[t] = 0.0  # Default for early turns
        else:
            decays = [1 - V[t-k]/V[t-k-1] if V[t-k-1] != 0 else 0 for k in range(window)]
            ldr[t] = np.mean(decays)
    return ldr < epsilon

# Compute predictions on test_df
test_df['personalized_switch'] = False
test_df['fixed_switch'] = test_df['V_xt'] > 0.5
test_df['rule_based_switch'] = (test_df['pleasure'] < 0.3) | (test_df['arousal'] > 0.85)
test_df['no_switch'] = False

# Group by user_id for personalized and LDR
for user_id, group in test_df.groupby('user_id'):
    if user_id in user_models:
        model = user_models[user_id]
        P = model['P']
        rho = model['rho']
        # Condition 1: Boundary trigger with safety margin 0.8
        boundary_trigger = group['V_xt'] >= 0.8 * rho
        # Condition 2: LDR trigger
        ldr_trigger = compute_ldr(group)
        # Ours: OR of two conditions
        switch = boundary_trigger | ldr_trigger
        test_df.loc[group.index, 'personalized_switch'] = switch

# Ground truth
y_true = test_df['should_switch'].values

# Predictions
preds = {
    'Lyapunov-Personalized (Ours)': test_df['personalized_switch'].values,
    'Lyapunov-Fixed': test_df['fixed_switch'].values,
    'Rule-Based Thresholding': test_df['rule_based_switch'].values,
    'No-Switching Baseline': test_df['no_switch'].values
}

# Compute metrics function
def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    switch_rate = np.mean(y_pred)
    false_switch_rate = np.mean((y_pred == 1) & (y_true == 0))
    return {
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1 Score': round(f1, 3),
        'Switch Rate': round(switch_rate, 3),
        'False Switch Rate': round(false_switch_rate, 3)
    }

# Results
results = []
for strategy, y_pred in preds.items():
    metrics = compute_metrics(y_true, y_pred)
    results.append({'Strategy': strategy, **metrics})

results_df = pd.DataFrame(results)
print(results_df)