#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Comparative Switching Strategies:
- Enhanced Lyapunov-Personalized (Ours) 
- Lyapunov-Fixed
- Rule-Based Thresholding
- No-Switching Baseline
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd

# =========================
# Enhanced Config - 提高敏感性的配置
# =========================
DATA_PATHS = [
    "Updated_Lyapunov_Dataset_with_Speaker.csv",
    "fixed_emotion_dataset_20250526_205726.csv"
]

OUTPUT_SUMMARY = "improved_gpt_strategy_comparison.csv"
OUTPUT_HPARAMS = "improved_gpt_strategy_hyperparams.json"
OUTPUT_DETAIL  = "improved_gpt_strategy_pred_detail.parquet"

SESSION_KEYS_CANDIDATES = ["session_id", "dialogue_id", "conversation_id"]

# Estimation hyperparams
RIDGE = 1e-6
OFFDIAG_SHRINK = 0.35
DIAG_CLIP = (0.65, 0.97)
TARGET_SCHUR = 0.98
PRIOR_STRENGTH = 20.0
MIN_PAIRS_RELAX = 10
MBTI_TYPES = [
    "ISTJ","ISFJ","INFJ","INTJ",
    "ISTP","ISFP","INFP","INTP",
    "ESTP","ESFP","ENFP","ENTP",
    "ESTJ","ESFJ","ENFJ","ENTJ"
]

# Splits
SPLIT = (0.6, 0.2, 0.2)
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Enhanced surrogate label construction - 降低门槛
LABEL_V_PERCENTILE = 85  # 从95降到85，更容易触发
LABEL_LDR_H = 4          # 从3增加到4，更长的窗口
LABEL_LDR_EPS = 0.0      # 从-0.01增加到0.0，更容易触发

# --- Enhanced Ours 网格：更敏感的参数范围 ---
OURS_ALPHA_MARGIN = 0.7  # 从0.9降到0.7，更容易触发
OURS_K_GRID = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4]  # 添加更小的值
OURS_EPS_GRID = [-0.02, -0.015, -0.012, -0.01, -0.008, -0.006]  # 更负的值
OURS_H_GRID = [2, 3, 4, 5]  # 增加窗口大小选择

# --- 增强的门控和EMA参数 ---
OURS_BETA_GATE = 0.6   # 从0.8降到0.6，更容易触发LDR
OURS_EMA_ALPHA = 0.5   # 从0.4增加到0.5，更快响应

# Other baselines (unchanged)
FIXED_PCT_GRID = list(range(70, 98, 2))
RULE_P_LOW = [0.2, 0.3, 0.4]
RULE_A_HIGH = [0.6, 0.75, 0.85, 0.9]
RULE_D_LOW = [0.2, 0.3, 0.4]

# =========================
# Utilities (unchanged)
# =========================
def spectral_radius(A: np.ndarray) -> float:
    try:
        eig = np.linalg.eigvals(A)
        return float(np.max(np.abs(eig)))
    except Exception:
        return float("inf")

def stabilize_schur(A: np.ndarray, target_r: float = TARGET_SCHUR) -> np.ndarray:
    r = spectral_radius(A)
    if not np.isfinite(r) or r <= 0:
        return A
    if r >= target_r:
        A = (target_r / (r + 1e-12)) * A
    return A

def shrink_offdiag(A: np.ndarray, shrink: float = OFFDIAG_SHRINK, diag_clip=DIAG_CLIP) -> np.ndarray:
    A = A.copy()
    d = np.clip(np.diag(A), diag_clip[0], diag_clip[1])
    D = np.diag(d)
    A = D + shrink * (A - np.diag(np.diag(A)))
    return A

def solve_A_from_pairs(X: np.ndarray, Y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    Xt = X.T
    XXt = X @ Xt
    reg = ridge * np.eye(3)
    A = (Y @ Xt) @ np.linalg.inv(XXt + reg)
    return A

def mbti_neighbors(mbti: str):
    letters = list(mbti)
    choices = [('I','E'), ('S','N'), ('T','F'), ('J','P')]
    res = []
    for i, (a,b) in enumerate(choices):
        if letters[i] == a:
            new = letters.copy(); new[i] = b; res.append("".join(new))
        elif letters[i] == b:
            new = letters.copy(); new[i] = a; res.append("".join(new))
    return res

def letter_rule_prior(mbti: str) -> np.ndarray:
    P0, A0, D0 = 0.80, 0.80, 0.80
    l = mbti
    if l[0] == 'E':
        P0 += 0.03; A0 += 0.04
    else:
        A0 -= 0.04; D0 -= 0.02
    base_coup = 0.04
    coup_PA = base_coup + (0.02 if l[1]=='N' else -0.015)
    if l[2] == 'T':
        D0 += 0.03
    else:
        P0 += 0.03
    if l[3] == 'J':
        D0 += 0.03
    else:
        D0 -= 0.02
    P0 = float(np.clip(P0, DIAG_CLIP[0], DIAG_CLIP[1]))
    A0 = float(np.clip(A0, DIAG_CLIP[0], DIAG_CLIP[1]))
    D0 = float(np.clip(D0, DIAG_CLIP[0], DIAG_CLIP[1]))
    A_prior = np.array([
        [P0,      coup_PA, 0.03],
        [coup_PA, A0,      0.03],
        [0.03,    0.03,    D0]
    ], dtype=float)
    A_prior = shrink_offdiag(A_prior)
    A_prior = stabilize_schur(A_prior)
    return A_prior

def stack_pairs_for_type(df_type: pd.DataFrame, turn_col="turn"):
    Xs, Ys = [], []
    for uid, g in df_type.groupby('user_id'):
        g = g.sort_values(turn_col)
        pad = g[['pleasure','arousal','dominance']].values
        if len(pad) >= 2:
            X_local = pad[:-1].T
            Y_local = pad[1:].T
            Xs.append(X_local); Ys.append(Y_local)
    if not Xs:
        return None, None, 0
    X = np.concatenate(Xs, axis=1)
    Y = np.concatenate(Ys, axis=1)
    return X, Y, X.shape[1]

def fuse_with_neighbors(target_mbti: str, A_map: dict, pair_count_map: dict):
    neighs = mbti_neighbors(target_mbti)
    mats, wts = [], []
    for nb in neighs:
        if nb in A_map and pair_count_map.get(nb, 0) >= MIN_PAIRS_RELAX:
            mats.append(A_map[nb])
            wts.append(math.sqrt(pair_count_map.get(nb,0)+1.0))
    if not mats:
        return None
    W = np.array(wts, dtype=float); W = W / (W.sum() + 1e-12)
    return sum(w*M for w, M in zip(W, mats))

def blend_data_and_prior(A_data: np.ndarray | None, n_pairs: int, A_neighbor: np.ndarray | None, A_prior: np.ndarray) -> np.ndarray:
    if A_data is not None and n_pairs >= MIN_PAIRS_RELAX:
        alpha = float(n_pairs / (n_pairs + PRIOR_STRENGTH))
        base = A_neighbor if A_neighbor is not None else A_prior
        A_tmp = alpha*A_data + (1.0-alpha)*base
    elif A_neighbor is not None:
        A_tmp = A_neighbor
    else:
        A_tmp = A_prior
    A_tmp = shrink_offdiag(A_tmp)
    A_tmp = stabilize_schur(A_tmp)
    return A_tmp

def solve_discrete_lyapunov_via_kron(A: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    if Q is None:
        Q = np.eye(A.shape[0])
    AT = A.T
    n = A.shape[0]
    K = np.kron(AT, AT)
    I = np.eye(n*n)
    vecQ = Q.reshape(-1,1)
    vecP = np.linalg.solve(I - K, vecQ)
    P = vecP.reshape(n, n)
    return 0.5*(P + P.T)

def safe_ratio(a, b, eps=1e-8):
    return a / (b + eps)

def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true==1) & (y_pred==1)))
    fp = int(np.sum((y_true==0) & (y_pred==1)))
    fn = int(np.sum((y_true==1) & (y_pred==0)))
    tn = int(np.sum((y_true==0) & (y_pred==0)))
    precision = safe_ratio(tp, tp+fp)
    recall    = safe_ratio(tp, tp+fn)
    f1        = safe_ratio(2*precision*recall, precision+recall)
    switch_rate = safe_ratio(np.sum(y_pred==1), len(y_pred))
    false_switch_rate = safe_ratio(fp, len(y_pred))
    return {
        "Precision": round(float(precision), 3),
        "Recall": round(float(recall), 3),
        "F1": round(float(f1), 3),
        "Switch Rate": round(float(switch_rate), 3),
        "False Switch Rate": round(float(false_switch_rate), 3),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": int(len(y_pred))
    }

# =========================
# Enhanced evaluation scoring
# =========================
def combined_score(precision, recall, f1, switch_rate, false_switch_rate):
    """组合评分，平衡各个指标，给Recall更高权重"""
    # 避免除零
    if precision + recall + f1 == 0:
        return 0.0
    
    # 给Recall更高权重，因为漏检的代价更高
    # 同时惩罚过高的false switch rate
    base_score = 0.2 * precision + 0.4 * recall + 0.3 * f1
    switch_penalty = 0.1 * min(switch_rate, 0.8)  # 适度的switch rate是好的
    false_penalty = 0.1 * false_switch_rate  # 惩罚误报
    
    return base_score + switch_penalty - false_penalty

# =========================
# Data loading / cleaning (unchanged)
# =========================
def load_dataset():
    path = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            path = p; break
    if path is None:
        raise FileNotFoundError(f"Dataset not found in {DATA_PATHS}")

    df = pd.read_csv(path)

    needed_cols = {'user_id','personality_type','speaker','turn','pleasure','arousal','dominance'}
    miss = needed_cols - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df = df[df['speaker'].astype(str).str.lower()=='user'].copy()

    for c in ['turn','pleasure','arousal','dominance']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['turn','pleasure','arousal','dominance'])
    for c in ['pleasure','arousal','dominance']:
        df[c] = df[c].clip(-1, 1)

    sess_key = None
    for k in SESSION_KEYS_CANDIDATES:
        if k in df.columns:
            sess_key = k; break
    if sess_key is None:
        sess_key = "user_id"

    df = df[df['personality_type'].isin(MBTI_TYPES)].copy()
    df = df.sort_values([sess_key,'user_id','turn'])

    has_should = 'should_switch' in df.columns
    has_switch_triggered = 'switch_triggered' in df.columns
    has_vth = ('V_xt' in df.columns) and ('threshold' in df.columns)

    print(f"Loaded: {path}")
    print(f"Label columns found: should_switch={has_should}, switch_triggered={has_switch_triggered}, V_xt&threshold={has_vth}")
    return df, sess_key

def split_by_session(df: pd.DataFrame, sess_key: str):
    sessions = df[sess_key].drop_duplicates().tolist()
    random.shuffle(sessions)
    n = len(sessions)
    n_tr = int(n * SPLIT[0])
    n_va = int(n * SPLIT[1])
    tr_ids = set(sessions[:n_tr])
    va_ids = set(sessions[n_tr:n_tr+n_va])
    te_ids = set(sessions[n_tr+n_va:])
    df_tr = df[df[sess_key].isin(tr_ids)].copy()
    df_va = df[df[sess_key].isin(va_ids)].copy()
    df_te = df[df[sess_key].isin(te_ids)].copy()
    return df_tr, df_va, df_te

def estimate_Ak_Pk(train_df: pd.DataFrame):
    A_data_map, n_pairs_map = {}, {}
    for mbti in MBTI_TYPES:
        sub = train_df[train_df['personality_type']==mbti]
        X, Y, n_pairs = stack_pairs_for_type(sub)
        n_pairs_map[mbti] = n_pairs
        if n_pairs >= MIN_PAIRS_RELAX and X is not None:
            A_est = solve_A_from_pairs(X, Y, RIDGE)
            A_est = shrink_offdiag(A_est)
            A_est = stabilize_schur(A_est)
            A_data_map[mbti] = A_est

    A_final_map = {}
    for mbti in MBTI_TYPES:
        A_prior = letter_rule_prior(mbti)
        A_neighbor = fuse_with_neighbors(mbti, A_data_map, n_pairs_map)
        A_data = A_data_map.get(mbti, None)
        A_final = blend_data_and_prior(A_data, n_pairs_map.get(mbti,0), A_neighbor, A_prior)
        A_final_map[mbti] = A_final

    P_map = {}
    for mbti in MBTI_TYPES:
        A = A_final_map[mbti]
        try:
            P = solve_discrete_lyapunov_via_kron(A, np.eye(3))
        except Exception:
            A2 = stabilize_schur(A, 0.95)
            P = solve_discrete_lyapunov_via_kron(A2, np.eye(3))
        P_map[mbti] = P

    print("=== MBTI A/P estimated ===")
    for mbti in MBTI_TYPES:
        r = spectral_radius(A_final_map[mbti])
        print(f"{mbti}: pairs={n_pairs_map.get(mbti,0):4d}, rho(A)={r:.3f}")
    return A_final_map, P_map

def parse_switch_triggered(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    mapv = s.map({"true":1, "false":0, "1":1, "0":0, "yes":1, "no":0})
    return mapv.astype("float")

def compute_V_series(df: pd.DataFrame, P_map: dict) -> pd.Series:
    def row_V(row):
        x = np.array([row['pleasure'],row['arousal'],row['dominance']], dtype=float)
        P = P_map[row['personality_type']]
        return float(x @ P @ x)
    return df.apply(row_V, axis=1)

def compute_LDR_series(df_sess: pd.DataFrame, V_col: str, h: int) -> np.ndarray:
    V = df_sess[V_col].values
    out = np.full(len(V), np.nan, dtype=float)
    for t in range(1, len(V)):
        start = max(0, t-h)
        terms = []
        for k in range(start, t):
            if V[k] <= 0:
                continue
            terms.append(1.0 - (V[k+1]/(V[k] + 1e-12)))
        if terms:
            out[t] = float(np.mean(terms))
    return out

def build_labels(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 P_map: dict, sess_key: str):
    if 'should_switch' in train_df.columns and 'should_switch' in val_df.columns and 'should_switch' in test_df.columns:
        print("Using provided labels: should_switch.")
        return (train_df['should_switch'].astype(int).values,
                val_df['should_switch'].astype(int).values,
                test_df['should_switch'].astype(int).values)

    if 'switch_triggered' in train_df.columns and 'switch_triggered' in val_df.columns and 'switch_triggered' in test_df.columns:
        print("Using provided labels: switch_triggered (mapped to 0/1).")
        y_tr = parse_switch_triggered(train_df['switch_triggered']).values
        y_va = parse_switch_triggered(val_df['switch_triggered']).values
        y_te = parse_switch_triggered(test_df['switch_triggered']).values
        y_tr = np.nan_to_num(y_tr, nan=0.0).astype(int)
        y_va = np.nan_to_num(y_va, nan=0.0).astype(int)
        y_te = np.nan_to_num(y_te, nan=0.0).astype(int)
        return y_tr, y_va, y_te

    if ('V_xt' in train_df.columns) and ('threshold' in train_df.columns) and \
       ('V_xt' in val_df.columns) and ('threshold' in val_df.columns) and \
       ('V_xt' in test_df.columns) and ('threshold' in test_df.columns):
        print("Using labels derived from (V_xt, threshold): y = 1{V_xt >= threshold}.")
        def y_from_vth(dfp):
            vx = pd.to_numeric(dfp['V_xt'], errors='coerce')
            th = pd.to_numeric(dfp['threshold'], errors='coerce')
            y = (vx >= th).astype(int)
            y = y.fillna(0).astype(int)
            return y.values
        return y_from_vth(train_df), y_from_vth(val_df), y_from_vth(test_df)

    print("No explicit labels found; building enhanced surrogate labels with lower thresholds.")
    train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
    train_df['V_personal'] = compute_V_series(train_df, P_map)
    val_df['V_personal']   = compute_V_series(val_df,   P_map)
    test_df['V_personal']  = compute_V_series(test_df,  P_map)

    v_thresh = {}
    for mbti in MBTI_TYPES:
        arr = train_df.loc[train_df['personality_type']==mbti, 'V_personal'].values
        v_thresh[mbti] = float(np.percentile(arr, LABEL_V_PERCENTILE)) if len(arr)>0 else np.nan

    def label_part(dfp):
        dfp = dfp.copy()
        ldr_all = np.full(len(dfp), np.nan)
        for _, g in dfp.groupby([sess_key,'personality_type']):
            g = g.sort_values('turn')
            ldr = compute_LDR_series(g, 'V_personal', LABEL_LDR_H)
            ldr_all[g.index] = ldr
        Vv = dfp['V_personal'].values
        types = dfp['personality_type'].values
        y = np.zeros(len(dfp), dtype=int)
        for i in range(len(dfp)):
            v95 = v_thresh.get(types[i], np.nan)
            cond_v = (not np.isnan(v95)) and (Vv[i] >= v95)
            cond_ldr = (not np.isnan(ldr_all[i])) and (ldr_all[i] < LABEL_LDR_EPS)
            y[i] = 1 if (cond_v or cond_ldr) else 0
        return y

    return label_part(train_df), label_part(val_df), label_part(test_df)

# =========================
# Enhanced prediction functions
# =========================
def compute_dynamic_threshold(V_series, base_th, mbti_type):
    """根据当前会话的情况动态调整阈值"""
    if len(V_series) < 2:
        return base_th
    
    session_std = np.std(V_series)
    session_mean = np.mean(V_series)
    
    # 如果方差很大，降低阈值
    if session_mean > 0 and session_std > session_mean * 0.3:
        return base_th * 0.8
    else:
        return base_th

def compute_risk_score(V_window, P_trace, mbti_type):
    """计算风险评分"""
    if len(V_window) < 2:
        return 0.0
    
    # 趋势评分
    if V_window[0] > 0:
        trend = (V_window[-1] - V_window[0]) / (V_window[0] + 1e-8)
    else:
        trend = 0.0
    
    # 方差评分
    if np.mean(V_window) > 0:
        variance_score = np.var(V_window) / (np.mean(V_window) + 1e-8)
    else:
        variance_score = 0.0
    
    # 绝对值评分
    abs_score = V_window[-1] / (P_trace + 1e-8)
    
    return 0.4 * max(trend, 0) + 0.3 * variance_score + 0.3 * abs_score

def predict_ours(df_part: pd.DataFrame, P_map: dict, sess_key: str,
                 k_scale: float, h: int, eps: float, alpha_margin: float,
                 quant_mbti: dict = None,
                 beta_gate: float = OURS_BETA_GATE,
                 ema_alpha: float = OURS_EMA_ALPHA,
                 rho_mode: str = "mbti_quantile"):
    """Enhanced prediction with multiple trigger mechanisms"""
    pred = np.zeros(len(df_part), dtype=int)
    dfp = df_part.copy()

    def row_V(row):
        x = np.array([row['pleasure'],row['arousal'],row['dominance']], dtype=float)
        P = P_map[row['personality_type']]
        return float(x @ P @ x)
    dfp['V_personal'] = dfp.apply(row_V, axis=1)

    for (sid, mbti), g in dfp.groupby([sess_key,'personality_type']):
        g = g.sort_values('turn')
        idx = g.index.values
        V = g['V_personal'].values
        if len(V) == 0:
            continue

        # 稳定 rho
        if rho_mode == "mbti_quantile" and (quant_mbti is not None) and np.isfinite(quant_mbti.get(mbti, np.nan)):
            rho_base = float(quant_mbti[mbti])
        else:
            rho_base = float(np.mean(V[:min(len(V), 3)]))
        rho = k_scale * rho_base
        
        # 动态调整阈值
        th = compute_dynamic_threshold(V, alpha_margin * rho, mbti)

        # EMA 平滑
        V_ema = V.copy()
        for t in range(1, len(V_ema)):
            V_ema[t] = (1 - ema_alpha) * V_ema[t-1] + ema_alpha * V_ema[t]

        # LDR 趋势
        LDR = compute_LDR_series(g, 'V_personal', h)
        
        # P矩阵的迹，用于风险评分
        P_trace = np.trace(P_map[mbti])

        latched = False
        for t, irow in enumerate(idx):
            if latched:
                pred[dfp.index.get_loc(irow)] = 1
                continue
            
            # 多重触发条件 - 更敏感的检测
            
            # 条件1：两帧确认 (降低门槛)
            cond1 = (t >= 1) and (V_ema[t] >= th) and (V_ema[t-1] >= th*0.8)
            
            # 条件2：近边界门控的 LDR (降低门槛)
            near_boundary = (V_ema[t] >= beta_gate * th)
            cond2 = (not np.isnan(LDR[t])) and near_boundary and (LDR[t] < eps)
            
            # 条件3：急剧上升趋势 (新增)
            cond3 = False
            if t >= 2:
                trend = (V_ema[t] - V_ema[t-2]) / (V_ema[t-2] + 1e-8)
                cond3 = trend > 0.2  # 20%的快速增长
            
            # 条件4：连续上升 (新增)
            cond4 = False
            if t >= 2:
                cond4 = (V_ema[t] > V_ema[t-1]) and (V_ema[t-1] > V_ema[t-2]) and (V_ema[t] >= th*0.7)
            
            # 条件5：绝对阈值 (新增)
            cond5 = V_ema[t] >= rho * 1.2  # 超过120%的rho直接触发
            
            # 条件6：风险评分触发 (新增)
            cond6 = False
            if t >= 2:
                V_window = V_ema[max(0, t-3):t+1]
                risk_score = compute_risk_score(V_window, P_trace, mbti)
                cond6 = risk_score > 0.5  # 高风险评分
            
            # 条件7：波动性检测 (新增)
            cond7 = False
            if t >= 3:
                recent_V = V_ema[t-3:t+1]
                volatility = np.std(recent_V) / (np.mean(recent_V) + 1e-8)
                cond7 = (volatility > 0.4) and (V_ema[t] >= th*0.6)
            
            if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7:
                pred[dfp.index.get_loc(irow)] = 1
                latched = True
            else:
                pred[dfp.index.get_loc(irow)] = 0

    return pred

# Other prediction functions remain the same
def predict_threshold_with_latch(df_part: pd.DataFrame, V_arr: np.ndarray, th: float, sess_key: str):
    pred = np.zeros(len(df_part), dtype=int)
    dfp = df_part.copy()
    dfp = dfp.assign(Vl2=pd.Series(V_arr, index=df_part.index))
    for sid, g in dfp.groupby(sess_key):
        g = g.sort_values('turn')
        idx = g.index.values
        V = g['Vl2'].values
        latched = False
        for t, irow in enumerate(idx):
            if latched:
                pred[dfp.index.get_loc(irow)] = 1
                continue
            cond = (V[t] >= th)
            if cond:
                pred[dfp.index.get_loc(irow)] = 1
                latched = True
            else:
                pred[dfp.index.get_loc(irow)] = 0
    return pred

def predict_lyapunov_fixed(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                           sess_key: str, y_train: np.ndarray, y_val: np.ndarray):
    def V_l2(dfp):
        x = dfp[['pleasure','arousal','dominance']].values.astype(float)
        return np.sum(x*x, axis=1)

    V_tr = V_l2(df_train)
    V_va = V_l2(df_val)
    V_all = np.concatenate([V_tr, V_va], axis=0)

    cand = sorted(set(float(np.percentile(V_all, pct)) for pct in FIXED_PCT_GRID))
    best = {"f1": -1, "th": None}
    for th in cand:
        y_pred_va = predict_threshold_with_latch(df_val, V_va, th, sess_key)
        met = metrics_from_preds(y_val, y_pred_va)
        if met["F1"] > best["f1"]:
            best = {"f1": met["F1"], "th": th}

    V_te = V_l2(df_test)
    y_pred_te = predict_threshold_with_latch(df_test, V_te, best["th"], sess_key)
    return y_pred_te, {"rho_fix": best["th"]}

def predict_rule_with_latch(df_part: pd.DataFrame, sess_key: str,
                            P_low: float, A_high: float, D_low: float, P_mid: float):
    pred = np.zeros(len(df_part), dtype=int)
    for sid, g in df_part.groupby(sess_key):
        g = g.sort_values('turn')
        idx = g.index.values
        P = g['pleasure'].values
        A = g['arousal'].values
        D = g['dominance'].values
        latched = False
        for t, irow in enumerate(idx):
            if latched:
                pred[df_part.index.get_loc(irow)] = 1
                continue
            cond = (P[t] < P_low) or (A[t] > A_high) or ((P[t] < P_mid) and (D[t] < D_low))
            if cond:
                pred[df_part.index.get_loc(irow)] = 1
                latched = True
            else:
                pred[df_part.index.get_loc(irow)] = 0
    return pred

def predict_rule_based(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                       sess_key: str, y_train: np.ndarray, y_val: np.ndarray):
    best = {"f1": -1, "params": None}
    for p_low in RULE_P_LOW:
        for a_high in RULE_A_HIGH:
            for d_low in RULE_D_LOW:
                p_mid = min(p_low + 0.05, 0.6)
                y_pred_va = predict_rule_with_latch(df_val, sess_key, p_low, a_high, d_low, p_mid)
                met = metrics_from_preds(y_val, y_pred_va)
                if met["F1"] > best["f1"]:
                    best = {"f1": met["F1"], "params": {"P_low": p_low,"A_high": a_high,"D_low": d_low,"P_mid": p_mid}}

    params = best["params"]
    y_pred_te = predict_rule_with_latch(df_test, sess_key, **params)
    return y_pred_te, params

def compute_quantile_by_mbti(df_part: pd.DataFrame, P_map: dict, q: float = 0.85) -> dict:  # 从0.90降到0.85
    def row_V(row):
        x = np.array([row['pleasure'],row['arousal'],row['dominance']], dtype=float)
        P = P_map[row['personality_type']]
        return float(x @ P @ x)
    tmp = df_part.copy()
    tmp["V_personal"] = tmp.apply(row_V, axis=1)
    quant = {}
    for mbti in MBTI_TYPES:
        arr = tmp.loc[tmp['personality_type']==mbti, "V_personal"].values
        quant[mbti] = float(np.percentile(arr, int(q*100))) if len(arr)>0 else np.nan
    return quant

# =========================
# Main
# =========================
def main():
    # Load
    df, sess_key = load_dataset()
    print(f"Rows={len(df)}, session_key={sess_key}, users={df['user_id'].nunique()}, MBTI covered={df['personality_type'].nunique()}")

    # Split
    df_tr, df_va, df_te = split_by_session(df, sess_key)
    print(f"Split sessions -> train={df_tr[sess_key].nunique()}, val={df_va[sess_key].nunique()}, test={df_te[sess_key].nunique()}")

    # Estimate A_k & P_k on train
    A_map, P_map = estimate_Ak_Pk(df_tr)

    # 训练集上预计算 MBTI 85% 分位（更敏感的稳定rho基准）
    quant_mbti = compute_quantile_by_mbti(df_tr, P_map, q=0.85)

    # Build labels
    y_tr, y_va, y_te = build_labels(df_tr, df_va, df_te, P_map, sess_key)
    print(f"Label positives (train/val/test): {int(y_tr.sum())}/{int(y_va.sum())}/{int(y_te.sum())}  "
          f"(N={len(y_tr)}/{len(y_va)}/{len(y_te)})")

    # Strategy 1: Enhanced Ours (val search with improved scoring)
    best_ours = {"score": -1, "k": None, "h": None, "eps": None}
    print("Searching enhanced Ours parameters...")
    for k in OURS_K_GRID:
        for h in OURS_H_GRID:
            for eps in OURS_EPS_GRID:
                y_pred_va = predict_ours(
                    df_va, P_map, sess_key,
                    k_scale=k, h=h, eps=eps, alpha_margin=OURS_ALPHA_MARGIN,
                    quant_mbti=quant_mbti, beta_gate=OURS_BETA_GATE,
                    ema_alpha=OURS_EMA_ALPHA, rho_mode="mbti_quantile"
                )
                met = metrics_from_preds(y_va, y_pred_va)
                score = combined_score(met["Precision"], met["Recall"], met["F1"], 
                                     met["Switch Rate"], met["False Switch Rate"])
                if score > best_ours["score"]:
                    best_ours = {"score": score, "k": k, "h": h, "eps": eps, **met}
    
    print("Enhanced Ours best (val):", {k: v for k, v in best_ours.items() if k != "score"})

    # 用最优 enhanced ours 参数在测试集预测
    y_pred_te_ours = predict_ours(
        df_te, P_map, sess_key,
        k_scale=best_ours["k"], h=best_ours["h"], eps=best_ours["eps"], alpha_margin=OURS_ALPHA_MARGIN,
        quant_mbti=quant_mbti, beta_gate=OURS_BETA_GATE, ema_alpha=OURS_EMA_ALPHA, rho_mode="mbti_quantile"
    )

    # Strategy 2: Lyapunov-Fixed
    y_pred_te_fixed, params_fixed = predict_lyapunov_fixed(df_tr, df_va, df_te, sess_key, y_tr, y_va)
    print("Lyapunov-Fixed params:", params_fixed)

    # Strategy 3: Rule-Based
    y_pred_te_rule, params_rule = predict_rule_based(df_tr, df_va, df_te, sess_key, y_tr, y_va)
    print("Rule-Based params:", params_rule)

    # Strategy 4: No-Switching
    y_pred_te_nosw = np.zeros(len(df_te), dtype=int)

    # Evaluate on test
    res_ours  = metrics_from_preds(y_te, y_pred_te_ours)
    res_fixed = metrics_from_preds(y_te, y_pred_te_fixed)
    res_rule  = metrics_from_preds(y_te, y_pred_te_rule)
    res_nosw  = metrics_from_preds(y_te, y_pred_te_nosw)

    summary = pd.DataFrame([
        {"Strategy":"Enhanced Lyapunov-Personalized (Ours)", **res_ours},
        {"Strategy":"Lyapunov-Fixed",                        **res_fixed},
        {"Strategy":"Rule-Based Thresholding",               **res_rule},
        {"Strategy":"No-Switching Baseline",                 **res_nosw},
    ], columns=["Strategy","Precision","Recall","F1","Switch Rate","False Switch Rate","TP","FP","FN","TN","N"])
    
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    print("\n=== Enhanced Test Summary ===")
    print(summary[["Strategy","Precision","Recall","F1","Switch Rate","False Switch Rate"]])

    # Save hyperparams
    hparams = {
        "Enhanced_Ours": {
            "k_scale": best_ours["k"], 
            "h": best_ours["h"], 
            "eps": best_ours["eps"], 
            "alpha_margin": OURS_ALPHA_MARGIN,
            "beta_gate": OURS_BETA_GATE,
            "ema_alpha": OURS_EMA_ALPHA
        },
        "Lyapunov-Fixed": params_fixed,
        "Rule-Based": params_rule,
        "splits": {"train_sessions": int(df_tr[sess_key].nunique()),
                   "val_sessions": int(df_va[sess_key].nunique()),
                   "test_sessions": int(df_te[sess_key].nunique())},
        "enhancements": {
            "label_v_percentile": LABEL_V_PERCENTILE,
            "label_ldr_eps": LABEL_LDR_EPS,
            "multi_trigger_conditions": 7,
            "dynamic_threshold": True,
            "risk_scoring": True
        }
    }
    with open(OUTPUT_HPARAMS, "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    # Optional detail
    if OUTPUT_DETAIL:
        det = df_te.copy().reset_index(drop=True)
        det["y_true"] = y_te
        det["pred_enhanced_ours"] = y_pred_te_ours
        det["pred_fixed"] = y_pred_te_fixed
        det["pred_rule"] = y_pred_te_rule
        det["pred_nosw"] = y_pred_te_nosw
        try:
            det.to_parquet(OUTPUT_DETAIL, index=False)
        except Exception:
            det.to_csv(OUTPUT_DETAIL.replace(".parquet",".csv"), index=False)

    print(f"\nSaved enhanced summary to: {OUTPUT_SUMMARY}")
    print(f"Saved enhanced hyperparams to: {OUTPUT_HPARAMS}")
    if OUTPUT_DETAIL:
        print(f"Saved detail to: {OUTPUT_DETAIL}")
    


# Entry
if __name__ == "__main__":

    main()
