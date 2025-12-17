import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def solve_discrete_lyapunov(A, Q):
    """
    求解离散Lyapunov方程: A^T P A - P = -Q
    使用迭代方法求解
    """
    try:
        # 使用scipy求解Lyapunov方程
        from scipy.linalg import solve_discrete_lyapunov
        P = solve_discrete_lyapunov(A.T, Q)
        return P
    except:
        # 如果scipy不可用，使用迭代方法
        n = A.shape[0]
        P = np.eye(n)
        for _ in range(100):  # 最大迭代次数
            P_new = Q + A.T @ P @ A
            if np.linalg.norm(P_new - P) < 1e-6:
                break
            P = P_new
        return P

def compute_ldr(V_history, window=5):
    """
    计算Lyapunov衰减率 (LDR)
    η̂_t(h) = (1/h) × Σ[1 - V(x_{t-k+1})/V(x_{t-k})]
    """
    if len(V_history) < 2:
        return 0.0
    
    window = min(window, len(V_history) - 1)
    decay_rates = []
    
    for k in range(window):
        if len(V_history) > k + 1:
            V_prev = V_history[-(k+2)]  # V(x_{t-k})
            V_curr = V_history[-(k+1)]  # V(x_{t-k+1})
            if V_prev > 1e-8:  # 避免除零
                decay_rate = 1 - V_curr / V_prev
                decay_rates.append(decay_rate)
    
    return np.mean(decay_rates) if decay_rates else 0.0

def fit_user_models(df):
    """
    为每个用户拟合个性化的A矩阵和P矩阵
    """
    user_models = {}
    Q = 0.001 * np.identity(3)  # 论文中使用的Q矩阵
    
    print("拟合用户情绪动态模型...")
    for user_id, group in df.groupby('user_id'):
        group_sorted = group.sort_values('turn')
        if len(group_sorted) >= 6:  # 至少需要6个数据点
            # 提取PAD轨迹
            X = group_sorted[['pleasure', 'arousal', 'dominance']].values[:5].T  # [3x5]
            Y = group_sorted[['pleasure', 'arousal', 'dominance']].values[1:6].T  # [3x5]
            
            try:
                # 系统辨识: A = Y X^T (XX^T)^(-1)
                A = Y @ X.T @ np.linalg.inv(X @ X.T)
                
                # 检查稳定性 (所有特征值模长 < 1)
                eigenvals = np.linalg.eigvals(A)
                if np.all(np.abs(eigenvals) < 0.99):  # 确保Schur稳定
                    # 求解Lyapunov方程: A^T P A - P = -Q
                    P = solve_discrete_lyapunov(A, Q)
                    
                    # 检查P是否正定
                    if np.all(np.linalg.eigvals(P) > 0):
                        # 计算初始ρ值
                        x0 = X[:, 0]  # 初始状态
                        rho = x0.T @ P @ x0
                        
                        user_models[user_id] = {
                            'A': A,
                            'P': P,
                            'rho': rho,
                            'x0': x0
                        }
            except (np.linalg.LinAlgError, ValueError):
                continue
    
    print(f"成功为 {len(user_models)} 个用户建立模型")
    return user_models

def compute_average_P(user_models):
    """
    计算所有用户P矩阵的平均值，用于Lyapunov-Fixed策略
    """
    P_matrices = [model['P'] for model in user_models.values()]
    if P_matrices:
        return np.mean(P_matrices, axis=0)
    else:
        return np.eye(3)  # 默认单位矩阵

def strategy_lyapunov_personalized(test_df, user_models, epsilon=-0.1, window=5):
    """
    Lyapunov-Personalized策略 (Our Method)
    双触发机制：椭圆边界 + LDR检测
    """
    predictions = []
    V_histories = {}  # 为每个用户维护V历史
    latch_states = {}  # 为每个用户维护latch状态
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        xt = np.array([row['pleasure'], row['arousal'], row['dominance']])
        
        if user_id in user_models:
            model = user_models[user_id]
            
            # 初始化用户状态
            if user_id not in V_histories:
                V_histories[user_id] = []
                latch_states[user_id] = False
            
            # 计算Lyapunov能量
            V_xt = xt.T @ model['P'] @ xt
            V_histories[user_id].append(V_xt)
            
            # 如果已经latch，保持切换状态
            if latch_states[user_id]:
                predictions.append(True)
                continue
            
            # 条件1：椭圆边界检测 (0.8倍安全余量)
            boundary_trigger = V_xt >= 0.8 * model['rho']
            
            # 条件2：LDR检测
            ldr = compute_ldr(V_histories[user_id], window)
            ldr_trigger = ldr < epsilon
            
            # 双触发机制
            should_switch = boundary_trigger or ldr_trigger
            
            # 更新latch状态
            if should_switch:
                latch_states[user_id] = True
            
            predictions.append(should_switch)
        else:
            predictions.append(False)  # 无模型的用户不切换
    
    return predictions

def strategy_lyapunov_fixed(test_df, P_avg, rho_fixed=3.0):
    """
    Lyapunov-Fixed策略
    使用固定阈值，无个性化，无LDR
    """
    predictions = []
    latch_states = {}
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        xt = np.array([row['pleasure'], row['arousal'], row['dominance']])
        
        # 初始化latch状态
        if user_id not in latch_states:
            latch_states[user_id] = False
        
        # 如果已经latch，保持切换状态
        if latch_states[user_id]:
            predictions.append(True)
            continue
        
        # 计算Lyapunov能量（使用平均P矩阵）
        V_xt = xt.T @ P_avg @ xt
        
        # 单一触发条件：固定阈值
        should_switch = V_xt >= rho_fixed
        
        # 更新latch状态
        if should_switch:
            latch_states[user_id] = True
        
        predictions.append(should_switch)
    
    return predictions

def strategy_rule_based(test_df, pleasure_threshold=0.3, arousal_threshold=0.7):
    """
    Rule-Based Thresholding策略
    基于PAD维度的简单规则
    """
    predictions = []
    latch_states = {}
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        
        # 初始化latch状态
        if user_id not in latch_states:
            latch_states[user_id] = False
        
        # 如果已经latch，保持切换状态
        if latch_states[user_id]:
            predictions.append(True)
            continue
        
        # 简单规则：低愉悦度或高唤醒度
        should_switch = (row['pleasure'] < pleasure_threshold) or (row['arousal'] > arousal_threshold)
        
        # 更新latch状态
        if should_switch:
            latch_states[user_id] = True
        
        predictions.append(should_switch)
    
    return predictions

def strategy_no_switching(test_df):
    """
    No-Switching Baseline策略
    永远不切换
    """
    return [False] * len(test_df)

def compute_metrics(y_true, y_pred):
    """
    计算评估指标
    """
    # 转换为numpy数组
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # 基本指标
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 切换率
    switch_rate = np.mean(y_pred)
    
    # 误切换率：预测切换但不应该切换的比例
    false_switches = np.sum((y_pred == 1) & (y_true == 0))
    false_switch_rate = false_switches / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Switch Rate": switch_rate,
        "False Switch Rate": false_switch_rate
    }

def main():
    print("=" * 60)
    print("Lyapunov-Based Emotion-Aware Switching Comparison")
    print("=" * 60)
    
    # 1. 读取数据
    print("\n1. 加载数据集...")
    try:
        fixed_df = pd.read_csv("fixed_emotion_dataset_20250526_205726.csv")
        test_df = pd.read_csv("Updated_Lyapunov_Dataset_with_Speaker.csv")
        print(f"训练数据: {len(fixed_df)} 条记录")
        print(f"测试数据: {len(test_df)} 条记录")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    # 只保留用户的记录
    fixed_df = fixed_df[fixed_df['speaker'] == 'user'].copy()
    
    # 清理PAD数据
    pad_columns = ['pleasure', 'arousal', 'dominance']
    fixed_df = fixed_df.dropna(subset=pad_columns)
    fixed_df[pad_columns] = fixed_df[pad_columns].apply(pd.to_numeric, errors='coerce')
    fixed_df = fixed_df.dropna(subset=pad_columns)
    
    print(f"清理后训练数据: {len(fixed_df)} 条记录")
    print(f"涉及用户数: {fixed_df['user_id'].nunique()}")
    
    # 3. 拟合用户模型
    print("\n3. 拟合用户情绪动态模型...")
    user_models = fit_user_models(fixed_df)
    
    if not user_models:
        print("错误：无法为任何用户建立有效模型")
        return
    
    # 4. 计算平均P矩阵（用于Fixed策略）
    P_avg = compute_average_P(user_models)
    print(f"平均P矩阵形状: {P_avg.shape}")
    
    # 5. 准备测试数据
    print("\n4. 准备测试数据...")
    # 确保测试数据包含ground truth
    if 'should_switch' not in test_df.columns:
        print("错误：测试数据缺少 'should_switch' 标签")
        return
    
    # 只保留有模型的用户的测试数据
    test_df_filtered = test_df[test_df['user_id'].isin(user_models.keys())].copy()
    print(f"可测试数据: {len(test_df_filtered)} 条记录")
    
    if len(test_df_filtered) == 0:
        print("错误：没有可用的测试数据")
        return
    
    # 6. 执行四种策略
    print("\n5. 执行四种切换策略...")
    
    # Ground truth
    oracle_labels = test_df_filtered['should_switch'].values
    
    # 策略1: Lyapunov-Personalized (Our Method)
    print("   - Lyapunov-Personalized (Our Method)")
    pred_personalized = strategy_lyapunov_personalized(test_df_filtered, user_models)
    
    # 策略2: Lyapunov-Fixed
    print("   - Lyapunov-Fixed")
    pred_fixed = strategy_lyapunov_fixed(test_df_filtered, P_avg, rho_fixed=3.0)
    
    # 策略3: Rule-Based Thresholding
    print("   - Rule-Based Thresholding")
    pred_rule_based = strategy_rule_based(test_df_filtered)
    
    # 策略4: No-Switching Baseline
    print("   - No-Switching Baseline")
    pred_no_switching = strategy_no_switching(test_df_filtered)
    
    # 7. 计算指标
    print("\n6. 计算性能指标...")
    
    results = {
        "Lyapunov-Personalized (Ours)": compute_metrics(oracle_labels, pred_personalized),
        "Lyapunov-Fixed": compute_metrics(oracle_labels, pred_fixed),
        "Rule-Based Thresholding": compute_metrics(oracle_labels, pred_rule_based),
        "No-Switching Baseline": compute_metrics(oracle_labels, pred_no_switching)
    }
    
    # 8. 输出对比结果
    print("\n" + "=" * 80)
    print("对比试验结果")
    print("=" * 80)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(3)
    
    print(results_df.to_string())
    
    # 9. 详细分析
    print("\n" + "=" * 80)
    print("详细分析")
    print("=" * 80)
    
    print(f"\n数据统计:")
    print(f"- 测试样本总数: {len(test_df_filtered)}")
    print(f"- 应该切换的样本数: {np.sum(oracle_labels)}")
    print(f"- 切换率 (Ground Truth): {np.mean(oracle_labels):.3f}")
    
    print(f"\n模型统计:")
    print(f"- 成功建模用户数: {len(user_models)}")
    print(f"- 平均P矩阵特征值: {np.linalg.eigvals(P_avg)}")
    
    # 找出最佳策略
    best_strategy = results_df['F1'].idxmax()
    best_f1 = results_df.loc[best_strategy, 'F1']
    
    print(f"\n最佳策略: {best_strategy}")
    print(f"最佳F1得分: {best_f1:.3f}")
    
    return results_df

if __name__ == "__main__":
    results = main()