import pandas as pd
import numpy as np

# 加载原始数据集
df = pd.read_csv("/mnt/data/fixed_emotion_dataset_20250526_205726.csv")

# 清洗和筛选有效数据（只保留用户说话轮次，且pleasure, arousal, dominance为数值型）
df = df[df['speaker'] == 'user'].copy()
df = df.dropna(subset=['pleasure', 'arousal', 'dominance'])
df[['pleasure', 'arousal', 'dominance']] = df[['pleasure', 'arousal', 'dominance']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['pleasure', 'arousal', 'dominance'])

# 提取用户前5轮对话数据用于A拟合
A_records = []
for user_id, group in df.groupby('user_id'):
    group_sorted = group.sort_values('turn')
    if len(group_sorted) >= 6:
        X = group_sorted[['pleasure', 'arousal', 'dominance']].values[:5].T
        Y = group_sorted[['pleasure', 'arousal', 'dominance']].values[1:6].T
        A = Y @ X.T @ np.linalg.inv(X @ X.T)
        latest_state = group_sorted[['pleasure', 'arousal', 'dominance']].values[5]
        A_records.append((user_id, A, latest_state, group_sorted.iloc[5]))

# 生成Updated_Lyapunov_Switch_Dataset.csv
updated_rows = []
Q = 0.001 * np.identity(3)

for user_id, A_i, x_t, row in A_records:
    P = np.linalg.solve(np.eye(9) - np.kron(A_i.T @ A_i, np.eye(3)), Q.flatten()).reshape(3, 3)
    V_xt = x_t.T @ P @ x_t
    threshold = np.mean([r[2].T @ P @ r[2] for r in A_records])  # 自适应threshold：所有V(x_t)的平均
    should_switch = V_xt > threshold
    updated_rows.append({
        'user_id': user_id,
        'turn': row['turn'],
        'V_xt': V_xt,
        'threshold': threshold,
        'should_switch': should_switch,
        'personality_type': row['personality_type'],
        'pleasure': x_t[0],
        'arousal': x_t[1],
        'dominance': x_t[2],
        'agent_type': row['agent_type'],
        'text': row['text'],
        'emotion_label': row['emotion_label'],
        'scenario': row.get('scenario', '')
    })

updated_df = pd.DataFrame(updated_rows)
updated_with_speaker = updated_df.copy()
updated_with_speaker['speaker'] = 'user'

# 保存两个版本的数据
updated_df.to_csv("/mnt/data/Updated_Lyapunov_Switch_Dataset.csv", index=False)
updated_with_speaker.to_csv("/mnt/data/Updated_Lyapunov_Switch_Dataset_with_Speaker.csv", index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Lyapunov Switch Dataset", dataframe=updated_df)


# 加载原始的 Updated_Lyapunov_Switch_Dataset.csv 数据
original_df = pd.read_csv('/mnt/data/Updated_Lyapunov_Switch_Dataset.csv')

# 添加 speaker 字段，根据 agent_type 判断是 user 还是 agent
def infer_speaker(row):
    if row['agent_type'] == 'AI' or row['agent_type'] == 'Human':
        return 'user'
    elif row['agent_type'] == 'Switched':
        return 'agent'
    else:
        return 'unknown'

original_df['speaker'] = original_df.apply(infer_speaker, axis=1)

# 调整列顺序以符合要求
column_order = [
    'user_id', 'personality_type', 'turn', 'agent_type', 'speaker',
    'emotion_label', 'text', 'pleasure', 'arousal', 'dominance',
    'scenario', 'V_xt', 'threshold', 'should_switch'
]

# 有些列可能在原始数据中不存在，这里先确认并剔除不存在的
column_order = [col for col in column_order if col in original_df.columns]

# 生成新数据集
updated_df = original_df[column_order]

# 保存新的包含 speaker 的数据集
output_path = '/mnt/data/Updated_Lyapunov_Switch_Dataset_with_Speaker.csv'
updated_df.to_csv(output_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Lyapunov Dataset with Speaker", dataframe=updated_df)

output_path
