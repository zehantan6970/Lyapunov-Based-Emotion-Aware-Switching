import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV数据
df = pd.read_csv('/mnt/data/Updated_Lyapunov_Switch_Dataset_with_Speaker.csv')

# 判断是否需要生成更多数据
data_count = len(df)

# 统计基本信息
summary_stats = {
    "Total Samples": data_count,
    "Unique Users": df['user_id'].nunique(),
    "Unique Personality Types": df['personality_type'].nunique(),
    "Switch Instances": df['should_switch'].sum(),
    "Switch Rate": df['should_switch'].mean()
}

import ace_tools as tools; tools.display_dataframe_to_user(name="Dataset Summary", dataframe=pd.DataFrame([summary_stats]))

# 如果样本少于100条，建议生成更多数据
need_more_data = data_count < 100


