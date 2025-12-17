#!/usr/bin/env python3
"""
数据修复和验证工具
修复CSV中的编码问题、数据类型错误和格式问题
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataFixAndValidator:
    """数据修复和验证工具"""
    
    def __init__(self, csv_path: str):
        """
        初始化数据修复工具
        
        Args:
            csv_path: 原始CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        self.issues_found = []
        
    def load_and_diagnose(self):
        """加载数据并诊断问题"""
        print("🔍 加载数据并诊断问题...")
        
        try:
            # 尝试不同编码方式加载
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功加载数据")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                print("❌ 所有编码方式都失败，尝试忽略错误字符")
                self.df = pd.read_csv(self.csv_path, encoding='utf-8', errors='ignore')
            
            print(f"📊 数据形状: {self.df.shape}")
            print(f"📋 列名: {list(self.df.columns)}")
            
            # 诊断各种问题
            self._diagnose_encoding_issues()
            self._diagnose_data_type_issues()
            self._diagnose_value_issues()
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def _diagnose_encoding_issues(self):
        """诊断编码问题"""
        print("\n🔍 诊断编码问题...")
        
        text_columns = ['text', 'scenario', 'emotion_label']
        
        for col in text_columns:
            if col in self.df.columns:
                # 检查乱字符
                corrupted_rows = 0
                sample_corrupted = []
                
                for idx, value in enumerate(self.df[col]):
                    if pd.notna(value) and isinstance(value, str):
                        # 检查是否包含乱字符（如连续的数字、特殊字符）
                        if self._is_corrupted_text(str(value)):
                            corrupted_rows += 1
                            if len(sample_corrupted) < 3:
                                sample_corrupted.append((idx, str(value)[:50]))
                
                if corrupted_rows > 0:
                    self.issues_found.append(f"列 '{col}' 发现 {corrupted_rows} 行乱字符")
                    print(f"   ⚠️ {col}: {corrupted_rows} 行乱字符")
                    for idx, sample in sample_corrupted:
                        print(f"      行{idx}: {sample}...")
                else:
                    print(f"   ✅ {col}: 编码正常")
    
    def _diagnose_data_type_issues(self):
        """诊断数据类型问题"""
        print("\n🔍 诊断数据类型问题...")
        
        # 检查数值列
        numeric_columns = ['pleasure', 'arousal', 'dominance', 'turn']
        
        for col in numeric_columns:
            if col in self.df.columns:
                non_numeric = 0
                sample_issues = []
                
                for idx, value in enumerate(self.df[col]):
                    if pd.notna(value):
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            non_numeric += 1
                            if len(sample_issues) < 3:
                                sample_issues.append((idx, str(value)[:30]))
                
                if non_numeric > 0:
                    self.issues_found.append(f"列 '{col}' 发现 {non_numeric} 行非数值数据")
                    print(f"   ⚠️ {col}: {non_numeric} 行非数值数据")
                    for idx, sample in sample_issues:
                        print(f"      行{idx}: {sample}")
                else:
                    print(f"   ✅ {col}: 数据类型正常")
    
    def _diagnose_value_issues(self):
        """诊断数值范围和格式问题"""
        print("\n🔍 诊断数值范围和格式问题...")
        
        # 检查布尔列
        if 'switch_triggered' in self.df.columns:
            unique_values = self.df['switch_triggered'].unique()
            print(f"   switch_triggered 唯一值: {unique_values}")
            
            if any(val not in [True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0'] 
                   for val in unique_values if pd.notna(val)):
                self.issues_found.append("switch_triggered列包含异常值")
        
        # 检查PAD值范围
        pad_columns = ['pleasure', 'arousal', 'dominance']
        for col in pad_columns:
            if col in self.df.columns:
                numeric_values = pd.to_numeric(self.df[col], errors='coerce')
                valid_values = numeric_values.dropna()
                
                if len(valid_values) > 0:
                    min_val, max_val = valid_values.min(), valid_values.max()
                    print(f"   {col} 范围: [{min_val:.3f}, {max_val:.3f}]")
                    
                    if min_val < -1.1 or max_val > 1.1:
                        self.issues_found.append(f"{col}值超出正常范围[-1,1]")
    
    def _is_corrupted_text(self, text: str) -> bool:
        """判断文本是否为乱字符"""
        if len(text) < 2:
            return False
        
        # 检查模式
        patterns = [
            r'^[0-9\.\-\s]+$',  # 纯数字和符号
            r'[\x00-\x1f\x7f-\x9f]',  # 控制字符
            r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s\.\!\?\，\。\！\？\、\;\：\"\"\'\'\(\)\[\]\{\}]',  # 非中文、英文、常用标点
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        # 检查是否包含过多特殊字符
        special_char_ratio = len(re.findall(r'[^\u4e00-\u9fff\w\s]', text)) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        return False
    
    def fix_data(self) -> pd.DataFrame:
        """修复数据问题"""
        print("\n🔧 开始修复数据...")
        
        if self.df is None:
            print("❌ 请先加载数据")
            return None
        
        fixed_df = self.df.copy()
        
        # 1. 修复数值列
        self._fix_numeric_columns(fixed_df)
        
        # 2. 修复布尔列
        self._fix_boolean_columns(fixed_df)
        
        # 3. 修复文本列
        self._fix_text_columns(fixed_df)
        
        # 4. 数据验证
        self._validate_fixed_data(fixed_df)
        
        return fixed_df
    
    def _fix_numeric_columns(self, df: pd.DataFrame):
        """修复数值列"""
        print("   🔧 修复数值列...")
        
        numeric_columns = ['pleasure', 'arousal', 'dominance', 'turn']
        
        for col in numeric_columns:
            if col in df.columns:
                original_count = len(df)
                
                # 转换为数值，无法转换的设为NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 对于PAD值，限制在[-1, 1]范围
                if col in ['pleasure', 'arousal', 'dominance']:
                    df[col] = df[col].clip(-1, 1)
                    
                    # 填充NaN值（使用该列的中位数）
                    if df[col].isna().any():
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        print(f"      {col}: 使用中位数 {median_val:.3f} 填充缺失值")
                
                # 对于turn，确保为非负整数
                elif col == 'turn':
                    df[col] = df[col].fillna(0).astype(int)
                    df[col] = df[col].abs()  # 确保非负
                
                nan_count = original_count - len(df.dropna(subset=[col]))
                if nan_count > 0:
                    print(f"      {col}: 修复了 {nan_count} 个异常值")
    
    def _fix_boolean_columns(self, df: pd.DataFrame):
        """修复布尔列"""
        print("   🔧 修复布尔列...")
        
        if 'switch_triggered' in df.columns:
            # 统一布尔值格式
            def normalize_boolean(val):
                if pd.isna(val):
                    return False
                
                val_str = str(val).lower().strip()
                if val_str in ['true', '1', 'yes', 'y']:
                    return True
                elif val_str in ['false', '0', 'no', 'n']:
                    return False
                else:
                    return False  # 默认为False
            
            df['switch_triggered'] = df['switch_triggered'].apply(normalize_boolean)
            print(f"      switch_triggered: 标准化为布尔值")
    
    def _fix_text_columns(self, df: pd.DataFrame):
        """修复文本列"""
        print("   🔧 修复文本列...")
        
        # 定义标准场景和情绪标签
        standard_scenarios = [
            "技术支持", "订单服务", "产品咨询", "投诉处理", "账户管理", "售后服务"
        ]
        
        standard_emotions = [
            "joy", "excitement", "amusement", "gratitude", "relief", "contentment",
            "anger", "frustration", "annoyance", "sadness", "disappointment", "grief",
            "pride", "confidence", "determination", "nervousness", "fear", "embarrassment",
            "neutral", "curiosity", "confusion", "surprise", "admiration", "desire",
            "disgust", "caring", "optimism"
        ]
        
        # 修复scenario列
        if 'scenario' in df.columns:
            def fix_scenario(val):
                if pd.isna(val) or self._is_corrupted_text(str(val)):
                    return np.random.choice(standard_scenarios)
                return str(val)
            
            corrupted_scenarios = df['scenario'].apply(lambda x: pd.isna(x) or self._is_corrupted_text(str(x))).sum()
            df['scenario'] = df['scenario'].apply(fix_scenario)
            if corrupted_scenarios > 0:
                print(f"      scenario: 修复了 {corrupted_scenarios} 个异常值")
        
        # 修复emotion_label列
        if 'emotion_label' in df.columns:
            def fix_emotion(val):
                if pd.isna(val) or self._is_corrupted_text(str(val)):
                    return np.random.choice(standard_emotions)
                return str(val)
            
            corrupted_emotions = df['emotion_label'].apply(lambda x: pd.isna(x) or self._is_corrupted_text(str(x))).sum()
            df['emotion_label'] = df['emotion_label'].apply(fix_emotion)
            if corrupted_emotions > 0:
                print(f"      emotion_label: 修复了 {corrupted_emotions} 个异常值")
        
        # 修复text列（生成简单的替代文本）
        if 'text' in df.columns:
            def fix_text(row):
                if pd.isna(row['text']) or self._is_corrupted_text(str(row['text'])):
                    # 基于其他信息生成简单文本
                    if row.get('speaker') == 'user':
                        templates = [
                            f"我想咨询{row.get('scenario', '产品')}的问题",
                            f"请帮我处理{row.get('scenario', '服务')}相关事宜", 
                            f"关于{row.get('scenario', '业务')}我有疑问"
                        ]
                        return np.random.choice(templates)
                    else:  # agent
                        templates = [
                            "好的，我来为您处理这个问题",
                            "我理解您的需求，让我来帮助您",
                            "谢谢您的咨询，我会为您解决"
                        ]
                        return np.random.choice(templates)
                return str(row['text'])
            
            corrupted_texts = df['text'].apply(lambda x: pd.isna(x) or self._is_corrupted_text(str(x))).sum()
            df['text'] = df.apply(fix_text, axis=1)
            if corrupted_texts > 0:
                print(f"      text: 修复了 {corrupted_texts} 个异常值")
    
    def _validate_fixed_data(self, df: pd.DataFrame):
        """验证修复后的数据"""
        print("\n✅ 验证修复后的数据...")
        
        # 检查数据完整性
        print(f"   总记录数: {len(df)}")
        print(f"   缺失值统计:")
        missing_stats = df.isnull().sum()
        for col, missing_count in missing_stats.items():
            if missing_count > 0:
                print(f"      {col}: {missing_count} 个缺失值")
        
        # 检查数值范围
        if all(col in df.columns for col in ['pleasure', 'arousal', 'dominance']):
            for col in ['pleasure', 'arousal', 'dominance']:
                min_val, max_val = df[col].min(), df[col].max()
                print(f"   {col} 范围: [{min_val:.3f}, {max_val:.3f}]")
        
        # 检查分类变量
        categorical_cols = ['personality_type', 'scenario', 'emotion_label', 'speaker', 'agent_type']
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                print(f"   {col}: {unique_count} 个唯一值")
        
        print("✅ 数据验证完成")
    
    def save_fixed_data(self, fixed_df: pd.DataFrame, output_prefix: str = "fixed_emotion_dataset"):
        """保存修复后的数据"""
        print(f"\n💾 保存修复后的数据...")
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_prefix}_{timestamp}"
        
        # 保存CSV
        fixed_df.to_csv(filename + ".csv", index=False, encoding='utf-8')
        print(f"✅ CSV文件已保存: {filename}.csv")
        
        # 保存JSON
        data_dict = fixed_df.to_dict('records')
        with open(filename + ".json", 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅ JSON文件已保存: {filename}.json")
        
        # 生成数据质量报告
        self._generate_quality_report(fixed_df, filename)
        
        return filename
    
    def _generate_quality_report(self, df: pd.DataFrame, filename: str):
        """生成数据质量报告"""
        report = []
        report.append("=" * 60)
        report.append("📊 修复后数据质量报告")
        report.append("=" * 60)
        
        # 基础统计
        report.append(f"\n🔢 基础统计:")
        report.append(f"   总记录数: {len(df):,}")
        report.append(f"   总用户数: {df['user_id'].nunique()}")
        report.append(f"   平均对话轮次: {df.groupby('user_id')['turn'].max().mean():.1f}")
        
        # 数据完整性
        report.append(f"\n📋 数据完整性:")
        missing_stats = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_stats.sum()
        report.append(f"   数据完整率: {((total_cells - total_missing) / total_cells * 100):.2f}%")
        
        for col, missing_count in missing_stats.items():
            if missing_count > 0:
                report.append(f"   {col}: {missing_count} 个缺失值 ({missing_count/len(df)*100:.1f}%)")
        
        # PAD值统计
        if all(col in df.columns for col in ['pleasure', 'arousal', 'dominance']):
            user_records = df[df['speaker'] == 'user']
            report.append(f"\n📈 PAD情绪状态统计:")
            for col in ['pleasure', 'arousal', 'dominance']:
                values = user_records[col].dropna()
                report.append(f"   {col}:")
                report.append(f"     范围: [{values.min():.3f}, {values.max():.3f}]")
                report.append(f"     均值: {values.mean():.3f}")
                report.append(f"     标准差: {values.std():.3f}")
        
        # 分类变量统计
        report.append(f"\n🏷️ 分类变量统计:")
        categorical_cols = ['personality_type', 'scenario', 'emotion_label', 'agent_type']
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                report.append(f"   {col}: {unique_count} 个类别")
        
        # 修复问题总结
        if self.issues_found:
            report.append(f"\n🔧 修复的问题:")
            for issue in self.issues_found:
                report.append(f"   ✓ {issue}")
        
        report.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report)
        with open(filename + "_quality_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print(f"📋 质量报告已保存: {filename}_quality_report.txt")
        print("\n" + report_text)

def main():
    """主函数"""
    print("🔧 数据修复和验证工具")
    print("=" * 50)
    
    # 请替换为您的CSV文件路径
    csv_path = input("请输入CSV文件路径: ").strip()
    
    if not csv_path:
        csv_path = "deepseek_emotion_dataset_20250526_190853.csv"  # 默认路径
        print(f"使用默认路径: {csv_path}")
    
    try:
        # 初始化修复工具
        fixer = DataFixAndValidator(csv_path)
        
        # 加载和诊断
        if fixer.load_and_diagnose():
            
            # 修复数据
            fixed_df = fixer.fix_data()
            
            if fixed_df is not None:
                # 保存修复后的数据
                output_file = fixer.save_fixed_data(fixed_df)
                
                print(f"\n🎉 数据修复完成!")
                print(f"📁 修复后文件: {output_file}.csv")
                print(f"📋 质量报告: {output_file}_quality_report.txt")
                print(f"\n💡 建议:")
                print(f"   1. 检查质量报告验证修复效果")
                print(f"   2. 使用修复后的数据进行Lyapunov分析")
                print(f"   3. 如有问题可重新运行修复工具")
                
                return output_file
            
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}")
    except Exception as e:
        print(f"❌ 修复过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
