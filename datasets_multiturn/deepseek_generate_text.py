#!/usr/bin/env python3
"""
DeepSeek API完整情绪演化数据集生成器
生成160个用户，16种MBTI类型，每用户5-12轮对话
使用DeepSeek API确保低成本高质量对话生成   anaconda use base
"""

import numpy as np
import pandas as pd
import json
import random
import time
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import os
from datetime import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmotionState:
    """情绪状态数据结构"""
    pleasure: float
    arousal: float 
    dominance: float
    turn: int
    
@dataclass
class ConversationRecord:
    """对话记录数据结构"""
    user_id: str
    personality_type: str
    turn: int
    agent_type: str
    speaker: str
    emotion_label: str
    text: str
    pleasure: float
    arousal: float
    dominance: float
    scenario: str
    switch_triggered: bool

class DeepSeekEmotionDatasetGenerator:
    """DeepSeek API完整版情绪演化数据集生成器"""
    
    def __init__(self, deepseek_api_key: str, base_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        初始化生成器
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            base_url: DeepSeek API基础URL
        """
        self.deepseek_api_key = "sk-455130ce27c748209222e3a14a9fa3c5"
        self.deepseek_base_url = base_url
        self.mbti_matrices = self._define_all_mbti_matrices()
        self.mbti_initial_states = self._define_all_initial_states()
        self.scenarios = self._define_all_scenarios()
        self.goemotion_labels = self._define_complete_goemotion_labels()
        self.conversation_templates = self._define_conversation_templates()
        
        # 统计信息
        self.total_api_calls = 0
        self.failed_calls = 0
        self.generation_start_time = None
        self.api_lock = threading.Lock()
        
    def _define_all_mbti_matrices(self) -> Dict[str, np.ndarray]:
        """定义完整的16种MBTI类型情绪响应矩阵"""
        matrices = {
            # 外向直觉类型 (EN**)
            "ENFP": np.array([[0.85, 0.05, 0.05],  # 高愉悦持续性
                             [0.10, 0.90, 0.10],   # 高唤醒活跃性
                             [0.05, 0.05, 0.85]]), # 较高主导感
            
            "ENFJ": np.array([[0.80, 0.10, 0.15],  # 愉悦度受主导感影响
                             [0.15, 0.85, 0.05],   # 唤醒度受愉悦驱动
                             [0.10, 0.05, 0.80]]), # 主导感较强但温和
            
            "ENTP": np.array([[0.75, 0.15, 0.10],  # 愉悦度活跃
                             [0.20, 0.85, 0.15],   # 高唤醒交互性强
                             [0.15, 0.10, 0.75]]), # 主导感灵活
            
            "ENTJ": np.array([[0.70, 0.10, 0.20],  # 愉悦度受主导感影响
                             [0.10, 0.80, 0.20],   # 唤醒度与主导相关
                             [0.15, 0.15, 0.90]]), # 极强主导感
            
            # 内向直觉类型 (IN**)
            "INFP": np.array([[0.80, 0.05, 0.10],  # 愉悦度稳定
                             [0.05, 0.75, 0.05],   # 唤醒度较低
                             [0.05, 0.05, 0.70]]), # 主导感较弱
            
            "INFJ": np.array([[0.85, 0.05, 0.10],  # 高度稳定愉悦
                             [0.10, 0.80, 0.05],   # 唤醒度受愉悦影响
                             [0.05, 0.05, 0.75]]), # 中等主导感
            
            "INTP": np.array([[0.75, 0.10, 0.05],  # 愉悦度受唤醒影响
                             [0.15, 0.80, 0.10],   # 唤醒度与思考相关
                             [0.05, 0.10, 0.70]]), # 主导感较弱
            
            "INTJ": np.array([[0.80, 0.05, 0.15],  # 愉悦度受主导影响
                             [0.05, 0.75, 0.15],   # 唤醒度与主导关联
                             [0.10, 0.10, 0.85]]), # 强主导感
            
            # 外向感知类型 (ES**)
            "ESFP": np.array([[0.90, 0.15, 0.05],  # 极高愉悦度
                             [0.20, 0.85, 0.05],   # 高唤醒与愉悦关联
                             [0.05, 0.05, 0.70]]), # 主导感较弱
            
            "ESFJ": np.array([[0.85, 0.10, 0.15],  # 愉悦度受主导影响
                             [0.15, 0.80, 0.10],   # 唤醒度与愉悦相关
                             [0.10, 0.05, 0.80]]), # 主导感中等偏强
            
            "ESTP": np.array([[0.80, 0.20, 0.10],  # 愉悦度受唤醒驱动
                             [0.25, 0.90, 0.15],   # 极高唤醒
                             [0.10, 0.15, 0.75]]), # 主导感活跃
            
            "ESTJ": np.array([[0.75, 0.10, 0.25],  # 愉悦度受主导强烈影响
                             [0.10, 0.80, 0.25],   # 唤醒度与主导相关
                             [0.15, 0.20, 0.90]]), # 极强主导感
            
            # 内向感知类型 (IS**)
            "ISFP": np.array([[0.85, 0.05, 0.05],  # 愉悦度稳定
                             [0.05, 0.70, 0.05],   # 低唤醒平静
                             [0.05, 0.05, 0.65]]), # 主导感弱
            
            "ISFJ": np.array([[0.80, 0.05, 0.10],  # 愉悦度稳定
                             [0.10, 0.75, 0.05],   # 唤醒度中等
                             [0.05, 0.05, 0.75]]), # 主导感中等
            
            "ISTP": np.array([[0.75, 0.10, 0.05],  # 愉悦度受唤醒影响
                             [0.10, 0.80, 0.10],   # 唤醒度与技能相关
                             [0.05, 0.15, 0.75]]), # 主导感受唤醒影响
            
            "ISTJ": np.array([[0.90, 0.05, 0.05],  # 极高稳定性
                             [0.05, 0.85, 0.05],   # 稳定唤醒度
                             [0.05, 0.05, 0.90]])  # 极强主导感
        }
        return matrices
    
    def _define_all_initial_states(self) -> Dict[str, Dict[str, np.ndarray]]:
        """定义所有MBTI类型的初始情绪状态分布"""
        initial_states = {
            # 外向直觉类型 (EN**)
            "ENFP": {"mean": np.array([0.6, 0.7, 0.5]), "cov": 0.12},
            "ENFJ": {"mean": np.array([0.5, 0.6, 0.6]), "cov": 0.10},
            "ENTP": {"mean": np.array([0.5, 0.8, 0.6]), "cov": 0.13},
            "ENTJ": {"mean": np.array([0.4, 0.7, 0.8]), "cov": 0.11},
            
            # 内向直觉类型 (IN**)
            "INFP": {"mean": np.array([0.4, 0.3, 0.3]), "cov": 0.15},
            "INFJ": {"mean": np.array([0.3, 0.4, 0.4]), "cov": 0.12},
            "INTP": {"mean": np.array([0.2, 0.5, 0.3]), "cov": 0.14},
            "INTJ": {"mean": np.array([0.3, 0.4, 0.7]), "cov": 0.10},
            
            # 外向感知类型 (ES**)
            "ESFP": {"mean": np.array([0.8, 0.8, 0.4]), "cov": 0.11},
            "ESFJ": {"mean": np.array([0.6, 0.6, 0.5]), "cov": 0.09},
            "ESTP": {"mean": np.array([0.7, 0.9, 0.6]), "cov": 0.12},
            "ESTJ": {"mean": np.array([0.5, 0.7, 0.8]), "cov": 0.08},
            
            # 内向感知类型 (IS**)
            "ISFP": {"mean": np.array([0.5, 0.2, 0.2]), "cov": 0.16},
            "ISFJ": {"mean": np.array([0.4, 0.3, 0.4]), "cov": 0.11},
            "ISTP": {"mean": np.array([0.3, 0.4, 0.5]), "cov": 0.13},
            "ISTJ": {"mean": np.array([0.3, 0.2, 0.7]), "cov": 0.07}
        }
        return initial_states
    
    def _define_all_scenarios(self) -> List[Dict[str, str]]:
        """定义完整的对话场景"""
        return [
            {
                "name": "技术支持",
                "description": "软件故障、设备问题、操作指导、系统异常",
                "context": "用户遇到技术问题需要解决"
            },
            {
                "name": "订单服务", 
                "description": "退换货、物流查询、支付异常、订单修改",
                "context": "用户对订单有疑问或需要处理"
            },
            {
                "name": "产品咨询",
                "description": "功能介绍、价格对比、购买建议、规格说明",
                "context": "用户想了解产品信息"
            },
            {
                "name": "投诉处理",
                "description": "服务不满、产品质量、损失赔偿、态度问题",
                "context": "用户对服务或产品不满意"
            },
            {
                "name": "账户管理",
                "description": "密码重置、信息修改、权限申请、账户安全",
                "context": "用户需要管理个人账户"
            },
            {
                "name": "售后服务",
                "description": "保修查询、维修预约、使用培训、配件更换",
                "context": "用户需要售后支持"
            }
        ]
    
    def _define_complete_goemotion_labels(self) -> List[str]:
        """定义完整的27类GoEmotions标签"""
        return [
            # 正面情绪
            "joy", "excitement", "amusement", "gratitude", "relief", "contentment",
            "pride", "confidence", "determination", "optimism", "caring", "admiration",
            "desire", "love", "approval",
            
            # 负面情绪  
            "anger", "frustration", "annoyance", "sadness", "disappointment", "grief",
            "nervousness", "fear", "embarrassment", "disgust", "disapproval",
            
            # 中性情绪
            "neutral", "curiosity", "confusion", "surprise", "realization"
        ]
    
    def _define_conversation_templates(self) -> Dict[str, Dict[str, str]]:
        """定义MBTI类型的对话模板"""
        return {
            "ENFP": {
                "traits": "热情开朗，富有创意，善于表达，情绪丰富",
                "style": "语言生动活泼，喜欢用感叹号，表达直接"
            },
            "ENFJ": {
                "traits": "关心他人，善于沟通，有责任感，重视和谐",
                "style": "语言温暖体贴，关注客服感受，表达礼貌"
            },
            "ENTP": {
                "traits": "机智灵活，喜欢辩论，思维跳跃，创新求变",
                "style": "语言机智幽默，可能提出质疑，思路清晰"
            },
            "ENTJ": {
                "traits": "目标导向，果断高效，具有领导力，直接坦率",
                "style": "语言简练有力，直奔主题，要求明确"
            },
            "INFP": {
                "traits": "理想主义，情感丰富，内向敏感，重视价值观",
                "style": "语言含蓄温和，可能表达犹豫，情感细腻"
            },
            "INFJ": {
                "traits": "深思熟虑，富有洞察力，关心他人，追求意义",
                "style": "语言深刻有内涵，表达谨慎，关注深层需求"
            },
            "INTP": {
                "traits": "逻辑分析，理性客观，喜欢思考，追求真理",
                "style": "语言精确理性，可能分析问题原因，逻辑清晰"
            },
            "INTJ": {
                "traits": "独立自主，有远见，系统思考，追求完美",
                "style": "语言简洁有条理，关注效率，可能提出改进建议"
            },
            "ESFP": {
                "traits": "活泼友好，关注当下，善于互动，情绪外显",
                "style": "语言轻松愉快，情绪表达明显，互动性强"
            },
            "ESFJ": {
                "traits": "友善合作，关心他人，注重和谐，遵守规则",
                "style": "语言礼貌友好，表达感谢，关注服务质量"
            },
            "ESTP": {
                "traits": "行动导向，适应性强，实用主义，享受当下",
                "style": "语言直接实用，关注实际效果，语速较快"
            },
            "ESTJ": {
                "traits": "组织能力强，重视效率，遵守规则，目标明确",
                "style": "语言正式条理，要求明确时间表，关注结果"
            },
            "ISFP": {
                "traits": "温和友善，重视个人价值，避免冲突，灵活适应",
                "style": "语言轻柔委婉，可能表达不确定，避免直接冲突"
            },
            "ISFJ": {
                "traits": "细心负责，服务他人，注重细节，保守稳定",
                "style": "语言谨慎礼貌，关注细节，表达担忧或感谢"
            },
            "ISTP": {
                "traits": "实用主义，逻辑分析，独立行动，关注事实",
                "style": "语言简洁实用，关注技术细节，少情感表达"
            },
            "ISTJ": {
                "traits": "可靠稳重，遵守规则，注重细节，系统有序",
                "style": "语言正式规范，逻辑清晰，关注流程和规则"
            }
        }
    
    def map_pad_to_goemotion(self, P: float, A: float, D: float) -> str:
        """PAD值映射到GoEmotions标签（完整版）"""
        # 基于PAD三维空间的精确映射
        if P > 0.6 and A > 0.6 and D > 0.3:
            return random.choice(['joy', 'excitement', 'amusement'])
        elif P > 0.4 and A > 0.4 and D > 0.5:
            return random.choice(['confidence', 'pride', 'determination'])
        elif P > 0.3 and A < -0.2 and D > 0.2:
            return random.choice(['contentment', 'relief', 'gratitude'])
        elif P < -0.4 and A > 0.4 and D > 0.2:
            return random.choice(['anger', 'frustration', 'annoyance'])
        elif P < -0.4 and A > 0.2 and D < -0.2:
            return random.choice(['nervousness', 'fear', 'embarrassment'])
        elif P < -0.5 and A < 0 and D < 0:
            return random.choice(['sadness', 'disappointment', 'grief'])
        elif P > 0.2 and A > 0.3 and D < 0.2:
            return random.choice(['optimism', 'caring', 'admiration'])
        elif P < -0.2 and A < -0.3 and D < -0.3:
            return random.choice(['disgust', 'disapproval'])
        elif A > 0.5 and abs(P) < 0.3:
            return random.choice(['surprise', 'curiosity'])
        elif A < -0.4 and abs(P) < 0.2:
            return random.choice(['neutral', 'realization'])
        elif P > 0.1 and abs(A) < 0.3:
            return random.choice(['approval', 'love'])
        elif P < -0.1 and abs(A) < 0.3:
            return 'confusion'
        else:
            return 'neutral'
    
    def generate_emotion_sequence(self, mbti_type: str, user_id: str, num_turns: int) -> List[EmotionState]:
        """生成用户的情绪状态序列"""
        A = self.mbti_matrices[mbti_type]
        initial_config = self.mbti_initial_states[mbti_type]
        
        # 采样初始状态
        cov_matrix = initial_config["cov"] ** 2 * np.eye(3)
        x0 = np.random.multivariate_normal(initial_config["mean"], cov_matrix)
        x0 = np.clip(x0, -1, 1)  # 边界约束
        
        # 生成情绪序列
        emotion_sequence = []
        x_current = x0
        
        for turn in range(num_turns):
            # 记录当前状态
            emotion_sequence.append(EmotionState(
                pleasure=float(x_current[0]),
                arousal=float(x_current[1]),
                dominance=float(x_current[2]),
                turn=turn
            ))
            
            # 计算下一状态
            xi = np.random.normal(0, 0.05, 3)  # 扰动项
            x_next = A @ x_current + xi
            x_next = np.clip(x_next, -1, 1)  # 边界约束
            x_current = x_next
            
        return emotion_sequence
    
    def call_deepseek_api(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> str:
        """调用DeepSeek API生成文本"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                with self.api_lock:
                    self.total_api_calls += 1
                
                response = requests.post(
                    self.deepseek_base_url, 
                    headers=headers, 
                    json=data, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:  # 限流
                    wait_time = 2 ** attempt
                    logger.warning(f"DeepSeek API限流，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"DeepSeek API错误: {response.status_code}, {response.text}")
                    
            except Exception as e:
                logger.warning(f"DeepSeek API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    with self.api_lock:
                        self.failed_calls += 1
                    logger.error(f"DeepSeek API调用彻底失败: {e}")
                    return "[API调用失败]抱歉，我遇到了一些技术问题，请您稍后再试。"
        
        return "[API调用失败]抱歉，我遇到了一些技术问题，请您稍后再试。"
    
    def generate_user_response(self, mbti_type: str, emotion_state: EmotionState, 
                              scenario: Dict[str, str], turn: int, emotion_label: str, 
                              agent_text: str = None) -> str:
        """生成用户回复"""
        template = self.conversation_templates[mbti_type]
        
        # 情绪描述
        if emotion_state.pleasure > 0.3:
            pleasure_desc = "愉快开心"
        elif emotion_state.pleasure < -0.3:
            pleasure_desc = "不愉快烦躁"
        else:
            pleasure_desc = "心情平静"
            
        if emotion_state.arousal > 0.3:
            arousal_desc = "精神兴奋"
        elif emotion_state.arousal < -0.3:
            arousal_desc = "状态低落"
        else:
            arousal_desc = "情绪稳定"
            
        if emotion_state.dominance > 0.3:
            dominance_desc = "主动积极"
        elif emotion_state.dominance < -0.3:
            dominance_desc = "被动消极"
        else:
            dominance_desc = "态度中性"
        
        context = f"客服刚才回复：{agent_text}" if agent_text else "开始新的对话"
        
        system_prompt = f"""你是一个{mbti_type}性格类型的用户，具有以下特征：
{template['traits']}

你的语言风格特点：{template['style']}"""
        
        prompt = f"""你正在与客服就{scenario['name']}问题进行在线对话。

对话背景：{scenario['context']}
场景描述：{scenario['description']}

当前是第{turn + 1}轮对话。{context}

你当前的情绪状态：
- 整体心情：{pleasure_desc}（愉悦度：{emotion_state.pleasure:.2f}）
- 精神状态：{arousal_desc}（唤醒度：{emotion_state.arousal:.2f}）
- 行为态度：{dominance_desc}（主导感：{emotion_state.dominance:.2f}）
- 具体情绪：{emotion_label}

请根据你的{mbti_type}性格特征和当前情绪状态，生成一句符合情境的用户回复。

要求：
1. 严格体现{mbti_type}的性格特征和语言风格
2. 准确反映当前的情绪状态"{emotion_label}"
3. 回复自然流畅，符合中文表达习惯
4. 长度控制在15-50字之间
5. 如果是投诉处理场景且情绪负面，可以适当表达不满

请直接输出用户回复，不要添加任何解释："""
        
        return self.call_deepseek_api(prompt, system_prompt)
    
    def generate_agent_response(self, user_text: str, emotion_label: str, 
                               agent_type: str, scenario: Dict[str, str], mbti_type: str) -> str:
        """生成客服回复"""
        
        if agent_type == "AI":
            agent_desc = "AI智能客服，回复专业标准化，语言正式规范"
            agent_style = "使用标准化的客服用语，保持专业性"
        elif agent_type == "Switched":
            agent_desc = "刚切换到人工客服，需要理解用户情绪并提供更个性化的服务"
            agent_style = "更加人性化和个性化，体现人工客服的温暖"
        else:  # Human
            agent_desc = "人工客服，回复更温暖个性化，能更好理解用户情绪"
            agent_style = "语言亲切自然，更有人情味，善于情绪安抚"
        
        template = self.conversation_templates[mbti_type]
        
        system_prompt = f"""你是一个{agent_desc}，正在为用户提供{scenario['name']}服务。

你的服务风格：{agent_style}"""
        
        prompt = f"""用户是{mbti_type}性格类型，具有以下特征：{template['traits']}

场景信息：
- 服务类型：{scenario['name']}
- 场景描述：{scenario['description']}
- 服务背景：{scenario['context']}

用户刚才说："{user_text}"
用户当前情绪状态：{emotion_label}

请生成一句专业、贴心的客服回复。

要求：
1. 体现{agent_type}客服的特点和服务风格
2. 针对用户的{mbti_type}性格特征做出合适回应
3. 根据用户情绪状态"{emotion_label}"给予适当关怀或回应
4. 积极推进问题解决
5. 回复长度控制在20-80字之间
6. 语言自然流畅，符合中文客服表达习惯

请直接输出客服回复，不要添加任何解释："""
        
        return self.call_deepseek_api(prompt, system_prompt)
    
    def determine_agent_switch(self, emotion_sequence: List[EmotionState], 
                              threshold: float = 0.3) -> Optional[int]:
        """确定客服切换时机"""
        for i in range(1, len(emotion_sequence)):
            delta_p = emotion_sequence[i].pleasure - emotion_sequence[i-1].pleasure
            if delta_p < -threshold:
                return i
        return None
    
    def generate_user_data(self, mbti_type: str, user_idx: int) -> List[ConversationRecord]:
        """生成单个用户的完整对话数据"""
        user_id = f"{mbti_type}_{user_idx}"
        
        try:
            # 随机选择对话轮次和场景
            num_turns = random.randint(5, 12)
            scenario = random.choice(self.scenarios)
            
            logger.info(f"生成用户 {user_id} - {num_turns}轮对话 - {scenario['name']}")
            
            # 生成情绪序列
            emotion_sequence = self.generate_emotion_sequence(mbti_type, user_id, num_turns)
            
            # 确定切换时机
            switch_turn = self.determine_agent_switch(emotion_sequence)
            if switch_turn:
                logger.info(f"用户 {user_id} 在第{switch_turn}轮触发客服切换")
            
            # 生成对话记录
            user_records = []
            agent_text = None
            
            for turn_idx, emotion_state in enumerate(emotion_sequence):
                # 确定客服类型
                if switch_turn is None:
                    agent_type = "AI"
                elif turn_idx < switch_turn:
                    agent_type = "AI"
                elif turn_idx == switch_turn:
                    agent_type = "Switched"
                else:
                    agent_type = "Human"
                
                # 生成情绪标签
                emotion_label = self.map_pad_to_goemotion(
                    emotion_state.pleasure, emotion_state.arousal, emotion_state.dominance
                )
                
                # 生成用户回复
                user_text = self.generate_user_response(
                    mbti_type, emotion_state, scenario, turn_idx, emotion_label, agent_text
                )
                
                # 添加用户记录
                user_records.append(ConversationRecord(
                    user_id=user_id,
                    personality_type=mbti_type,
                    turn=turn_idx,
                    agent_type=agent_type,
                    speaker="user",
                    emotion_label=emotion_label,
                    text=user_text,
                    pleasure=emotion_state.pleasure,
                    arousal=emotion_state.arousal,
                    dominance=emotion_state.dominance,
                    scenario=scenario["name"],
                    switch_triggered=(switch_turn == turn_idx) if switch_turn else False
                ))
                
                # 生成客服回复
                agent_text = self.generate_agent_response(
                    user_text, emotion_label, agent_type, scenario, mbti_type
                )
                
                # 添加客服记录
                user_records.append(ConversationRecord(
                    user_id=user_id,
                    personality_type=mbti_type,
                    turn=turn_idx,
                    agent_type=agent_type,
                    speaker="agent",
                    emotion_label="",  # 客服不标注情绪
                    text=agent_text,
                    pleasure=np.nan,  # 客服不记录情绪状态
                    arousal=np.nan,
                    dominance=np.nan,
                    scenario=scenario["name"],
                    switch_triggered=False
                ))
                
                # 控制API调用频率
                time.sleep(0.05)
            
            return user_records
            
        except Exception as e:
            logger.error(f"生成用户 {user_id} 数据时出错: {e}")
            return []
    
    def generate_complete_dataset(self, max_workers: int = 3) -> List[ConversationRecord]:
        """生成完整的160用户数据集（支持并发）"""
        logger.info(" 开始生成完整情绪演化数据集")
        logger.info("数据规模：160用户，16种MBTI类型，每类型10个用户")
        logger.info(f"使用DeepSeek API，预计成本：$2-5")
        
        self.generation_start_time = time.time()
        all_records = []
        
        mbti_types = list(self.mbti_matrices.keys())
        total_users = len(mbti_types) * 10  # 160用户
        user_count = 0
        
        # 创建用户任务列表
        user_tasks = []
        for mbti_type in mbti_types:
            for user_idx in range(10):
                user_tasks.append((mbti_type, user_idx))
        
        # 使用线程池并发生成
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_user = {
                executor.submit(self.generate_user_data, mbti_type, user_idx): (mbti_type, user_idx)
                for mbti_type, user_idx in user_tasks
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_user), total=len(user_tasks), desc="生成用户数据"):
                mbti_type, user_idx = future_to_user[future]
                user_count += 1
                
                try:
                    user_records = future.result()
                    all_records.extend(user_records)
                    
                    # 定期报告进度
                    if user_count % 20 == 0:
                        elapsed = time.time() - self.generation_start_time
                        avg_time_per_user = elapsed / user_count
                        eta = avg_time_per_user * (total_users - user_count)
                        logger.info(f"进度：{user_count}/{total_users} ({user_count/total_users*100:.1f}%) - "
                                  f"已用时：{elapsed/60:.1f}分钟 - 预计剩余：{eta/60:.1f}分钟")
                        
                except Exception as e:
                    logger.error(f"生成用户 {mbti_type}_{user_idx} 失败: {e}")
                    continue
        
        logger.info(f" 数据集生成完成！共生成 {len(all_records)} 条记录")
        logger.info(f" DeepSeek API调用统计：总调用{self.total_api_calls}次，失败{self.failed_calls}次")
        
        return all_records
    
    def save_complete_dataset(self, records: List[ConversationRecord], filename_prefix: str = "deepseek_emotion_dataset"):
        """保存完整数据集"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}"
        
        # 转换为字典列表
        data = []
        for record in records:
            data.append(asdict(record))
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存为多种格式
        logger.info(" 保存数据集文件...")
        df.to_csv(filename + ".csv", index=False, encoding='utf-8')
        df.to_json(filename + ".json", orient='records', ensure_ascii=False, indent=2)
        try:
            df.to_excel(filename + ".xlsx", index=False)
        except Exception as e:
            logger.warning(f"Excel保存失败: {e}")
        
        logger.info(f" 数据集已保存:")
        logger.info(f"   - CSV: {filename}.csv")
        logger.info(f"   - JSON: {filename}.json")
        logger.info(f"   - Excel: {filename}.xlsx (如果安装了openpyxl)")
        
        # 生成统计报告
        self.generate_dataset_report(df, filename)
        
        return filename
    
    def generate_dataset_report(self, df: pd.DataFrame, filename: str):
        """生成数据集统计报告"""
        report = []
        report.append("=" * 60)
        report.append(" DeepSeek API完整情绪演化数据集统计报告")
        report.append("=" * 60)
        
        # 基础统计
        report.append(f"\n 基础统计:")
        report.append(f"   总记录数: {len(df):,} 条")
        report.append(f"   总用户数: {df['user_id'].nunique()} 个")
        report.append(f"   平均对话轮次: {df.groupby('user_id')['turn'].max().mean():.1f} 轮")
        
        # MBTI分布
        report.append(f"\n BTI人格类型分布:")
        mbti_counts = df['personality_type'].value_counts()
        for mbti_type, count in mbti_counts.items():
            users = count // 2  # 每个用户有用户+客服两条记录每轮
            report.append(f"   {mbti_type}: {users} 个用户")
        
        # 情绪标签分布
        report.append(f"\n 情绪标签分布 (用户发言):")
        emotion_counts = df[df['speaker'] == 'user']['emotion_label'].value_counts()
        for emotion, count in emotion_counts.head(10).items():
            report.append(f"   {emotion}: {count} 次")
        
        # 场景分布
        report.append(f"\n 对话场景分布:")
        scenario_counts = df['scenario'].value_counts()
        for scenario, count in scenario_counts.items():
            report.append(f"   {scenario}: {count} 条记录")
        
        # 客服切换统计
        switch_count = df['switch_triggered'].sum()
        total_users = df['user_id'].nunique()
        report.append(f"\n 客服切换统计:")
        report.append(f"   触发切换次数: {switch_count} 次")
        report.append(f"   切换比例: {switch_count/total_users*100:.1f}%")
        
        # 情绪状态统计
        user_records = df[df['speaker'] == 'user'].copy()
        user_records = user_records.dropna(subset=['pleasure', 'arousal', 'dominance'])
        
        if len(user_records) > 0:
            report.append(f"\n PAD情绪状态统计:")
            report.append(f"   愉悦度 (Pleasure):")
            report.append(f"     均值: {user_records['pleasure'].mean():.3f}")
            report.append(f"     标准差: {user_records['pleasure'].std():.3f}")
            report.append(f"   唤醒度 (Arousal):")
            report.append(f"     均值: {user_records['arousal'].mean():.3f}")
            report.append(f"     标准差: {user_records['arousal'].std():.3f}")
            report.append(f"   主导感 (Dominance):")
            report.append(f"     均值: {user_records['dominance'].mean():.3f}")
            report.append(f"     标准差: {user_records['dominance'].std():.3f}")
        
        # API调用统计
        report.append(f"\n DeepSeek API调用统计:")
        report.append(f"   总调用次数: {self.total_api_calls}")
        report.append(f"   失败次数: {self.failed_calls}")
        success_rate = (self.total_api_calls - self.failed_calls) / self.total_api_calls * 100 if self.total_api_calls > 0 else 0
        report.append(f"   成功率: {success_rate:.1f}%")
        
        # 生成时间统计
        if self.generation_start_time:
            total_time = time.time() - self.generation_start_time
            report.append(f"\n  生成时间统计:")
            report.append(f"   总耗时: {total_time/60:.1f} 分钟")
            report.append(f"   每用户平均: {total_time/total_users:.1f} 秒")
        
        # 成本估算
        cost_estimate = self.total_api_calls * 0.0005  # DeepSeek约$0.0005/1k tokens
        report.append(f"\n 成本统计:")
        report.append(f"   预计成本: ${cost_estimate:.2f}")
        report.append(f"   成本效率: 每条记录约${cost_estimate/len(df):.4f}")
        
        report.append(f"\n 数据集应用建议:")
        report.append(f"   - 可用于Lyapunov稳定性分析的情绪动力学建模")
        report.append(f"   - 支持不同人格类型的个性化情绪预测模型训练")
        report.append(f"   - 提供AI/人工客服切换机制的实证验证数据")
        report.append(f"   - 包含丰富的多轮对话语料用于情感计算研究")
        
        report.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report)
        with open(filename + "_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # 打印到控制台
        print(report_text)
        
        logger.info(f" 统计报告已保存: {filename}_report.txt")

    # 在您的代码最后添加这个修复函数
    def save_dataset_fixed(records, filename_prefix="deepseek_emotion_dataset"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}"

        # 转换数据
        data = [record.__dict__ if hasattr(record, '__dict__') else record for record in records]
        df = pd.DataFrame(data)

        # 保存CSV（主要格式）
        df.to_csv(filename + ".csv", index=False, encoding='utf-8')
        print(f" 数据集已保存: {filename}.csv")

        # 修复JSON保存
        try:
            df.to_json(filename + ".json", orient='records', force_ascii=False, indent=2)
        except TypeError:
            # 兼容旧版本pandas
            with open(filename + ".json", 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return filename, df

def main():
    """主函数"""
    print(" DeepSeek API完整版情绪演化数据集生成器")
    print("=" * 50)
    print("数据规模：160个用户，16种MBTI类型")
    print("预计生成：~2,736条对话记录")
    print("使用模型：DeepSeek-Chat")
    print("优势：成本低，质量高，速度快")
    print("=" * 50)
    
    # 获取DeepSeek API密钥
    deepseek_api_key = input("请输入您的DeepSeek API密钥: ").strip()
    
    if not deepseek_api_key:
        print(" API密钥不能为空")
        return
    
    # 可选的自定义API地址
    use_custom_url = input("是否使用自定义API地址？(y/n，默认n): ").strip().lower()
    if use_custom_url == 'y':
        custom_url = input("请输入自定义API地址: ").strip()
        base_url = custom_url if custom_url else "https://api.deepseek.com/v1/chat/completions"
    else:
        base_url = "https://api.deepseek.com/v1/chat/completions"
    
    # 估算成本
    print(f"\n 成本估算:")
    print(f"   预计API调用次数: ~5,440次")
    print(f"   DeepSeek-Chat成本: 约$2-5 (非常划算！)")
    print(f"   预计生成时间: 15-30分钟")
    
    # 询问并发设置
    use_concurrent = input("\n是否启用并发生成以提高速度？(y/n，默认y): ").strip().lower()
    if use_concurrent != 'n':
        max_workers = int(input("并发线程数 (建议2-4，默认3): ").strip() or "3")
    else:
        max_workers = 1
    
    confirm = input(f"\n确认开始生成完整数据集？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消生成")
        return
    
    # 创建生成器并开始生成
    try:
        generator = DeepSeekEmotionDatasetGenerator(deepseek_api_key, base_url)
        print(" DeepSeek API配置成功，开始生成数据集...")
        
        # 生成数据集
        records = generator.generate_complete_dataset(max_workers=max_workers)
        
        if not records:
            print(" 没有生成任何数据")
            return
        
        # 保存数据集
        # filename = generator.save_complete_dataset(records)
        filename, df = save_dataset_fixed(records)
        
        print(f"\n 数据集生成完成!")
        print(f"文件位置: {filename}.csv/.json/.xlsx")
        print(f"统计报告: {filename}_report.txt")
        print(f"\n DeepSeek API优势:")
        print(f"   ✓ 成本低廉，仅$2-5就能生成完整数据集")
        print(f"   ✓ 质量优秀，支持中文对话生成")
        print(f"   ✓ 速度快，并发处理效率高")
        print(f"   ✓ 稳定性好，API响应快速")
        
    except KeyboardInterrupt:
        print("\n  用户中断生成过程")
    except Exception as e:
        print(f"\n 生成过程出错: {e}")
        logger.error(f"生成失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()