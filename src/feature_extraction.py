"""LLM特征提取模块 - 支持多种API"""
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

class ReviewFeatureExtractor:
    """
    使用LLM从车主评论中提取结构化特征
    支持: OpenAI API, Hugging Face Inference API, 阿里云百炼API, 模拟数据
    """
    
    def __init__(self, api_type: str = 'mock', api_key: Optional[str] = None, 
                 api_base: Optional[str] = None, model: Optional[str] = None):
        """
        初始化
        
        Args:
            api_type: API类型 ('openai', 'huggingface', 'mock')
            api_key: API密钥
            api_base: API基础URL（用于自定义API端点，如阿里云百炼）
            model: 默认模型名称
        """
        self.api_type = api_type
        self.default_model = model
        
        if api_type == 'openai':
            try:
                from openai import OpenAI
                if api_key:
                    if api_base:
                        self.client = OpenAI(api_key=api_key, base_url=api_base)
                    else:
                        self.client = OpenAI(api_key=api_key)
                else:
                    self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                if not self.client.api_key:
                    raise ValueError("未提供OpenAI API密钥")
            except ImportError:
                print("警告: 未安装openai库，将使用模拟数据")
                self.api_type = 'mock'
                
        elif api_type == 'huggingface':
            try:
                from huggingface_hub import InferenceClient
                if api_key:
                    self.client = InferenceClient(api_key=api_key)
                else:
                    self.client = InferenceClient(api_key=os.getenv('HUGGINGFACE_API_KEY'))
            except ImportError:
                print("警告: 未安装huggingface_hub库，将使用模拟数据")
                self.api_type = 'mock'
                
        elif api_type == 'mock':
            print("使用模拟数据模式")
            np.random.seed(42)
            
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
    
    def _build_prompt(self, review_text: str) -> str:
        """构建LLM提示词"""
        prompt = f"""
        请分析以下汽车评论，并在各个维度上给出0-10的评分：
        
        评论内容：{review_text}
        
        评分维度（请严格按照JSON格式返回，不要添加任何额外内容）：
        {{
            "battery_anxiety": 电池衰减焦虑程度（0=完全不焦虑，10=非常焦虑）,
            "charging_convenience": 充电便利性（0=非常不便，10=非常便利）,
            "range_satisfaction": 续航满意度（0=非常不满意，10=非常满意）,
            "smart_driving_satisfaction": 智能驾驶功能满意度（0=非常不满意，10=非常满意）,
            "interior_quality": 内饰品质（0=很差，10=很好）,
            "value_for_money": 性价比（0=很差，10=很好）,
            "after_sales_service": 售后服务满意度（0=非常不满意，10=非常满意）
        }}
        
        要求：
        1. 每个维度必须返回0-10之间的数字
        2. 只返回JSON格式，不要其他文字解释
        3. 如果评论中未提及某个维度，请根据上下文合理推断
        """
        return prompt.strip()
    
    def _extract_openai(self, review_text: str, model: str = None) -> Dict[str, float]:
        """使用OpenAI API提取特征（支持阿里云百炼等兼容API）"""
        prompt = self._build_prompt(review_text)
        # 使用默认模型或传入的模型
        target_model = model or self.default_model or "gpt-3.5-turbo"
        
        try:
            response = self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的汽车评论分析专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"API错误: {e}")
            return self._generate_mock_features()
    
    def _extract_huggingface(self, review_text: str, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> Dict[str, float]:
        """使用Hugging Face API提取特征"""
        prompt = self._build_prompt(review_text)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的汽车评论分析专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content
            # 尝试解析JSON
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                print(f"Hugging Face返回非JSON格式: {result_text[:100]}...")
                return self._generate_mock_features()
                
        except Exception as e:
            print(f"Hugging Face API错误: {e}")
            return self._generate_mock_features()
    
    def _generate_mock_features(self) -> Dict[str, float]:
        """生成模拟特征"""
        return {
            'battery_anxiety': round(np.random.uniform(1, 9), 2),
            'charging_convenience': round(np.random.uniform(2, 10), 2),
            'range_satisfaction': round(np.random.uniform(3, 10), 2),
            'smart_driving_satisfaction': round(np.random.uniform(3, 9.5), 2),
            'interior_quality': round(np.random.uniform(4, 9.5), 2),
            'value_for_money': round(np.random.uniform(3, 9), 2),
            'after_sales_service': round(np.random.uniform(3, 9.5), 2)
        }
    
    def extract_features(self, review_text: str, model: str = None) -> Dict[str, float]:
        """
        从单条评论中提取特征
        
        Args:
            review_text: 评论内容
            model: 使用的LLM模型
            
        Returns:
            各维度评分字典
        """
        if self.api_type == 'openai':
            return self._extract_openai(review_text, model)
        elif self.api_type == 'huggingface':
            return self._extract_huggingface(review_text, model or "mistralai/Mistral-7B-Instruct-v0.3")
        else:
            return self._generate_mock_features()
    
    def batch_extract(self, df: pd.DataFrame, text_column: str = 'review_text', 
                     model: str = None, batch_size: int = 10) -> pd.DataFrame:
        """
        批量提取特征
        
        Args:
            df: 包含评论的数据框
            text_column: 评论文本列名
            model: 使用的LLM模型
            batch_size: 每批处理的数量
            
        Returns:
            添加了特征列的数据框
        """
        features_list = []
        total_reviews = len(df)
        
        for idx, row in df.iterrows():
            if (idx + 1) % 20 == 0:
                print(f"处理评论 {idx + 1}/{total_reviews}")
            
            features = self.extract_features(row[text_column], model)
            features_list.append(features)
            
            # 控制API调用频率
            if self.api_type != 'mock' and (idx + 1) % batch_size == 0 and idx + 1 < total_reviews:
                time.sleep(1)
        
        feature_df = pd.DataFrame(features_list)
        return pd.concat([df.reset_index(drop=True), feature_df], axis=1)

def aggregate_review_features(review_df: pd.DataFrame, group_col: str = 'car_model') -> pd.DataFrame:
    """
    将评论特征按车型聚合
    """
    df_copy = review_df.copy()
    
    if 'rating' in df_copy.columns:
        df_copy['rating'] = pd.to_numeric(df_copy['rating'], errors='coerce')
    
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    aggregation_rules = {}
    for col in ['battery_anxiety', 'charging_convenience', 'range_satisfaction',
                'smart_driving_satisfaction', 'interior_quality', 'value_for_money',
                'after_sales_service']:
        if col in numeric_cols:
            aggregation_rules[col] = ['mean', 'std', 'min', 'max']
    
    if 'rating' in numeric_cols:
        aggregation_rules['rating'] = ['mean', 'count']
    
    if not aggregation_rules:
        print("警告: 没有找到可聚合的数值列")
        return pd.DataFrame()
    
    aggregated = df_copy.groupby(group_col).agg(aggregation_rules)
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    return aggregated.reset_index()

def merge_datasets(ev_data: pd.DataFrame, review_features: pd.DataFrame, 
                  left_on: str = 'Model', right_on: str = 'car_model') -> pd.DataFrame:
    """
    融合EV数据和评论特征
    """
    merged = pd.merge(
        ev_data,
        review_features,
        left_on=left_on,
        right_on=right_on,
        how='left'
    )
    
    feature_cols = [col for col in merged.columns if any(keyword in col for keyword in 
                   ['anxiety', 'satisfaction', 'quality', 'service', 'convenience', 'value'])]
    merged[feature_cols] = merged[feature_cols].fillna(5.0)
    
    return merged

if __name__ == "__main__":
    extractor = ReviewFeatureExtractor(api_type='mock')
    
    sample_review = """
    这辆车的续航非常棒，实际能跑400公里以上，充电也很方便。智能驾驶功能很好用，
    高速上解放双手。就是内饰用料一般，有点异味。售后服务态度不错，响应很快。
    """
    
    features = extractor.extract_features(sample_review)
    print("提取的特征：")
    for key, value in features.items():
        print(f"{key}: {value}")