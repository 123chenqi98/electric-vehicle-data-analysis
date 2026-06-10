"""数据处理模块"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """加载数据集"""
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except:
        try:
            return pd.read_csv(file_path, encoding='gbk')
        except:
            return pd.read_csv(file_path, encoding='utf-8-sig')

def load_ev_data() -> pd.DataFrame:
    """加载电动汽车人口数据集"""
    return load_data('data/raw/Electric_Vehicle_Population_Data.csv')

def load_review_data(with_sentiment: bool = True) -> pd.DataFrame:
    """加载用户评论数据"""
    if with_sentiment:
        return load_data('data/raw/Online-Reviews-with-Sentiment-Labels.csv', encoding='gbk')
    else:
        return load_data('data/raw/Online-Reviews-without-Sentiment-Labels.csv', encoding='gbk')

def get_basic_info(df: pd.DataFrame) -> Dict:
    """获取数据基本信息"""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_count': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'unique_counts': df.nunique().to_dict()
    }
    return info

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """处理缺失值"""
    df_processed = df.copy()
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            if strategy == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            elif strategy == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna('Unknown', inplace=True)
    
    return df_processed

def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.DataFrame:
    """使用IQR方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def remove_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.DataFrame:
    """使用IQR方法移除异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
    """使用Z-score方法检测异常值"""
    mean = df[column].mean()
    std = df[column].std()
    z_scores = (df[column] - mean) / std
    
    return df[np.abs(z_scores) > threshold]

def validate_data(df: pd.DataFrame) -> List[str]:
    """数据验证"""
    issues = []
    
    if df.empty:
        issues.append("数据集为空")
    
    for col in df.columns:
        if df[col].isnull().sum() > len(df) * 0.5:
            issues.append(f"列 {col} 缺失值超过50%")
        
        if df[col].nunique() == 1:
            issues.append(f"列 {col} 只有一个唯一值，缺乏信息量")
    
    return issues

def clean_review_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗评论数据"""
    df_clean = df.copy()
    
    # 重命名列
    if 'Column1' in df_clean.columns and 'Column3' in df_clean.columns:
        df_clean = df_clean.rename(columns={
            'Column1': 'brand',
            'Column2': 'model_id',
            'Column3': 'review_text',
            'Column4': 'sentiment'
        })
    
    # 过滤无效评论
    if 'review_text' in df_clean.columns:
        df_clean['review_length'] = df_clean['review_text'].apply(lambda x: len(str(x)))
        df_clean = df_clean[df_clean['review_length'] >= 10]
    
    return df_clean

def preprocess_ev_data(df: pd.DataFrame) -> pd.DataFrame:
    """预处理电动汽车数据"""
    df_processed = df.copy()
    
    # 选择关键列
    key_columns = ['Model Year', 'Make', 'Model', 'Electric Vehicle Type', 
                   'Electric Range', 'Base MSRP', 'State', 'County']
    df_processed = df_processed[key_columns]
    
    # 重命名列
    df_processed = df_processed.rename(columns={
        'Model Year': 'model_year',
        'Make': 'make',
        'Model': 'model',
        'Electric Vehicle Type': 'ev_type',
        'Electric Range': 'electric_range',
        'Base MSRP': 'base_msrp',
        'State': 'state',
        'County': 'county'
    })
    
    # 处理MSRP为0的情况（可能是数据缺失）
    df_processed['base_msrp'] = df_processed['base_msrp'].replace(0, np.nan)
    
    # 计算车龄
    current_year = 2024
    df_processed['age'] = current_year - df_processed['model_year']
    
    return df_processed

def save_processed_data(df: pd.DataFrame, filename: str):
    """保存处理后的数据"""
    df.to_csv(f'data/processed/{filename}', index=False, encoding='utf-8-sig')

def load_processed_data(filename: str) -> pd.DataFrame:
    """加载处理后的数据"""
    return pd.read_csv(f'data/processed/{filename}', encoding='utf-8-sig')
