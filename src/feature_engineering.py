"""特征工程模块"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import List, Dict

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建派生特征
    
    Args:
        df: 原始数据集
    
    Returns:
        添加了派生特征的数据集
    """
    df_engineered = df.copy()
    
    # 1. 车龄特征（假设当前年份为2024）
    if 'model_year' in df_engineered.columns:
        df_engineered['car_age'] = 2024 - df_engineered['model_year']
    elif 'Model Year' in df_engineered.columns:
        df_engineered['car_age'] = 2024 - df_engineered['Model Year']
    
    # 2. 价格区间分箱
    if 'base_msrp' in df_engineered.columns:
        df_engineered['price_range'] = pd.cut(
            df_engineered['base_msrp'],
            bins=[0, 30000, 50000, 80000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Luxury']
        )
    elif 'Base MSRP' in df_engineered.columns:
        df_engineered['price_range'] = pd.cut(
            df_engineered['Base MSRP'],
            bins=[0, 30000, 50000, 80000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Luxury']
        )
    
    # 3. 续航效率（每万元价格对应的续航里程）
    if 'electric_range' in df_engineered.columns and 'base_msrp' in df_engineered.columns:
        df_engineered['range_efficiency'] = (
            df_engineered['electric_range'] / (df_engineered['base_msrp'] + 1) * 10000
        )
    elif 'Electric Range' in df_engineered.columns and 'Base MSRP' in df_engineered.columns:
        df_engineered['range_efficiency'] = (
            df_engineered['Electric Range'] / (df_engineered['Base MSRP'] + 1) * 10000
        )
    
    # 4. 充电速度特征（如果有相关数据）
    if 'fast_charge_speed' in df_engineered.columns:
        df_engineered['fast_charge_capable'] = (df_engineered['fast_charge_speed'] > 0).astype(int)
    elif 'Fast Charge Speed' in df_engineered.columns:
        df_engineered['fast_charge_capable'] = (df_engineered['Fast Charge Speed'] > 0).astype(int)
    
    # 5. 车型年代分组
    if 'model_year' in df_engineered.columns:
        df_engineered['generation'] = pd.cut(
            df_engineered['model_year'],
            bins=[2010, 2015, 2018, 2020, 2022, 2025],
            labels=['2010-2015', '2016-2018', '2019-2020', '2021-2022', '2023-2025']
        )
    elif 'Model Year' in df_engineered.columns:
        df_engineered['generation'] = pd.cut(
            df_engineered['Model Year'],
            bins=[2010, 2015, 2018, 2020, 2022, 2025],
            labels=['2010-2015', '2016-2018', '2019-2020', '2021-2022', '2023-2025']
        )
    
    return df_engineered

def encode_categorical_features(df: pd.DataFrame, columns: List[str], 
                                method: str = 'label') -> pd.DataFrame:
    """
    编码分类特征
    
    Args:
        df: 数据集
        columns: 需要编码的列名列表
        method: 编码方法（label/onehot/frequency）
    
    Returns:
        编码后的数据集
    """
    df_encoded = df.copy()
    
    for col in columns:
        if col not in df_encoded.columns:
            continue
        
        if method == 'label':
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        
        elif method == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded = ohe.fit_transform(df_encoded[[col]])
            feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
            df_encoded[feature_names] = encoded
        
        elif method == 'frequency':
            freq = df_encoded[col].value_counts(normalize=True)
            df_encoded[f'{col}_freq'] = df_encoded[col].map(freq)
    
    return df_encoded

def normalize_numeric_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    标准化数值特征
    
    Args:
        df: 数据集
        columns: 需要标准化的列名列表
    
    Returns:
        标准化后的数据集
    """
    df_normalized = df.copy()
    
    scaler = StandardScaler()
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    
    return df_normalized, scaler

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 20, 
                   method: str = 'f_regression') -> Dict:
    """
    特征选择
    
    Args:
        X: 特征矩阵
        y: 目标变量
        k: 选择的特征数量
        method: 特征选择方法
    
    Returns:
        特征选择结果字典
    """
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
    else:
        raise ValueError("method参数只能是'f_regression'或'mutual_info'")
    
    X_selected = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'score': feature_scores
    }).sort_values('score', ascending=False)
    
    return {
        'X_selected': X_selected,
        'selected_features': selected_features.tolist(),
        'feature_importance': feature_importance,
        'selector': selector
    }

def build_feature_matrix(df: pd.DataFrame, target_col: str, 
                         categorical_cols: List[str] = None,
                         numeric_cols: List[str] = None,
                         ai_feature_cols: List[str] = None) -> Dict:
    """
    构建特征矩阵
    
    Args:
        df: 数据集
        target_col: 目标列名
        categorical_cols: 分类特征列名列表
        numeric_cols: 数值特征列名列表
        ai_feature_cols: AI特征列名列表
    
    Returns:
        特征矩阵和目标变量
    """
    # 确定特征列
    if categorical_cols is None:
        categorical_cols = []
    
    if numeric_cols is None:
        numeric_cols = []
    
    if ai_feature_cols is None:
        ai_feature_cols = []
    
    # 合并所有特征列
    all_feature_cols = list(set(categorical_cols + numeric_cols + ai_feature_cols))
    
    # 确保特征列都存在
    all_feature_cols = [col for col in all_feature_cols if col in df.columns]
    
    # 创建特征矩阵
    X = df[all_feature_cols].copy()
    
    # 编码分类特征
    X = encode_categorical_features(X, categorical_cols, method='label')
    
    # 标准化数值特征和AI特征
    cols_to_normalize = [col for col in numeric_cols + ai_feature_cols if col in X.columns]
    if cols_to_normalize:
        X, scaler = normalize_numeric_features(X, cols_to_normalize)
    else:
        scaler = None
    
    # 移除原始分类列（保留编码后的列）
    X = X.drop([col for col in categorical_cols if col in X.columns], axis=1)
    
    # 确保所有特征都是数值类型
    for col in X.columns:
        if X[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                X = X.drop(col, axis=1)
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 获取目标变量
    y = df[target_col]
    
    # 确保目标变量也是数值类型
    y = pd.to_numeric(y, errors='coerce')
    
    return {
        'X': X,
        'y': y,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }

def full_feature_engineering_pipeline(df: pd.DataFrame, target_col: str) -> Dict:
    """
    完整的特征工程流程，确保所有分类特征都被编码
    
    Args:
        df: 原始数据集
        target_col: 目标列名
    
    Returns:
        特征工程结果字典
    """
    # 识别特征类型
    feature_types = identify_feature_types(df)
    
    # 构建特征矩阵
    result = build_feature_matrix(
        df,
        target_col,
        categorical_cols=feature_types['categorical_cols'],
        numeric_cols=feature_types['numeric_cols'],
        ai_feature_cols=feature_types['ai_feature_cols']
    )
    
    # 添加额外信息
    result['df_engineered'] = df
    result['feature_types'] = feature_types
    
    return result

def identify_feature_types(df: pd.DataFrame) -> Dict:
    """
    自动识别特征类型
    
    Args:
        df: 数据集
    
    Returns:
        特征类型字典
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 识别AI特征（包含特定关键词的列）
    ai_keywords = ['anxiety', 'satisfaction', 'quality', 'service', 'convenience', 'value']
    ai_feature_cols = [
        col for col in df.columns 
        if any(keyword in col.lower() for keyword in ai_keywords)
    ]
    
    return {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'ai_feature_cols': ai_feature_cols
    }

def feature_engineering_pipeline(df: pd.DataFrame, target_col: str) -> Dict:
    """
    完整的特征工程流程
    
    Args:
        df: 原始数据集
        target_col: 目标列名
    
    Returns:
        特征工程结果字典
    """
    # 1. 创建派生特征
    df_engineered = create_derived_features(df)
    
    # 2. 识别特征类型
    feature_types = identify_feature_types(df_engineered)
    
    # 3. 构建特征矩阵
    result = build_feature_matrix(
        df_engineered,
        target_col,
        categorical_cols=feature_types['categorical_cols'],
        numeric_cols=feature_types['numeric_cols'],
        ai_feature_cols=feature_types['ai_feature_cols']
    )
    
    # 添加额外信息
    result['df_engineered'] = df_engineered
    result['feature_types'] = feature_types
    
    return result