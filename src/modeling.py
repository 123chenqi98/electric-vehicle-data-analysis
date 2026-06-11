"""机器学习建模模块"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    分割训练集和测试集
    
    Args:
        df: 数据集
        target_col: 目标列名
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 移除目标变量为空的行
    df_clean = df.dropna(subset=[target_col]).copy()
    
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]
    
    # 确保y是Series（不是DataFrame）
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # 确保y是数值类型
    y = pd.to_numeric(y, errors='coerce')
    
    # 再次移除NaN值
    valid_mask = ~y.isna()
    X = X[valid_mask].copy()
    y = y[valid_mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 20, method: str = 'f_regression'):
    """
    特征选择
    
    Args:
        X: 特征矩阵
        y: 目标变量
        k: 选择的特征数量
        method: 特征选择方法
        
    Returns:
        X_selected: 选择后的特征矩阵
        selected_features: 选择的特征名称列表
        feature_scores: 特征评分
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
    
    return X_selected, selected_features.tolist(), feature_importance

def build_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame = None, y_test: pd.Series = None,
                       params: dict = None):
    """
    构建XGBoost回归模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征（可选）
        y_test: 测试目标（可选）
        params: 模型参数（可选）
    
    Returns:
        model: 训练好的模型
        metrics: 评估指标（如果提供了测试集）
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
    
    # 确保y_train是Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    
    # 确保X_train只包含数值列
    X_train = X_train.select_dtypes(include=[np.number])
    
    # 处理缺失值
    X_train = X_train.fillna(X_train.median())
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    metrics = None
    if X_test is not None and y_test is not None:
        # 确保y_test是Series
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]
        
        # 确保X_test只包含数值列
        X_test = X_test.select_dtypes(include=[np.number])
        X_test = X_test.fillna(X_test.median())
        
        y_pred = model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return model, metrics

def build_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame = None, y_test: pd.Series = None,
                         params: dict = None):
    """
    构建LightGBM回归模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征（可选）
        y_test: 测试目标（可选）
        params: 模型参数（可选）
    
    Returns:
        model: 训练好的模型
        metrics: 评估指标（如果提供了测试集）
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'regression'
        }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    metrics = None
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return model, metrics

def build_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame = None, y_test: pd.Series = None,
                        params: dict = None):
    """
    构建随机森林回归模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征（可选）
        y_test: 测试目标（可选）
        params: 模型参数（可选）
    
    Returns:
        model: 训练好的模型
        metrics: 评估指标（如果提供了测试集）
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    metrics = None
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return model, metrics

def build_linear_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame = None, y_test: pd.Series = None,
                       model_type: str = 'linear'):
    """
    构建线性回归模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征（可选）
        y_test: 测试目标（可选）
        model_type: 模型类型（linear, ridge, lasso）
    
    Returns:
        model: 训练好的模型
        metrics: 评估指标（如果提供了测试集）
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=1.0)
    else:
        raise ValueError("model_type参数只能是'linear', 'ridge'或'lasso'")
    
    model.fit(X_train, y_train)
    
    metrics = None
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return model, metrics

def compare_models(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    对比多个模型的性能
    
    Args:
        models_dict: 模型字典，key为模型名称，value为模型对象
        X_test: 测试特征
        y_test: 测试目标
    
    Returns:
        模型性能对比数据框
    """
    results = []
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        result = {
            'model': name,
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        results.append(result)
    
    return pd.DataFrame(results).sort_values('r2', ascending=False)

def hyperparameter_tuning(model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                         param_grid: dict, cv: int = 5):
    """
    超参数调优
    
    Args:
        model_type: 模型类型（xgboost, lightgbm, random_forest）
        X_train: 训练特征
        y_train: 训练目标
        param_grid: 参数网格
        cv: 交叉验证折数
    
    Returns:
        best_model: 最优模型
        best_params: 最优参数
    """
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(random_state=42, objective='regression')
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("model_type参数只能是'xgboost', 'lightgbm'或'random_forest'")
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, file_path: str):
    """保存模型"""
    joblib.dump(model, file_path)
    print(f"模型已保存到: {file_path}")

def load_model(file_path: str):
    """加载模型"""
    return joblib.load(file_path)

def ablation_study(X_traditional: pd.DataFrame, X_enhanced: pd.DataFrame, y: pd.Series,
                  model_type: str = 'xgboost') -> dict:
    """
    消融实验：对比传统特征与AI增强特征的效果
    
    Args:
        X_traditional: 传统特征
        X_enhanced: 增强特征（传统+AI）
        y: 目标变量
        model_type: 模型类型
    
    Returns:
        对比结果字典
    """
    # 处理y中的NaN值
    y_clean = pd.to_numeric(y, errors='coerce')
    valid_mask = ~y_clean.isna()
    
    X_traditional_clean = X_traditional[valid_mask].copy()
    X_enhanced_clean = X_enhanced[valid_mask].copy()
    y_clean = y_clean[valid_mask]
    
    # 分割数据
    X_train_t, X_test_t, y_train, y_test = train_test_split(
        X_traditional_clean, y_clean, test_size=0.2, random_state=42
    )
    X_train_e, X_test_e, _, _ = train_test_split(
        X_enhanced_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # 构建模型
    if model_type == 'xgboost':
        baseline_model, baseline_metrics = build_xgboost_model(X_train_t, y_train, X_test_t, y_test)
        enhanced_model, enhanced_metrics = build_xgboost_model(X_train_e, y_train, X_test_e, y_test)
    elif model_type == 'lightgbm':
        baseline_model, baseline_metrics = build_lightgbm_model(X_train_t, y_train, X_test_t, y_test)
        enhanced_model, enhanced_metrics = build_lightgbm_model(X_train_e, y_train, X_test_e, y_test)
    
    # 计算提升幅度
    improvement = {
        'r2_improvement': (enhanced_metrics['r2'] - baseline_metrics['r2']) / abs(baseline_metrics['r2']) * 100,
        'mae_improvement': (baseline_metrics['mae'] - enhanced_metrics['mae']) / baseline_metrics['mae'] * 100,
        'rmse_improvement': (baseline_metrics['rmse'] - enhanced_metrics['rmse']) / baseline_metrics['rmse'] * 100
    }
    
    return {
        'comparison': pd.DataFrame([
            {'model': 'Baseline (Traditional)', **baseline_metrics},
            {'model': 'Enhanced (AI Features)', **enhanced_metrics}
        ]),
        'improvement': improvement,
        'baseline_model': baseline_model,
        'enhanced_model': enhanced_model
    }