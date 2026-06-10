"""SHAP可解释性分析模块"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def shap_analysis(model, X: pd.DataFrame, feature_names: Optional[List[str]] = None):
    """
    SHAP值分析主函数
    
    Args:
        model: 训练好的模型
        X: 特征数据
        feature_names: 特征名称列表
    
    Returns:
        分析结果字典
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # 创建解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. 全局特征重要性
    global_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    # 2. 特征影响方向分析（汇总）
    feature_effects = []
    for i, name in enumerate(feature_names):
        # 计算正SHAP值比例
        positive_ratio = np.mean(shap_values[:, i] > 0)
        feature_effects.append({
            'feature': name,
            'mean_shap': np.mean(shap_values[:, i]),
            'positive_ratio': positive_ratio,
            'importance': np.mean(np.abs(shap_values[:, i]))
        })
    feature_effects_df = pd.DataFrame(feature_effects).sort_values('importance', ascending=False)
    
    # 3. AI特征贡献度分析
    ai_feature_indices = [i for i, name in enumerate(feature_names) 
                         if any(keyword in name.lower() for keyword in 
                               ['anxiety', 'satisfaction', 'quality', 'service', 'convenience', 'value'])]
    
    if ai_feature_indices:
        ai_contributions = {
            feature_names[i]: {
                'mean_abs_shap': np.mean(np.abs(shap_values[:, i])),
                'mean_shap': np.mean(shap_values[:, i]),
                'positive_ratio': np.mean(shap_values[:, i] > 0)
            }
            for i in ai_feature_indices
        }
        
        ai_contributions_df = pd.DataFrame(ai_contributions).T.reset_index()
        ai_contributions_df.columns = ['AI_Feature', 'Mean_Absolute_SHAP', 'Mean_SHAP', 'Positive_Ratio']
        ai_contributions_df = ai_contributions_df.sort_values('Mean_Absolute_SHAP', ascending=False)
    else:
        # 如果没有AI特征，返回空DataFrame
        ai_contributions_df = pd.DataFrame(columns=['AI_Feature', 'Mean_Absolute_SHAP', 'Mean_SHAP', 'Positive_Ratio'])
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'global_importance': global_importance,
        'feature_effects': feature_effects_df,
        'ai_contributions': ai_contributions_df
    }

def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, feature_names: List[str],
                      plot_type: str = 'dot', save_path: Optional[str] = None):
    """
    绘制SHAP汇总图
    
    Args:
        shap_values: SHAP值数组
        X: 特征数据
        feature_names: 特征名称
        plot_type: 图表类型（dot, bar, violin）
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 8))
    
    if plot_type == 'dot':
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    elif plot_type == 'bar':
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    elif plot_type == 'violin':
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="violin", show=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def plot_shap_force_plot(explainer, shap_values: np.ndarray, X: pd.Series, 
                         feature_names: List[str], sample_idx: int = 0,
                         save_path: Optional[str] = None):
    """
    绘制单个样本的SHAP力图
    
    Args:
        explainer: SHAP解释器
        shap_values: SHAP值数组
        X: 单个样本数据
        feature_names: 特征名称
        sample_idx: 样本索引
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 6))
    
    shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X.iloc[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame, 
                         feature_idx: int, feature_names: List[str],
                         interaction_idx: int = 'auto', save_path: Optional[str] = None):
    """
    绘制特征依赖图
    
    Args:
        shap_values: SHAP值数组
        X: 特征数据
        feature_idx: 特征索引
        feature_names: 特征名称
        interaction_idx: 交互特征索引
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False
    )
    
    plt.title(f'{feature_names[feature_idx]} 对预测的影响')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def plot_ai_feature_importance(ai_contributions_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    绘制AI特征重要性图
    
    Args:
        ai_contributions_df: AI特征贡献度数据框
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=ai_contributions_df,
        x='Mean_Absolute_SHAP',
        y='AI_Feature',
        palette='viridis'
    )
    
    plt.title('AI提取特征的SHAP重要性')
    plt.xlabel('平均绝对SHAP值')
    plt.ylabel('AI特征')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def calculate_feature_contribution(shap_results: dict, feature_name: str) -> dict:
    """
    计算单个特征的贡献度
    
    Args:
        shap_results: SHAP分析结果
        feature_name: 特征名称
    
    Returns:
        特征贡献度字典
    """
    feature_idx = list(shap_results['global_importance']['feature']).index(feature_name)
    
    return {
        'feature': feature_name,
        'importance_rank': feature_idx + 1,
        'mean_abs_shap': shap_results['global_importance'].iloc[feature_idx]['shap_importance'],
        'mean_shap': shap_results['feature_effects'].loc[
            shap_results['feature_effects']['feature'] == feature_name, 'mean_shap'
        ].values[0],
        'positive_ratio': shap_results['feature_effects'].loc[
            shap_results['feature_effects']['feature'] == feature_name, 'positive_ratio'
        ].values[0]
    }

def generate_interpretation_report(shap_results: dict, model_metrics: dict = None) -> str:
    """
    生成可解释性分析报告
    
    Args:
        shap_results: SHAP分析结果
        model_metrics: 模型性能指标（可选）
    
    Returns:
        分析报告字符串
    """
    report = []
    
    report.append("=" * 60)
    report.append("SHAP可解释性分析报告")
    report.append("=" * 60)
    
    if model_metrics:
        report.append("\n一、模型性能概览")
        report.append("-" * 40)
        report.append(f"R²分数: {model_metrics.get('r2', 'N/A'):.4f}")
        report.append(f"MAE: {model_metrics.get('mae', 'N/A'):.2f}")
        report.append(f"RMSE: {model_metrics.get('rmse', 'N/A'):.2f}")
    
    report.append("\n二、全局特征重要性（Top 10）")
    report.append("-" * 40)
    top_10 = shap_results['global_importance'].head(10)
    for idx, row in top_10.iterrows():
        report.append(f"{idx + 1}. {row['feature']}: {row['shap_importance']:.4f}")
    
    report.append("\n三、AI特征贡献度分析")
    report.append("-" * 40)
    if shap_results['ai_contributions'].empty:
        report.append("未检测到AI特征（如情感分析、评论数量等）")
    else:
        for _, row in shap_results['ai_contributions'].iterrows():
            report.append(f"{row['AI_Feature']}:")
            report.append(f"  - 平均绝对SHAP值: {row['Mean_Absolute_SHAP']:.4f}")
            report.append(f"  - 平均SHAP值: {row['Mean_SHAP']:.4f}")
            report.append(f"  - 正向贡献比例: {row['Positive_Ratio']:.2%}")
    
    report.append("\n四、关键发现")
    report.append("-" * 40)
    
    # 识别最重要的AI特征
    if not shap_results['ai_contributions'].empty:
        top_ai_feature = shap_results['ai_contributions'].iloc[0]
        report.append(f"1. 最重要的AI特征是'{top_ai_feature['AI_Feature']}'，其平均绝对SHAP值为{top_ai_feature['Mean_Absolute_SHAP']:.4f}")
    else:
        report.append("1. 数据集中未包含AI特征，无法进行AI特征贡献分析")
    
    # 分析特征影响方向
    positive_features = shap_results['feature_effects'][
        shap_results['feature_effects']['positive_ratio'] > 0.7
    ]['feature'].tolist()
    negative_features = shap_results['feature_effects'][
        shap_results['feature_effects']['positive_ratio'] < 0.3
    ]['feature'].tolist()
    
    if positive_features:
        report.append(f"2. 以下特征通常对价格产生正向影响: {', '.join(positive_features)}")
    if negative_features:
        report.append(f"3. 以下特征通常对价格产生负向影响: {', '.join(negative_features)}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)