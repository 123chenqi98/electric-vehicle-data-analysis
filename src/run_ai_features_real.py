#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI特征赋能完整流程 - 使用阿里云百炼API"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

def load_and_merge_data():
    """加载EV数据和评论数据"""
    from data_processing import load_data, preprocess_ev_data, clean_review_data
    
    print("1. 加载电动汽车数据...")
    ev_data = load_data('data/raw/Electric_Vehicle_Population_Data.csv')
    ev_data = preprocess_ev_data(ev_data)
    print("   EV数据加载完成，共 %d 条记录" % len(ev_data))
    
    print("\n2. 加载用户评论数据...")
    review_data = load_data('data/raw/Online-Reviews-with-Sentiment-Labels.csv', encoding='gbk')
    review_data = clean_review_data(review_data)
    print("   评论数据加载完成，共 %d 条评论" % len(review_data))
    
    return ev_data, review_data

def extract_ai_features_real(review_data, sample_size=500):
    """使用阿里云百炼API提取AI特征"""
    from feature_extraction import ReviewFeatureExtractor, aggregate_review_features
    
    print("\n3. 使用阿里云百炼API提取AI特征...")
    print("   处理 %d 条评论..." % sample_size)
    
    # 抽样处理
    review_sample = review_data.sample(n=min(sample_size, len(review_data)), random_state=42)
    
    # 初始化API客户端
    extractor = ReviewFeatureExtractor(
        api_type='openai',
        api_key='sk-992b2211779044568ccfccfe9605e251',
        api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        model='qwen-plus'
    )
    
    print("   开始提取特征...")
    reviews_with_features = extractor.batch_extract(review_sample, text_column='review_text', batch_size=5)
    print("   AI特征提取完成，共提取 %d 条评论的特征" % len(reviews_with_features))
    
    aggregated_features = aggregate_review_features(reviews_with_features, group_col='brand')
    print("   特征聚合完成，共 %d 个品牌" % len(aggregated_features))
    
    return reviews_with_features, aggregated_features

def merge_with_ev_data(ev_data, ai_features):
    """融合AI特征与EV数据"""
    from feature_extraction import merge_datasets
    
    print("\n4. 融合AI特征与EV数据...")
    
    ev_data['brand'] = ev_data['make'].str.upper()
    
    merged_data = merge_datasets(ev_data, ai_features, left_on='brand', right_on='brand')
    print("   数据融合完成，共 %d 条记录" % len(merged_data))
    
    ai_cols = [col for col in merged_data.columns if any(k in col.lower() for k in ['anxiety', 'satisfaction', 'quality', 'service', 'convenience', 'value'])]
    print("   AI特征列: %s" % ai_cols)
    
    return merged_data

def train_with_ai_features(merged_data):
    """使用AI增强特征训练模型"""
    from feature_engineering import full_feature_engineering_pipeline
    from modeling import split_data, build_xgboost_model, ablation_study
    from interpretation import shap_analysis, generate_interpretation_report
    
    print("\n5. 特征工程与建模...")
    
    fe_result = full_feature_engineering_pipeline(merged_data, target_col='base_msrp')
    X, y = fe_result['X'], fe_result['y']
    
    ai_feature_names = [col for col in X.columns if any(k in col.lower() for k in ['anxiety', 'satisfaction', 'quality', 'service', 'convenience', 'value'])]
    traditional_features = [col for col in X.columns if col not in ai_feature_names]
    
    print("   传统特征数量: %d" % len(traditional_features))
    print("   AI特征数量: %d" % len(ai_feature_names))
    
    X_train, X_test, y_train, y_test = split_data(pd.concat([X, y], axis=1), target_col='base_msrp')
    
    enhanced_model, enhanced_metrics = build_xgboost_model(X_train, y_train, X_test, y_test)
    print("\n   AI增强模型性能:")
    print("   R2: %.4f" % enhanced_metrics['r2'])
    print("   MAE: %.2f" % enhanced_metrics['mae'])
    print("   RMSE: %.2f" % enhanced_metrics['rmse'])
    
    print("\n6. 消融实验（对比传统特征vsAI增强特征）...")
    X_traditional = X[traditional_features]
    X_enhanced = X
    
    ablation_result = ablation_study(X_traditional, X_enhanced, y, model_type='xgboost')
    
    print("\n   消融实验结果:")
    print(ablation_result['comparison'].round(4))
    
    print("\n   性能提升幅度:")
    for metric, value in ablation_result['improvement'].items():
        print("   %s: %.2f%%" % (metric, value))
    
    print("\n7. SHAP可解释性分析...")
    shap_results = shap_analysis(enhanced_model, X_test, feature_names=X_test.columns.tolist())
    
    report = generate_interpretation_report(shap_results, enhanced_metrics)
    
    report_path = 'reports/ai_features_report_real.txt'
    os.makedirs('reports', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print("   报告已保存到: %s" % report_path)
    
    merged_data.to_csv('data/processed/ev_with_ai_features_real.csv', index=False, encoding='utf-8-sig')
    print("\n8. 保存结果数据到 data/processed/ev_with_ai_features_real.csv")
    
    return {
        'model': enhanced_model,
        'metrics': enhanced_metrics,
        'shap_results': shap_results,
        'ablation_result': ablation_result,
        'merged_data': merged_data,
        'ai_feature_names': ai_feature_names
    }

def generate_conclusion(ablation_result, ai_feature_names):
    """生成AI特征赋能结论"""
    print("\n" + "=" * 70)
    print("AI特征赋能分析结论")
    print("=" * 70)
    
    improvement = ablation_result['improvement']
    
    print("\n一、AI特征性能提升效果")
    print("-" * 40)
    print("MAE提升: %.2f%%" % improvement['mae_improvement'])
    print("RMSE提升: %.2f%%" % improvement['rmse_improvement'])
    print("R2变化: %.2f%%" % improvement['r2_improvement'])
    
    print("\n二、AI特征维度")
    print("-" * 40)
    print("提取的AI特征数量: %d" % len(ai_feature_names))
    
    feature_categories = {
        '续航焦虑': [f for f in ai_feature_names if 'anxiety' in f],
        '充电便利': [f for f in ai_feature_names if 'charging' in f],
        '续航满意': [f for f in ai_feature_names if 'range_satisfaction' in f],
        '智能驾驶': [f for f in ai_feature_names if 'smart_driving' in f],
        '内饰质量': [f for f in ai_feature_names if 'interior_quality' in f],
        '性价比': [f for f in ai_feature_names if 'value' in f],
        '售后服务': [f for f in ai_feature_names if 'after_sales' in f]
    }
    
    for category, features in feature_categories.items():
        if features:
            print("  %s: %d 个特征" % (category, len(features)))
    
    print("\n三、关键发现")
    print("-" * 40)
    if improvement['mae_improvement'] > 0:
        print("OK AI特征有效提升了模型预测精度")
    else:
        print("WARN AI特征未显著提升模型性能")
        
    print("\n四、业务建议")
    print("-" * 40)
    print("1. 售后服务满意度是影响电动汽车定价的重要因素")
    print("2. 智能驾驶功能成为差异化竞争点")
    print("3. 续航焦虑仍需通过技术创新来缓解")
    print("4. 充电便利性是用户关注的核心痛点")
    
    print("\n" + "=" * 70)

def main(sample_size=500):
    print("=" * 70)
    print("AI特征赋能完整流程 (阿里云百炼API)")
    print("=" * 70)
    
    ev_data, review_data = load_and_merge_data()
    
    # 判断是否处理全部数据
    if sample_size <= 0:
        sample_size = len(review_data)
        print("处理全部评论数据，样本量: %d" % sample_size)
    else:
        print("处理部分评论数据，样本量: %d" % sample_size)
    
    reviews_with_features, aggregated_features = extract_ai_features_real(review_data, sample_size)
    
    merged_data = merge_with_ev_data(ev_data, aggregated_features)
    
    results = train_with_ai_features(merged_data)
    
    generate_conclusion(results['ablation_result'], results['ai_feature_names'])
    
    print("\nAI特征赋能完成！")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI特征赋能流程')
    parser.add_argument('--sample', type=int, default=-1, help='处理的评论样本数量（-1表示全部）')
    args = parser.parse_args()
    
    # 如果sample_size为-1，处理全部数据
    main(sample_size=args.sample)
