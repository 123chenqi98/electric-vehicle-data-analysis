"""Week2任务执行脚本：AI特征工程与EDA"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os
from dotenv import load_dotenv

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing import load_data, save_data
from feature_extraction import ReviewFeatureExtractor, aggregate_review_features, merge_datasets

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_reviews():
    """生成示例车主评论数据"""
    reviews = [
        {'car_model': 'Model 3', 'review_text': '特斯拉Model 3的续航非常给力，实际能跑400公里以上，充电也很方便。智能驾驶功能很好用，高速上解放双手。就是内饰用料一般，有点异味。售后服务态度不错，响应很快。', 'rating': '4.5'},
        {'car_model': 'Model 3', 'review_text': '开了一年多，电池衰减不明显，续航依然很扎实。FSD功能虽然不是每次都完美，但确实提升了驾驶体验。唯一不满意的是后排空间有点小。', 'rating': '4.0'},
        {'car_model': 'Model Y', 'review_text': '空间很大，适合家庭使用。续航比官方标称略低，但日常通勤完全够用。Autopilot在高速上非常好用，大大减轻了长途驾驶的疲劳。', 'rating': '4.5'},
        {'car_model': 'Model Y', 'review_text': '提车三个月，整体很满意。加速很快，操控性也不错。就是车机偶尔会卡顿，希望后续OTA能优化。', 'rating': '4.0'},
        {'car_model': 'Han EV', 'review_text': '比亚迪汉EV的内饰质感很好，科技感十足。续航表现不错，冬季打空调续航下降也在可接受范围内。DiPilot智能驾驶系统很好用。', 'rating': '4.5'},
        {'car_model': 'Han EV', 'review_text': '外观设计很大气，配置丰富。电池续航很扎实，充电速度也快。就是车机系统反应有点慢，期待优化。', 'rating': '4.0'},
        {'car_model': 'ET5', 'NIO': '蔚来ET5的颜值很高，内饰设计简约大方。换电模式太方便了，3分钟就能换完电。NOMI助手很智能，交互体验很好。', 'rating': '4.5'},
        {'car_model': 'ET5', 'review_text': '驾驶感受很好，加速线性。续航在CLTC工况下能达到官方标称，表现不错。服务体验一流，蔚来的换电网络很方便。', 'rating': '4.5'},
        {'car_model': 'P7', 'review_text': '小鹏P7的外观很科幻，风阻系数很低。XPILOT 4.0智驾系统非常强大，城市NGP很好用。续航表现中规中矩。', 'rating': '4.0'},
        {'car_model': 'P7', 'review_text': '车机系统很流畅，语音助手反应很快。续航冬季会缩水，但还能接受。就是后排头部空间有点局促。', 'rating': '3.5'},
        {'car_model': '001', 'review_text': '极氪001的猎装造型很独特，空间很大。续航表现不错，底盘质感很好。车机系统偶尔会卡顿，希望改进。', 'rating': '4.0'},
        {'car_model': '001', 'review_text': '加速很快，操控性很好。空气悬架舒适性不错。就是软件体验还有提升空间，偶尔会有小bug。', 'rating': '4.0'},
        {'car_model': 'ID.4', 'review_text': '大众ID.4的驾驶感受很像燃油车，容易上手。续航表现一般，冬季衰减比较明显。车机系统反应有点慢。', 'rating': '3.5'},
        {'car_model': 'ID.4', 'review_text': '空间很大，乘坐舒适。续航在同级别中不算突出，但日常够用。价格比较实惠，性价比不错。', 'rating': '3.5'},
        {'car_model': 'e-tron', 'review_text': '奥迪e-tron的内饰做工很好，豪华感十足。续航表现一般，充电速度还可以。底盘质感很好，行驶很平稳。', 'rating': '4.0'},
        {'car_model': 'e-tron', 'review_text': '品牌溢价比较高，配置丰富。续航不是强项，但日常使用足够。售后服务很好，奥迪的品质值得信赖。', 'rating': '4.0'},
        {'car_model': 'iX', 'review_text': '宝马iX的设计很前卫，内饰很有科技感。续航表现不错，充电速度快。驾驶感受依然是宝马的风格，操控很好。', 'rating': '4.5'},
        {'car_model': 'iX', 'review_text': '空间很大，舒适性很好。iDrive系统很流畅，语音助手好用。就是价格偏高，性价比一般。', 'rating': '4.0'},
        {'car_model': 'Model S', 'review_text': '特斯拉Model S Plaid加速太猛了，推背感十足。续航很长，适合长途旅行。内饰太简约了，很多功能都在屏幕里。', 'rating': '4.5'},
        {'car_model': 'Model X', 'review_text': '鹰翼门很酷炫，实用性也不错。空间非常大，适合大家庭。续航表现不错，充电速度快。', 'rating': '4.0'},
    ]
    
    # 添加更多随机评论
    car_models = ['Model 3', 'Model Y', 'Han EV', 'ET5', 'P7', '001', 'ID.4', 'e-tron', 'iX']
    sentiments = [
        ('续航', ['续航很给力', '续航表现不错', '续航一般', '续航有点短', '冬季续航衰减明显']),
        ('智驾', ['智驾很好用', '自动驾驶很方便', '智能驾驶体验不错', '智驾功能一般', '辅助驾驶不够成熟']),
        ('内饰', ['内饰质感很好', '内饰设计很棒', '内饰用料一般', '内饰有异味', '内饰科技感十足']),
        ('空间', ['空间很大', '空间宽敞', '后排空间有点小', '空间够用', '储物空间丰富']),
        ('充电', ['充电很方便', '充电速度快', '充电不太方便', '快充速度不错', '家用充电很方便']),
        ('售后', ['售后服务很好', '服务态度不错', '售后响应慢', '服务体验很好', '维保价格偏高'])
    ]
    
    for _ in range(100):
        model = np.random.choice(car_models)
        review_parts = []
        for aspect, options in sentiments:
            if np.random.random() > 0.3:
                review_parts.append(np.random.choice(options))
        
        review_text = '。'.join(review_parts) + '。'
        rating = str(np.round(np.random.uniform(3, 5), 1))
        
        reviews.append({
            'car_model': model,
            'review_text': review_text,
            'rating': rating
        })
    
    return pd.DataFrame(reviews)

def main():
    print("=" * 60)
    print("Week2任务：AI特征工程与EDA")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 1. 评论爬取
    print("\n📥 步骤1：评论数据获取")
    print("-" * 40)
    
    # 使用模拟数据（实际项目中可以使用scraper.py爬取真实数据）
    print("使用模拟评论数据（实际项目中可使用scraper.py爬取真实数据）")
    review_df = generate_sample_reviews()
    print(f"✅ 生成评论数据 {len(review_df)} 条")
    
    # 保存评论数据
    review_df.to_csv('data/external/car_reviews.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 评论数据已保存到 data/external/car_reviews.csv")
    
    # 2. LLM特征提取
    print("\n🤖 步骤2：LLM特征提取")
    print("-" * 40)
    
    if api_key and api_key != 'sk-your-api-key-here':
        try:
            extractor = ReviewFeatureExtractor(api_key=api_key)
            print(f"✅ 使用OpenAI API提取特征...")
            review_df_with_features = extractor.batch_extract(review_df, text_column='review_text', batch_size=10)
        except Exception as e:
            print(f"❌ API调用失败，使用模拟特征: {e}")
            review_df_with_features = generate_simulated_features(review_df)
    else:
        print("⚠️ 未配置有效API密钥，使用模拟特征")
        review_df_with_features = generate_simulated_features(review_df)
    
    # 保存带特征的评论数据
    review_df_with_features.to_csv('data/external/reviews_with_features.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 带特征的评论数据已保存")
    
    # 3. 数据融合
    print("\n🔗 步骤3：数据融合")
    print("-" * 40)
    
    # 加载清洗后的EV数据
    ev_data = load_data('data/processed/cleaned_ev_data.csv')
    print(f"加载EV数据: {len(ev_data)} 条")
    
    # 聚合评论特征
    aggregated_features = aggregate_review_features(review_df_with_features)
    print(f"聚合评论特征: {len(aggregated_features)} 个车型")
    
    # 数据融合
    merged_df = merge_datasets(ev_data, aggregated_features, left_on='Model', right_on='car_model')
    print(f"融合后数据: {len(merged_df)} 条")
    
    # 保存融合后的数据
    merged_df.to_csv('data/processed/final_dataset.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 融合后数据已保存到 data/processed/final_dataset.csv")
    
    # 4. EDA分析
    print("\n🔍 步骤4：EDA分析")
    print("-" * 40)
    
    # 价格与续航关系
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=merged_df, x='Electric Range', y='Base MSRP', alpha=0.6)
    plt.title('价格与续航里程的关系')
    plt.xlabel('续航里程 (英里)')
    plt.ylabel('价格 ($)')
    plt.savefig('reports/price_vs_range.png', dpi=300, bbox_inches='tight')
    print("✅ 价格vs续航图已保存")
    
    # 不同动力类型价格分布
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df, x='Electric Vehicle Type', y='Base MSRP')
    plt.title('不同动力类型的价格分布')
    plt.xlabel('动力类型')
    plt.ylabel('价格 ($)')
    plt.xticks(rotation=45)
    plt.savefig('reports/price_by_type.png', dpi=300, bbox_inches='tight')
    print("✅ 动力类型价格分布图已保存")
    
    # AI特征相关性分析
    ai_cols = [col for col in merged_df.columns if any(keyword in col.lower() for keyword in ['anxiety', 'satisfaction', 'quality', 'service'])]
    if ai_cols:
        plt.figure(figsize=(12, 10))
        ai_corr = merged_df[ai_cols + ['Base MSRP']].corr()
        sns.heatmap(ai_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('AI特征与价格相关性热力图')
        plt.savefig('reports/ai_features_correlation.png', dpi=300, bbox_inches='tight')
        print("✅ AI特征相关性热力图已保存")
    
    # 5. 统计检验
    print("\n📊 步骤5：统计检验")
    print("-" * 40)
    
    # T检验：纯电 vs 插混价格差异
    bev_prices = merged_df[merged_df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']['Base MSRP']
    phev_prices = merged_df[merged_df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']['Base MSRP']
    
    if len(bev_prices) > 0 and len(phev_prices) > 0:
        t_stat, p_value = stats.ttest_ind(bev_prices, phev_prices)
        print(f"T检验结果 - 纯电 vs 插混价格差异:")
        print(f"  T统计量: {t_stat:.3f}")
        print(f"  P值: {p_value:.3f}")
        print(f"  结论: {'存在显著差异' if p_value < 0.05 else '无显著差异'}")
    
    # 相关性分析：续航与价格
    range_corr, range_p = stats.pearsonr(merged_df['Electric Range'], merged_df['Base MSRP'])
    print(f"\n相关性分析 - 续航与价格:")
    print(f"  Pearson相关系数: {range_corr:.3f}")
    print(f"  P值: {range_p:.3f}")
    
    print("\n" + "=" * 60)
    print("Week2任务完成！")
    print("=" * 60)
    print(f"\n📁 输出文件:")
    print(f"   - data/external/car_reviews.csv")
    print(f"   - data/external/reviews_with_features.csv")
    print(f"   - data/processed/final_dataset.csv")
    print(f"   - reports/price_vs_range.png")
    print(f"   - reports/price_by_type.png")
    print(f"   - reports/ai_features_correlation.png")

def generate_simulated_features(df):
    """生成模拟的AI特征（当API不可用时使用）"""
    np.random.seed(42)
    df['battery_anxiety'] = np.random.uniform(1, 9, len(df))
    df['charging_convenience'] = np.random.uniform(2, 10, len(df))
    df['range_satisfaction'] = np.random.uniform(3, 10, len(df))
    df['smart_driving_satisfaction'] = np.random.uniform(3, 9.5, len(df))
    df['interior_quality'] = np.random.uniform(4, 9.5, len(df))
    df['value_for_money'] = np.random.uniform(3, 9, len(df))
    df['after_sales_service'] = np.random.uniform(3, 9.5, len(df))
    return df

if __name__ == "__main__":
    main()