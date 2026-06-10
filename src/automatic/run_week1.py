"""Week1任务执行脚本：数据准备与清洗"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing import load_data, get_basic_info, clean_data, save_data, validate_data

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 60)
    print("Week1任务：数据准备与清洗")
    print("=" * 60)
    
    # 1. 数据获取
    print("\n📥 步骤1：数据获取")
    print("-" * 40)
    
    # 加载示例数据（实际项目中应从Kaggle下载）
    data_path = 'data/raw/electric_vehicle_population_data.csv'
    
    try:
        df = load_data(data_path)
        print(f"✅ 数据加载成功，共 {len(df)} 条记录")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 初步探索
    print("\n🔍 步骤2：数据初步探索")
    print("-" * 40)
    
    info = get_basic_info(df)
    
    print(f"📊 数据形状: {info['shape']}")
    print(f"\n📋 列名: {info['columns']}")
    
    print("\n📈 数据类型:")
    dtype_df = pd.DataFrame.from_dict(info['dtypes'], orient='index', columns=['dtype'])
    print(dtype_df)
    
    print("\n🔍 缺失值统计:")
    missing_df = pd.DataFrame.from_dict(info['missing_count'], orient='index', columns=['count'])
    missing_df['percentage'] = (missing_df['count'] / len(df) * 100).round(2)
    print(missing_df)
    
    print("\n📊 数值特征统计描述:")
    print(df.describe().round(2))
    
    # 3. 业务假设
    print("\n📝 步骤3：业务假设设计")
    print("-" * 40)
    
    business_hypotheses = [
        "H1: 续航里程越长，车辆价格越高",
        "H2: 纯电车型价格高于插混车型",
        "H3: 车龄越大，车辆残值越低",
        "H4: 不同品牌的保值率存在显著差异",
        "H5: 智驾满意度高的车型保值率更高",
        "H6: 电池焦虑程度与二手车价格负相关"
    ]
    
    print("业务假设树：")
    for i, hypothesis in enumerate(business_hypotheses, 1):
        print(f"{i}. {hypothesis}")
    
    # 4. 数据验证
    print("\n✅ 步骤4：数据验证")
    print("-" * 40)
    
    validation = validate_data(df)
    print(f"重复行数: {validation['duplicates']}")
    print(f"负价格数量: {validation['negative_prices']}")
    print(f"无效续航数量: {validation['invalid_range']}")
    print(f"无效年份数量: {validation['invalid_year']}")
    
    # 5. 数据清洗
    print("\n🧹 步骤5：数据清洗")
    print("-" * 40)
    
    df_clean = clean_data(df)
    print(f"清洗前: {len(df)} 条记录")
    print(f"清洗后: {len(df_clean)} 条记录")
    
    # 6. 保存清洗后的数据
    print("\n💾 步骤6：保存数据")
    print("-" * 40)
    
    output_path = 'data/processed/cleaned_ev_data.csv'
    save_data(df_clean, output_path)
    
    # 7. 可视化探索
    print("\n📊 步骤7：可视化探索")
    print("-" * 40)
    
    # 价格分布
    plt.figure(figsize=(12, 6))
    sns.histplot(df_clean['Base MSRP'], bins=50, kde=True)
    plt.title('车辆价格分布')
    plt.xlabel('价格 ($)')
    plt.ylabel('数量')
    plt.savefig('reports/price_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 价格分布图已保存")
    
    # 续航分布
    plt.figure(figsize=(12, 6))
    sns.histplot(df_clean['Electric Range'], bins=50, kde=True)
    plt.title('续航里程分布')
    plt.xlabel('续航里程 (英里)')
    plt.ylabel('数量')
    plt.savefig('reports/range_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 续航分布图已保存")
    
    # 品牌分布
    plt.figure(figsize=(12, 6))
    top_brands = df_clean['Make'].value_counts().head(10)
    sns.barplot(x=top_brands.values, y=top_brands.index)
    plt.title('品牌分布（Top 10）')
    plt.xlabel('数量')
    plt.ylabel('品牌')
    plt.savefig('reports/brand_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 品牌分布图已保存")
    
    # 相关性热力图
    plt.figure(figsize=(12, 10))
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_clean[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('特征相关性热力图')
    plt.savefig('reports/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ 相关性热力图已保存")
    
    print("\n" + "=" * 60)
    print("Week1任务完成！")
    print("=" * 60)
    print(f"\n📁 输出文件:")
    print(f"   - {output_path}")
    print(f"   - reports/price_distribution.png")
    print(f"   - reports/range_distribution.png")
    print(f"   - reports/brand_distribution.png")
    print(f"   - reports/correlation_heatmap.png")

if __name__ == "__main__":
    main()