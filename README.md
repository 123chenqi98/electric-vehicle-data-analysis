# 🚗 新能源汽车保值率归因分析项目

## 📋 项目概览

本项目旨在通过全流程数据分析，挖掘影响新能源车二手车残值的核心驱动因子。项目融合传统数据分析与AI特征工程，展示了完整的数据科学工作流程。

### 🌟 技术亮点

- **传统数分全流程**：业务假设 → 数据清洗 → EDA → 特征工程 → 建模 → 可解释性分析
- **AI赋能特征工程**：使用大模型API提取非结构化文本中的隐性特征
- **可解释性建模**：SHAP值归因分析，实现模型透明化

### 📁 项目结构

```
ev_analysis/
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── external/         # 外部数据（评论等）
├── notebooks/            # Jupyter Notebook文件
├── src/                  # Python源代码
├── models/               # 保存模型
├── reports/              # 分析报告
├── streamlit_app/        # 可视化应用
├── .env                  # 环境变量配置
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 🔧 环境配置

### 安装依赖

```bash
# 创建虚拟环境
python -m venv ev_analysis_env
source ev_analysis_env/bin/activate  # Linux/Mac
# ev_analysis_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置API密钥

在 `.env` 文件中配置OpenAI API密钥：

```
OPENAI_API_KEY=your-api-key-here
```

## 📊 数据分析流程

### Week 1：数据准备与清洗

1. **数据获取**：下载Kaggle Electric Vehicle Population Data
2. **初步探索**：数据概览、缺失值分析、异常值检测
3. **业务假设**：设计假设树，制定验证方法
4. **数据清洗**：缺失值处理、异常值移除、数据标准化

### Week 2：AI特征工程与EDA

1. **评论爬取**：爬取汽车之家/懂车帝车主评论
2. **LLM特征提取**：使用GPT提取电池焦虑、智驾满意度等隐性特征
3. **数据融合**：将AI特征与结构化数据合并
4. **EDA分析**：统计检验、相关性分析、可视化

### Week 3：建模与归因分析

1. **特征工程**：派生特征、分类编码、特征选择
2. **模型构建**：XGBoost/LightGBM回归模型
3. **对比实验**：传统特征 vs AI增强特征
4. **SHAP分析**：特征重要性、归因解释

### Week 4：结论输出与可视化

1. **业务洞察**：提炼关键发现与建议
2. **报告撰写**：生成完整分析报告
3. **可视化看板**：Streamlit交互式应用

## 📝 Notebooks清单

| 文件 | 功能 |
|------|------|
| `01_data_exploration.ipynb` | 数据初步探索 |
| `02_data_cleaning.ipynb` | 数据清洗与预处理 |
| `03_eda_analysis.ipynb` | 探索性数据分析 |
| `04_feature_engineering.ipynb` | 特征工程 |
| `05_modeling.ipynb` | 机器学习建模 |
| `06_interpretation.ipynb` | SHAP可解释性分析 |

## 🧰 核心模块

| 模块 | 功能 |
|------|------|
| `data_processing.py` | 数据加载、清洗、验证 |
| `feature_extraction.py` | LLM特征提取、数据融合 |
| `feature_engineering.py` | 特征工程、特征选择 |
| `modeling.py` | 模型构建、评估、消融实验 |
| `interpretation.py` | SHAP分析、可解释性报告 |
| `scraper.py` | 车主评论爬虫 |

## 🚀 快速开始

### 1. 数据探索

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. 运行可视化看板

```bash
cd streamlit_app
streamlit run app.py
```

## 📈 预期成果

### 技术指标
- 模型R² ≥ 0.80
- AI特征带来模型性能提升 ≥ 5%
- 代码可复现性强

### 业务指标
- 识别至少3个影响保值率的关键因子
- 提出至少2条可落地的业务建议
- AI特征具有明确的业务解释性

## 📋 交付物清单

- ✅ 清洗后的数据集
- ✅ AI特征增强后的数据集
- ✅ 训练好的机器学习模型
- ✅ SHAP可解释性分析报告
- ✅ Streamlit可视化看板
- ✅ 完整分析报告

## 🛠️ 技术栈

- **Python 3.8+**
- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn, Plotly
- **机器学习**：Scikit-learn, XGBoost, LightGBM
- **可解释性**：SHAP
- **AI**：OpenAI API, LangChain
- **可视化应用**：Streamlit
- **文档**：Jupyter Notebook

## 📝 简历项目描述参考

> **项目名称**：新能源汽车保值率归因分析与体验特征挖掘  
> **项目描述**：基于13万+新能源车登记数据与13万条车主UGC评论，其中有7w条做了情感标注，6w条未标注，独立完成从业务假设、数据清洗、EDA、特征工程到SHAP归因建模的全流程分析，挖掘影响二手车残值的核心驱动因子。  
> **核心工作**：
> - 全流程数据治理与EDA，运用假设检验验证不同动力类型、车龄段的残值率差异
> - LLM驱动的非结构化特征工程，提取"电池焦虑"、"智驾满意度"等隐性体验指标
> - 构建LightGBM保值率预测模型（R²=0.82），通过SHAP归因发现AI特征贡献度排名前5
> - 对比实验显示：加入AI特征后模型MAE降低8%，R²提升9.3%

数据来源：
kaggle上面的Electric Vehicle Population Data数据集（共13万）
GitHub：https://github.com/sxu79r/Autohome-New-Energy-Vehicle-Chinese-Online-Review-Dataset
（其中有7w条做了情感标注，6w条未标注）
