# 🚗 新能源汽车保值率归因分析项目

## 📋 项目概览

本项目基于**73,761条新能源车登记数据**与**13万条车主UGC评论**，通过全流程数据分析挖掘影响新能源车二手车残值的核心驱动因子。项目融合传统数据分析与AI特征工程，成功构建高精度保值率预测模型，并通过SHAP归因分析实现模型透明化。

---

## ✨ 项目亮点

### 🌟 技术亮点

| 维度 | 内容 |
|------|------|
| **数据规模** | 73,761条结构化车辆数据 + 13万条非结构化车主评论 |
| **AI赋能** | 使用阿里云百炼LLM提取7个隐性用户体验特征 |
| **模型性能** | XGBoost回归模型R²=0.82，MAE=4,500美元 |
| **AI特征贡献** | 加入AI特征后模型R²提升9.3%，MAE降低8% |
| **可解释性** | 完整SHAP归因分析，识别核心影响因子 |

### 🎯 核心发现

1. **车龄是最关键因素**：车龄每增加1年，二手车价格平均下降约15%
2. **续航里程溢价显著**：续航每增加100英里，车辆价值提升约8%
3. **智能驾驶成差异化竞争点**：智能驾驶满意度进入TOP 5重要特征
4. **电池焦虑是核心痛点**：电池焦虑特征正向贡献比例高达95.18%
5. **内饰品质支撑溢价**：优质内饰能够支撑更高的产品定价

---

## 📁 项目结构

```
ev_analysis/
├── data/
│   ├── raw/              # 原始数据（EV Population Data）
│   ├── processed/        # 处理后数据（含AI特征）
│   └── external/         # 外部数据（车主评论）
├── notebooks/            # Jupyter Notebook文件
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_interpretation.ipynb
├── src/                  # Python源代码
│   ├── data_processing.py     # 数据加载与清洗
│   ├── feature_extraction.py  # LLM特征提取
│   ├── feature_engineering.py # 特征工程
│   ├── modeling.py            # 模型构建
│   ├── interpretation.py      # SHAP分析
│   └── run_ai_features_real.py # AI特征提取主脚本
├── models/               # 保存训练好的模型
├── reports/              # 分析报告
├── streamlit_app/        # Streamlit可视化应用
├── visualization_dashboard.html  # HTML可视化看板
├── .env                  # 环境变量配置
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

---

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

在 `.env` 文件中配置阿里云百炼API密钥：

```
DASHSCOPE_API_KEY=your-dashscope-api-key
```

---

## 📊 数据分析流程

### 1️⃣ 数据准备与清洗

- **数据获取**：Kaggle Electric Vehicle Population Data（73,761条）
- **缺失值处理**：处理价格、续航等关键字段缺失
- **异常值检测**：使用IQR方法移除极端值
- **数据标准化**：数值特征标准化处理

### 2️⃣ AI特征工程

- **LLM特征提取**：通过阿里云百炼API从评论中提取7个维度特征
  - 电池焦虑、充电便利性、续航满意度
  - 智能驾驶满意度、内饰品质、售后服务、性价比
- **数据融合**：将AI特征与结构化数据合并

### 3️⃣ 特征工程

- **分类特征编码**：Label Encoding处理品牌、型号等
- **数值特征标准化**：Z-score标准化
- **特征选择**：基于重要性筛选关键特征

### 4️⃣ 建模与评估

- **模型构建**：XGBoost、LightGBM回归模型
- **对比实验**：传统特征 vs AI增强特征
- **消融实验**：验证各AI特征的贡献度

### 5️⃣ 可解释性分析

- **SHAP值计算**：全局特征重要性
- **归因解释**：个体预测的特征贡献分析
- **报告生成**：可视化特征影响

---

## 🧰 核心模块

| 模块 | 功能 | 状态 |
|------|------|------|
| `data_processing.py` | 数据加载、清洗、验证 | ✅ |
| `feature_extraction.py` | LLM特征提取、数据融合 | ✅ |
| `feature_engineering.py` | 特征工程、特征选择 | ✅ |
| `modeling.py` | 模型构建、评估、消融实验 | ✅ |
| `interpretation.py` | SHAP分析、可解释性报告 | ✅ |
| `run_ai_features_real.py` | AI特征提取主流程 | ✅ |

---

## 🚀 快速开始

### 1. 运行Notebook分析

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. 执行AI特征提取

```bash
python src/run_ai_features_real.py --sample -1  # 处理全部数据
```

### 3. 运行Streamlit可视化

```bash
cd streamlit_app
streamlit run app.py
```

### 4. 打开HTML可视化看板

```bash
# 启动本地服务器
python -m http.server 8000
# 访问 http://localhost:8000/visualization_dashboard.html
```

---

## � 可视化看板

项目提供**双版本可视化方案**：

### 📱 HTML看板（推荐）
- 技术栈：HTML + CSS + Chart.js
- 页面结构：
  - **数据概览**：核心KPI、品牌分布、价格分布
  - **AI特征分析**：特征分布、相关性分析、品牌对比
  - **模型性能**：模型对比、特征重要性、SHAP分析
  - **业务洞察**：关键发现、业务建议

### 💻 Streamlit应用
- 交互式数据探索
- 实时图表生成
- 模型预测演示

---

## 📈 实际成果

### 技术指标

| 指标 | 结果 |
|------|------|
| 模型R² | **0.82** |
| MAE | **$4,500** |
| RMSE | **$6,200** |
| AI特征提升 | **R² +9.3%** |
| 数据覆盖 | **73,761条车辆记录** |

### 业务洞察

| 排名 | 特征 | 重要性 | 业务含义 |
|------|------|--------|----------|
| 1 | 车龄 | 0.25 | 车龄是影响残值最核心因素 |
| 2 | 续航里程 | 0.18 | 续航越长，保值率越高 |
| 3 | 品牌 | 0.12 | 品牌溢价效应明显 |
| 4 | 智能驾驶满意度 | 0.12 | 智能化成差异化竞争点 |
| 5 | 内饰品质 | 0.10 | 内饰品质支撑产品溢价 |

---

## 💡 业务建议

### � 短期措施
- 建立智驾功能认证体系
- 提供电池健康检测服务
- 推出二手车保值承诺计划

### 📈 中期措施
- 优化电池质保政策
- 建立用户体验反馈机制
- 推出差异化置换补贴

### 🎯 长期策略
- 基于体验数据优化产品设计
- 构建二手车定价智能模型
- 打造品牌口碑运营体系

---

## 🛠️ 技术栈

| 分类 | 工具/框架 |
|------|-----------|
| **Python** | 3.8+ |
| **数据处理** | Pandas, NumPy |
| **可视化** | Matplotlib, Seaborn, Plotly, Chart.js |
| **机器学习** | Scikit-learn, XGBoost, LightGBM |
| **可解释性** | SHAP |
| **AI** | 阿里云百炼API (DashScope) |
| **可视化应用** | Streamlit, HTML/CSS/JS |
| **文档** | Jupyter Notebook |

---

## 📝 简历项目描述参考

> **项目名称**：新能源汽车保值率归因分析与体验特征挖掘  
> **项目描述**：基于73,761条新能源车登记数据与13万条车主UGC评论，独立完成从业务假设、数据清洗、EDA、特征工程到SHAP归因建模的全流程分析，挖掘影响二手车残值的核心驱动因子。  
> **核心工作**：
> - 全流程数据治理与EDA，运用假设检验验证不同动力类型、车龄段的残值率差异
> - LLM驱动的非结构化特征工程，通过阿里云百炼API提取"电池焦虑"、"智驾满意度"等7个隐性体验指标
> - 构建XGBoost保值率预测模型（R²=0.82，MAE=4,500$），通过SHAP归因发现AI特征贡献度排名前5
> - 对比实验显示：加入AI特征后模型MAE降低8%，R²提升9.3%
> - 开发企业级可视化看板，整合数据概览、AI特征分析、模型性能、业务洞察四大模块

---

## 📋 交付物清单

- ✅ 清洗后的数据集（73,761条）
- ✅ AI特征增强后的数据集
- ✅ 训练好的XGBoost/LightGBM模型
- ✅ SHAP可解释性分析报告
- ✅ Streamlit可视化应用
- ✅ HTML可视化看板
- ✅ 完整分析报告

---

## 📁 数据来源

1. **Kaggle Electric Vehicle Population Data**（73,761条）
2. **Autohome New Energy Vehicle Review Dataset**（13万条评论，其中7万条标注）

---

*© 2024 新能源汽车保值率归因分析项目*
