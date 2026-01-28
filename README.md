AlloyDesign-AL: Aluminum Alloy Composition & Process Inverse Design System

AlloyDesign-AL: 铝合金成分与工艺逆向设计系统

🌟 Project Overview / 项目简介

AlloyDesign-AL is an integrated framework based on PyTorch for the forward property prediction and inverse design of aluminum alloys. The system bridges the gap between material composition/processing and mechanical performance (Yield Strength, Tensile Strength, and Elongation). It features a gradient-based optimization engine for "Inverse Engineering" and SHAP for model interpretability.

AlloyDesign-AL 是一个基于 PyTorch 的集成框架，用于铝合金的正向性能预测与逆向设计。该系统建立了材料成分/工艺与力学性能（屈服强度、抗拉强度和延伸率）之间的桥梁。其特色在于采用了基于梯度的优化引擎实现“逆向工程”，并集成 SHAP 分析提供模型的可解释性。

🚀 Key Features / 核心功能

Forward Modeling (正向建模): High-precision prediction of YS, TS, and EL using deep neural networks (DNN). 
使用深度神经网络（DNN）对屈服强度（YS）、抗拉强度（TS）和延伸率（EL）进行高精度预测。

Inverse Design (逆向设计): Searches for optimal composition and process parameters for target properties using historical-data hot start and constrained gradient optimization. 
利用历史数据热启动和约束梯度优化算法，反向寻优满足目标性能的成分与工艺参数。

SHAP Interpretability (可解释性分析): Identifies the physical influence of chemical elements and heat treatment parameters on alloy properties. 
识别化学元素及热处理参数对合金性能影响的物理贡献度。

Modular Architecture (模块化架构): Clean separation of training, analysis, and design modules for high maintainability. 
训练、分析与设计模块解耦，具有极高的可维护性。
📂 Project Structure / 项目结构

AlloyProject/

├── main.py                 # Main entry / 交互入口 (CLI)

├── train_module.py         # Training logic / 模型训练模块

├── design_module.py        # Inverse design engine / 逆向设计引擎

├── shap_module.py          # Interpretability / SHAP 分析模块

├── model_utils.py          # Data & Network utils / 数据处理与网络定义

├── data.xlsx               # Raw dataset / 原始数据集 (Sheet6)

└── results/                # Output images & logs / 输出图表与日志

🛠 Installation / 环境安装

Dependencies / 依赖库
Python 3.8+

PyTorch, NumPy, Pandas

Scikit-learn, Joblib

SHAP, Matplotlib, Openpyxl

pip install torch numpy pandas scikit-learn joblib shap matplotlib openpyxl

📖 Quick Start / 快速上手

Prepare Data (准备数据): Place data.xlsx in the root directory. Ensure it has 12 input columns and 3 target columns. 将 data.xlsx 放入根目录。确保包含 12 列输入和 3 列目标输出。

Train Model (训练模型): Run main.py and select option 1. 运行 main.py 并选择选项 1。生成 model_weights.pth 和 scaler 文件。

SHAP Analysis (分析): Select option 2 to generate feature importance plots. 选择选项 2 生成特征重要性热点图。

Inverse Design (逆向设计): Select option 3 to input target properties (e.g., 650, 700, 12) and get the recommended recipe. 选择选项 3 输入目标性能（如 650, 700, 12），获取推荐的成分工艺配方。

🧠 Implementation Details / 实现细节

1. Inverse Design Logic (逆向设计逻辑)
2. Unlike random search, our InverseDesigner uses:
K-Nearest Neighbor Search: Finds the best starting point from the historical dataset.
Constrained Optimization: Ensures the designed chemical composition and temperatures are within physical limits using penalty functions and clipping.
Trajectory Logging: All optimization steps are saved to design_optimization_log.xlsx.

不同于随机搜索，本系统的逆向设计器采用：
K-最近邻搜索：从历史数据集中寻找最佳初始点。 
约束优化：通过惩罚函数和裁剪操作，确保设计的成分和工艺在物理可行范围内。
轨迹记录：所有优化步骤均保存至 design_optimization_log.xlsx 供可视化分析。
4. Reproducibility (可复现性)
Random seeds are managed in model_utils.py to ensure consistency across different training runs.
随机种子在 model_utils.py 中统一管理，确保训练过程的可重复性。

⚠️ Notes / 注意事项

Weights First: You must run the Training module before attempting SHAP or Inverse Design. 

权重优先：在进行 SHAP 分析或逆向设计前，必须先完成模型训练。

File Format: Ensure the sheet name in data.xlsx matches PARAMS['SHEET_NAME']. 

文件格式：确保 Excel 的工作表名称与 model_utils.py 中的配置一致。
