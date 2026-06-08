
# 酒店预订智能管理系统

一个完整的酒店预订管理和取消预测系统，集成了机器学习模型和 Web 管理界面。

## 功能特点

- ✅ **数据预处理与清洗** - 完整的数据预处理流程
- 🤖 **四种机器学习模型** - 集成学习（随机森林）、逻辑回归、支持向量机、神经网络
- 🗄️ **SQLite 数据库** - 完整的增删改查功能
- 🎨 **美观的 Web 界面** - 数据可视化和交互操作
- 🔮 **预订取消预测** - 实时预测预订是否会取消

## 项目结构

```
├── train_models.py          # 模型训练脚本
├── database.py              # 数据库操作模块
├── prediction_service.py    # 预测服务模块
├── app.py                   # Flask 后端 API
├── index.html               # 前端界面
├── requirements.txt         # Python 依赖
├── hotel_bookings.csv       # 原始数据集
├── prd.md                   # 产品需求文档
├── architecture.md          # 技术架构文档
└── README.md                # 项目说明
```

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练机器学习模型

```bash
python train_models.py
```

这将：
- 加载并预处理数据
- 训练四种模型
- 评估模型性能
- 保存模型到 `models/` 目录

### 3. 启动后端服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动

### 4. 打开前端界面

直接在浏览器中打开 `index.html` 文件即可

## API 接口

### 预订管理

- `GET /api/bookings` - 获取预订列表
- `GET /api/bookings/&lt;id&gt;` - 获取单个预订
- `POST /api/bookings` - 创建预订
- `PUT /api/bookings/&lt;id&gt;` - 更新预订
- `DELETE /api/bookings/&lt;id&gt;` - 删除预订

### 预测服务

- `POST /api/predict` - 预测单个预订
- `POST /api/predict/all` - 用所有模型预测
- `GET /api/models` - 获取可用模型列表
- `GET /api/models/performance` - 获取模型性能

### 统计

- `GET /api/statistics` - 获取统计数据

## 使用说明

1. **数据概览** - 查看预订统计和最新记录
2. **预订管理** - 添加、编辑、删除预订记录
3. **取消预测** - 输入预订信息，预测是否会取消
4. **模型评估** - 查看四种模型的性能对比

## 模型性能

训练完成后，在模型评估页面可以看到：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-score)

## 技术栈

- **后端**: Flask + Python
- **数据库**: SQLite
- **机器学习**: scikit-learn
- **前端**: HTML + CSS + JavaScript

## 注意事项

- 首次运行需要先执行 `train_models.py` 训练模型
- 确保 `hotel_bookings.csv` 文件在项目根目录
- 后端服务需要保持运行才能使用前端界面
