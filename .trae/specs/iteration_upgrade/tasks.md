# Tasks - 酒店预订取消预测系统迭代升级

## Phase 1: 模型能力增强（高优先级）

- [x] Task 1.1: 集成 SHAP 可解释性分析模块
  - [x] 安装 shap 依赖并编写 SHAP 分析服务类 (shap_analyzer.py)
  - [x] 实现 Summary Plot（全局特征重要性）生成与缓存
  - [x] 实现 Force Plot（单样本解释）生成 API
  - [x] 实现 Dependence Plot（特征依赖关系）生成
  - [x] 预测接口返回结果中附加 Top-5 关键特征贡献度
  - [x] 新增5个SHAP API端点

- [x] Task 1.2: 集成 Optuna 超参数自动优化框架
  - [x] 安装 optuna 依赖，创建超参数搜索空间定义 (hyperparameter_optimizer.py)
  - [x] 为 XGBoost/LightGBM/MLP 分别定义搜索空间
  - [x] 实现贝叶斯优化训练流程（含早停 + 交叉验证）
  - [x] 新增 `/api/models/optimize` 触发异步优化任务
  - [x] 新增 `/api/models/optimize/status` 查询优化进度
  - [x] 优化完成后自动替换模型文件并记录实验日志

- [x] Task 1.3: 实现模型融合策略（Stacking/Voting）
  - [x] 在 prediction_service.py 中新增 EnsemblePredictor 类
  - [x] 实现 VotingClassifier（软投票）
  - [x] 实现 StackingClassifier（基于AUC加权融合）
  - [x] 融合模型参与全模型预测接口 `/api/predict/all`
  - [x] 新增 `/api/predict/ensemble` API 端点
  - [x] 返回置信区间（均值/标准差/一致性指标）

- [x] Task 1.4: 新增 CatBoost 模型
  - [x] 安装 catboost 依赖
  - [x] 在 train_models.py 中添加 CatBoost 训练流程（优雅降级）
  - [x] CatBoost 融入现有评估和保存流程

## Phase 2: 特征工程升级（高优先级）

- [x] Task 2.1: 实现领域知识驱动的复合特征构造
  - [x] 新建 feature_engineering.py 模块（FeatureEngineer 类）
  - [x] 实现预订稳定性指数、总消费预估、忠诚度衰减、人均房价等8个复合特征
  - [x] 更新 train_models.py 和 prediction_service.py 的预处理流程

- [x] Task 2.2: 引入时序聚合特征
  - [x] 创建 temporal_features.py（TemporalFeatureEngineer 类）
  - [x] 实现7个时序聚合特征（滚动取消率、历史ADR、预订频次、提前期趋势等）
  - [x] 处理冷启动问题（新客户使用全局均值填充）

- [x] Task 2.3: 支持特征交互自动发现
  - [x] 创建 feature_interaction.py（FeatureInteractionDiscovery 类）
  - [x] 实现数值×数值(乘积/比率)、类别×类别(卡方检验)、数值×类别(F检验)三种交互
  - [x] 使用互信息评分排序，限制top-20防止维度爆炸

- [x] Task 2.4: 外部数据源接入
  - [x] 创建 external_data.py（WeatherDataService + HolidayCalendarService + ExternalDataIntegrator）
  - [x] 封装天气API调用（支持OpenWeatherMap/和风天气 + 季节性默认值降级）
  - [x] 封装节假日日历（中国+葡萄牙+国际主要节假日）
  - [x] 新增14个外部特征（7个天气 + 7个节假日）

## Phase 3: 业务智能增强（中优先级）

- [x] Task 3.1: 构建动态超售策略推荐引擎
  - [x] 新建 overbooking_engine.py（OverbookingEngine 类）
  - [x] 基于期望值优化 + 风险约束的超售量计算算法
  - [x] 支持三种风险偏好：保守(<1% walk) / 中性(<3%) / 激进(<5%)
  - [x] Walk概率正态近似估算
  - [x] 新增3个API端点（超售建议/多场景对比/每日分析）

- [x] Task 3.2: 实现客户画像与分群系统
  - [x] 新建 customer_analytics.py（CustomerAnalytics 类）
  - [x] 实现RFM分析（Recency/Frequency/Monetary代理指标）
  - [x] 实现客户取消风险评分（0-100，三级风险等级）
  - [x] K-Means客户分群（5群体自动命名）
  - [x] 新增3个API端点（洞察报告/分群详情/高风险列表）

- [x] Task 3.3: 开发高风险订单预警通知模块
  - [x] 新建 alert_service.py（AlertService 类）
  - [x] 预警规则引擎（默认70%阈值可配置）
  - [x] 支持站内消息/邮件/Webhook三渠道通知
  - [x] 自动预警集成到创建预订流程
  - [x] 新增5个API端点（列表/统计/解决/配置/批量扫描）

## Phase 4: 工程化与架构升级（中优先级）

- [x] Task 4.1: 集成 MLflow 模型版本管理
  - [x] 创建 mlflow_tracker.py（本地JSON实验追踪器，兼容MLflow接口设计）
  - [x] 修改 train_models.py 每次训练自动记录实验
  - [x] 记录内容：超参数、指标、模型 artifact、特征名
  - [x] 新增5个API端点（实验列表/详情/最佳/对比/删除）

- [x] Task 4.2: 数据库迁移至 PostgreSQL
  - [x] 创建 database_abstract.py 数据库抽象层
  - [x] 通过环境变量 DATABASE_URL 控制 SQLite/PostgreSQL 切换
  - [x] 所有函数签名保持向后兼容
  - [x] 新增数据库状态查询 API

- [x] Task 4.3: 引入 Redis 缓存层
  - [x] 创建 cache_service.py（RedisCache + DictCache 双后端）
  - [x] 声明式缓存装饰器 @cached()
  - [x] Redis不可用时自动降级为内存字典缓存
  - [x] 统计信息60s/模型信息300s/预测结果600s TTL
  - [x] 数据变更后主动使相关缓存失效
  - [x] 新增缓存状态/清除管理 API

## Phase 5: 前沿技术探索（低优先级/长期）

- [ ] Task 5.1: 研究 Survival Modeling（生存分析）（待后续实施）
- [ ] Task 5.2: 探索 LSTM/Transformer 时序预测模型（待后续实施）

- [x] Task 5.3: 实现概念漂移检测机制
  - [x] 创建 drift_detector.py（PSICalculator + DDMDetector + ConceptDriftMonitor）
  - [x] PSI（Population Stability Index）特征分布偏移检测（阈值0.1/0.25）
  - [x] DDM（Drift Detection Method）预测误差模式变化检测
  - [x] 综合监控报告 + 优先级排序操作建议
  - [x] 新增4个API端点（监控状态/设置基线/手动检测/事件历史）
  - [x] 训练完成后自动建立基线快照

# Task Dependencies
- [Task 1.2] 依赖于 [Task 1.1] ✅ 已完成
- [Task 1.3] 依赖于 [Task 1.1, Task 1.2] ✅ 已完成
- [Task 2.1] 可与 [Phase 1] 并行开发 ✅ 已完成
- [Task 3.1] 依赖于 [Task 1.3] ✅ 已完成
- [Task 3.2] 依赖于 [Task 2.2] ✅ 已完成
- [Task 3.3] 依赖于 [Task 1.3] ✅ 已完成
- [Task 4.1] 依赖于 [Phase 1] ✅ 已完成
- [Task 5.3] 依赖于 [Task 4.2] ✅ 已完成
