"""
特征交互自动发现模块
通过二阶/三阶特征交叉和统计检验筛选有效交互特征
控制维度爆炸，提升模型判别力
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif

class FeatureInteractionDiscovery:
    """特征交互发现器"""

    def __init__(self, max_interactions=30, significance_level=0.05):
        """
        初始化
        max_interactions: 最大允许生成的交互特征数量
        significance_level: 统计检验显著性水平
        """
        self.max_interactions = max_interactions
        self.significance_level = significance_level
        self.selected_interactions = []  # 选中的交互特征列表
        self.interaction_scores = {}     # 交互特征评分

    def discover_interactions(self, X, y, numeric_cols=None, categorical_cols=None):
        """
        自动发现有效的特征交互
        输入:
            X: 特征 DataFrame
            y: 目标变量 Series
            numeric_cols: 数值型特征列名列表
            categorical_cols: 类别型特征列名列表
        返回:
            包含交互特征的新 DataFrame
        """
        X_new = X.copy()
        candidates = []

        # 确定数值型和类别型列
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"开始特征交互发现... (数值特征{len(numeric_cols)}个, 类别特征{len(categorical_cols)}个)")

        # === 1. 数值 × 数值 交互（乘积、比率）===
        candidates.extend(self._discover_numeric_numeric(X, y, numeric_cols))

        # === 2. 类别 × 类别 交互（交叉组合）===
        candidates.extend(self._discover_categorical_categorical(X, y, categorical_cols))

        # === 3. 数值 × 类别 交互（条件统计）===
        candidates.extend(self._discover_numeric_categorical(X, y, numeric_cols, categorical_cols))

        # 按评分排序，取 top-N
        candidates.sort(key=lambda x: x['score'], reverse=True)
        selected = candidates[:self.max_interactions]

        print(f"发现 {len(candidates)} 个候选交互特征，选取 top-{len(selected)} 个")

        # 应用选中的交互特征
        for item in selected:
            col_name = item['name']
            interaction_fn = item['function']
            X_new[col_name] = interaction_fn(X)
            self.selected_interactions.append(col_name)
            self.interaction_scores[col_name] = item['score']

        return X_new

    def _discover_numeric_numeric(self, X, y, numeric_cols):
        """发现数值×数值的有效交互"""
        candidates = []

        # 预定义一些业务上有意义的特征对（避免 O(n^2) 爆炸）
        meaningful_pairs = [
            ('lead_time', 'adr'),                    # 提前期 × 房价 → 价格敏感度
            ('adr', 'total_of_special_requests'),     # 房价 × 特殊请求 → 高价值客户指标
            ('adults', 'adr'),                       # 成人数 × 房价 → 人均成本
            ('stays_in_week_nights', 'adr'),         # 工作日晚数 × 房价
            ('previous_cancellations', 'lead_time'),  # 历史取消 × 提前期 → 风险累积
            ('booking_changes', 'lead_time'),        # 变更次数 × 提前期 → 决策摇摆
            ('adr', 'required_car_parking_spaces'),  # 房价 × 停车位需求
            ('adults', 'children'),                  # 成人 × 儿童 → 家庭规模
            ('stays_in_weekend_nights', 'stays_in_week_nights'),  # 周末 × 工作日
        ]

        for col1, col2 in meaningful_pairs:
            if col1 not in X.columns or col2 not in X.columns:
                continue

            # 交互方式1: 乘积
            prod_name = f'{col1}_x_{col2}'
            try:
                product = X[col1].astype(float) * X[col2].astype(float)
                score = self._score_feature(product, y)
                candidates.append({
                    'name': prod_name,
                    'type': 'numeric_numeric_product',
                    'features': [col1, col2],
                    'score': score,
                    'function': lambda d, c1=col1, c2=col2: d[c1].astype(float) * d[c2].astype(float)
                })
            except:
                pass

            # 交互方式2: 比率（当第二个特征不为0时）
            ratio_name = f'{col1}_div_{col2}'
            try:
                ratio = X[col1].astype(float) / (X[col2].astype(float).replace(0, np.nan))
                ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
                score = self._score_feature(ratio, y)
                candidates.append({
                    'name': ratio_name,
                    'type': 'numeric_numeric_ratio',
                    'features': [col1, col2],
                    'score': score,
                    'function': lambda d, c1=col1, c2=col2: (
                        d[c1].astype(float) / d[c2].astype(float).replace(0, 1e-6)
                    ).replace([np.inf, -np.inf], 0).fillna(0)
                })
            except:
                pass

        return candidates

    def _discover_categorical_categorical(self, X, y, categorical_cols):
        """发现类别×类别的有效交互（卡方检验筛选）"""
        candidates = []

        # 有意义的类别对
        meaningful_pairs = [
            ('hotel', 'market_segment'),
            ('hotel', 'distribution_channel'),
            ('hotel', 'customer_type'),
            ('market_segment', 'distribution_channel'),
            ('market_segment', 'customer_type'),
            ('meal', 'reserved_room_type'),
            ('country', 'market_segment'),
            ('deposit_type', 'customer_type'),
        ]

        for col1, col2 in meaningful_pairs:
            if col1 not in X.columns or col2 not in X.columns:
                continue

            cross_name = f'{col1}_x_{col2}'

            # 卡方检验评估交互显著性
            try:
                contingency = pd.crosstab(X[col1], X[col2])
                chi2, p_value, dof, expected = chi2_contingency(contingency)

                if p_value < self.significance_level:
                    # 显著性足够，创建交互特征（简单编码：拼接字符串后 hash）
                    score = -np.log(p_value + 1e-10)  # p越小分数越高

                    # 使用频率编码代替直接交叉（避免高基数）
                    def make_cross_func(c1, c2):
                        def fn(d):
                            combined = d[c1].astype(str) + '_' + d[c2].astype(str)
                            # 返回组合值的哈希编码
                            return pd.util.hash_pandas_object(combined).values % 10000 / 10000.0
                        return fn

                    candidates.append({
                        'name': cross_name,
                        'type': 'categorical_categorical',
                        'features': [col1, col2],
                        'score': score,
                        'p_value': p_value,
                        'function': make_cross_func(col1, col2)
                    })
            except:
                pass

        return candidates

    def _discover_numeric_categorical(self, X, y, numeric_cols, categorical_cols):
        """发现数值×类别的有效交互（条件统计）"""
        candidates = []

        # 有意义的数值-类别对
        meaningful_pairs = [
            ('adr', 'hotel'),                      # 不同酒店的房价水平
            ('lead_time', 'market_segment'),       # 不同渠道的提前期分布
            ('adr', 'customer_type'),              # 不同客户类型的消费能力
            ('lead_time', 'customer_type'),        # 不同客户类型的提前习惯
            ('adr', 'deposit_type'),               # 不同押金类型的房价
            ('total_of_special_requests', 'hotel'), # 不同酒店的特殊请求
            ('stays_in_week_nights', 'meal'),      # 不同餐标的工作日停留
        ]

        for num_col, cat_col in meaningful_pairs:
            if num_col not in X.columns or cat_col not in X.columns:
                continue

            inter_name = f'{num_col}_by_{cat_col}'

            try:
                # F 检验评估不同类别组的数值分布差异
                groups = [group[num_col].dropna().values for name, group in X.groupby(cat_col)]
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = f_oneway(*groups)

                    if p_value < self.significance_level:
                        score = -np.log(p_value + 1e-10)

                        # 创建条件偏差特征：值 - 该类别均值
                        def make_conditional_func(nc, cc):
                            category_means = X.groupby(cc)[nc].transform('mean')
                            def fn(d):
                                return d[nc].astype(float) - d[cc].map(
                                    category_means.drop_duplicates()
                                ).reindex(d[cc]).fillna(d[nc].mean())
                            return fn

                        candidates.append({
                            'name': inter_name,
                            'type': 'numeric_categorical',
                            'features': [num_col, cat_col],
                            'score': score,
                            'p_value': p_value,
                            'function': make_conditional_func(num_col, cat_col)
                        })
            except:
                pass

        return candidates

    def _score_feature(self, feature_values, y):
        """
        评估单个特征与目标变量的相关性
        使用互信息（Mutual Information）作为评分标准
        """
        try:
            feature_clean = feature_values.dropna().values.reshape(-1, 1)
            y_clean = y.loc[feature_values.dropna().index].values

            if len(feature_clean) < 50 or len(np.unique(y_clean)) < 2:
                return 0.0

            mi = mutual_info_classif(feature_clean, y_clean, random_state=42)
            return mi[0] if len(mi) > 0 else 0.0
        except:
            return 0.0

    def transform(self, X):
        """
        使用已发现的交互特征转换新数据
        （推理时调用，复用训练时发现的交互模式）
        """
        X_new = X.copy()

        for item in self.selected_interactions:
            if item in self.interaction_scores:
                # 需要重建 function 或存储已计算的值
                # 这里简化处理：实际应用中应持久化 interaction functions
                pass

        return X_new

    def get_interaction_summary(self):
        """获取交互特征摘要"""
        return [
            {'name': name, 'score': self.interaction_scores.get(name, 0)}
            for name in self.selected_interactions
        ]
