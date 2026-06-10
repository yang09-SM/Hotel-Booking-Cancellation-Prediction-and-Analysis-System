"""
客户画像与分群系统
基于 RFM 模型和取消风险评分构建客户维度分析能力
支持客户分群、生命周期识别和高风险客户预警
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CustomerAnalytics:
    """客户分析引擎"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.cluster_centers = None
        self.feature_importance = None

    def compute_rfm_features(self, df):
        """
        计算 RFM 特征
        R (Recency): 最近一次预订距今的时间间隔（用提前期代理）
        F (Frequency): 预订频次（用是否重复客 + 历史预订数代理）
        M (Monetary): 消费金额（用 ADR × 入住晚数代理）

        由于数据集中没有真实客户ID，使用 country + 其他字段作为客户代理分组
        """
        df = df.copy()

        # Recency: 用 lead_time 的倒数作为代理（lead_time 越短越"近期"）
        # 归一化到 0-1
        max_lead = df['lead_time'].max() if df['lead_time'].max() > 0 else 365
        df['rfm_recency'] = 1 - (df['lead_time'] / max_lead)  # 越近越大

        # Frequency: 综合指标
        df['rfm_frequency'] = (
            df['is_repeated_guest'].astype(float) * 0.5 +
            df['previous_bookings_not_canceled'].astype(float) / (
                df['previous_bookings_not_canceled'].max() + 1
            ) * 0.3 +
            (1 / (df['lead_time'].replace(0, 1)).astype(float)) * 0.2  # 频次代理
        )

        # Monetary: 总消费预估
        df['rfm_monetary'] = df['adr'].astype(float) * (
            df['stays_in_weekend_nights'].astype(float) +
            df['stays_in_week_nights'].astype(float)
        )

        # RFM 得分（各维度归一化后加权平均）
        for col in ['rfm_recency', 'rfm_frequency', 'rfm_monetary']:
            col_max = df[col].max() if df[col].max() != 0 else 1
            df[f'{col}_norm'] = df[col] / col_max

        df['rfm_score'] = (
            df['rfm_recency_norm'] * 0.25 +
            df['rfm_frequency_norm'] * 0.35 +
            df['rfm_monetary_norm'] * 0.40
        )

        return df

    def compute_cancellation_risk_score(self, df):
        """
        计算客户取消风险评分
        综合多个维度的风险因子
        """
        df = df.copy()

        # 风险因子1: 历史取消率
        total_history = (
            df['previous_cancellations'].astype(float) +
            df['previous_bookings_not_canceled'].astype(float) + 1
        )
        hist_cancel_rate = df['previous_cancellations'].astype(float) / total_history

        # 风险因子2: 提前期过长（提前期越长越不稳定）
        max_lead = df['lead_time'].max() if df['lead_time'].max() > 0 else 365
        lead_time_risk = df['lead_time'].astype(float) / max_lead

        # 风险因子3: 非重复客人
        new_guest_risk = (1 - df['is_repeated_guest'].astype(float)) * 0.3

        # 风险因子4: 变更次数过多
        max_changes = df['booking_changes'].max() if df['booking_changes'].max() > 0 else 1
        change_risk = df['booking_changes'].astype(float) / max_changes * 0.15

        # 风险因子5: 低价预订（价格敏感度高）
        avg_adr = df['adr'].mean() if df['adr'].mean() > 0 else 100
        price_sensitivity = np.where(df['adr'].astype(float) < avg_adr * 0.7, 0.15, 0)

        # 综合风险评分 (0-100)
        df['cancellation_risk_score'] = (
            hist_cancel_rate * 40 +
            lead_time_risk * 20 +
            new_guest_risk * 15 +
            change_risk * 10 +
            price_sensitivity
        ) * 100

        df['cancellation_risk_score'] = df['cancellation_risk_score'].clip(0, 100)

        # 风险等级划分
        def _risk_level(score):
            if score >= 70: return 'high'
            elif score >= 40: return 'medium'
            else: return 'low'

        df['risk_level'] = df['cancellation_risk_score'].apply(_risk_level)

        return df

    def perform_customer_segmentation(self, df, n_clusters=5):
        """
        执行 K-Means 客户分群
        基于 RFM + 行为特征进行聚类
        """
        # 先计算 RFM 和风险评分
        df = self.compute_rfm_features(df)
        df = self.compute_cancellation_risk_score(df)

        # 选择聚类特征
        cluster_features = [
            'rfm_recency_norm', 'rfm_frequency_norm', 'rfm_monetary_norm',
            'cancellation_risk_score', 'lead_time', 'adr',
            'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'total_of_special_requests'
        ]

        # 确保特征存在
        available_features = [f for f in cluster_features if f in df.columns]

        if len(available_features) < 3:
            return df, {'error': '特征不足，无法聚类'}

        X = df[available_features].fillna(0).values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # K-Means 聚类
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = self.cluster_model.fit_predict(X_scaled)
        self.cluster_centers = self.cluster_model.cluster_centers_

        # 为每个群体命名
        cluster_names = self._name_clusters(df)
        df['cluster_name'] = df['cluster'].map(cluster_names)

        # 聚类统计
        cluster_stats = df.groupby('cluster').agg({
            'rfm_score': 'mean',
            'cancellation_risk_score': 'mean',
            'adr': 'mean',
            'lead_time': 'mean',
            'is_canceled': 'mean',
            'cluster': 'count'
        }).rename(columns={'cluster': 'count'})

        cluster_stats['cancel_rate'] = cluster_stats['is_canceled']
        cluster_stats = cluster_stats.drop(columns=['is_canceled'])
        cluster_stats['name'] = cluster_stats.index.map(cluster_names)
        cluster_stats = cluster_stats.to_dict('index')

        return df, {
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'cluster_names': cluster_names,
            'features_used': available_features
        }

    def _name_clusters(self, df):
        """根据聚类中心的特征为每个群体命名"""
        if self.cluster_centers is None:
            return {i: f'群体{i+1}' for i in range(len(df['cluster'].unique()))}

        names = {}
        centers_df = pd.DataFrame(
            self.cluster_centers,
            columns=['recency', 'frequency', 'monetary', 'risk', 'lead_time', 'adr',
                     'weekend_stays', 'weekday_stays', 'adults', 'special_requests']
        )

        for i, center in centers_df.iterrows():
            r, f, m, risk = center['recency'], center['frequency'], center['monetary'], center['risk']

            if m > 0.6 and f > 0.6 and risk < 30:
                names[i] = '高价值稳定客'
            elif m > 0.5 and risk >= 50:
                names[i] = '高价值高风险客'
            elif risk >= 60 and f < 0.3:
                names[i] = '高风险新客'
            elif m < 0.3 and r < 0.4:
                names[i] = '价格敏感客'
            elif r > 0.6 and f > 0.4:
                names[i] = '忠诚复购客'
            else:
                names[i] = f'常规客户群{i+1}'

        return names

    def get_customer_insights(self, df):
        """
        生成完整的客户洞察报告
        """
        # 分群
        df_with_segments, segmentation_info = self.perform_customer_segmentation(df)

        # 整体统计
        insights = {
            'summary': {
                'total_customers_proxy': len(df),  # 用记录数代理客户数
                'avg_rfm_score': round(df_with_segments['rfm_score'].mean(), 2),
                'avg_risk_score': round(df_with_segments['cancellation_risk_score'].mean(), 1),
                'high_risk_count': int((df_with_segments['cancellation_risk_score'] >= 70).sum()),
                'high_risk_pct': round((df_with_segments['cancellation_risk_score'] >= 70).mean() * 100, 1),
                'overall_cancel_rate': round(df['is_canceled'].mean() * 100, 1)
            },
            'segmentation': segmentation_info,
            'segment_distribution': df_with_segments.groupby('cluster_name').size().to_dict(),
            'risk_distribution': df_with_segments['risk_level'].value_counts().to_dict(),
            'top_high_risk_customers': self._get_top_risk_customers(df_with_segments, top_n=10),
            'rfm_distribution': {
                'high_value': int((df_with_segments['rfm_score'] >= 0.6).sum()),
                'medium_value': int(((df_with_segments['rfm_score'] >= 0.3) & (df_with_segments['rfm_score'] < 0.6)).sum()),
                'low_value': int((df_with_segments['rfm_score'] < 0.3).sum())
            },
            'generated_at': datetime.now().isoformat()
        }

        return insights

    def _get_top_risk_customers(self, df, top_n=10):
        """获取 TOP-N 高风险客户"""
        high_risk = df.nlargest(top_n, 'cancellation_risk_score')[
            ['country', 'market_segment', 'customer_type', 'lead_time', 'adr',
             'previous_cancellations', 'cancellation_risk_score', 'risk_level', 'cluster_name']
        ].to_dict('records')

        return high_risk


def run_customer_analysis_from_db(db_path='hotel_bookings.db'):
    """
    从数据库加载数据并运行完整的客户分析
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM bookings', conn)
    conn.close()

    if len(df) == 0:
        return {'error': '无数据'}

    analyzer = CustomerAnalytics()
    insights = analyzer.get_customer_insights(df)

    return insights
