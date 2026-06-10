"""
特征工程模块 - 领域知识驱动的复合特征构造
基于酒店行业业务经验和学术研究构建高信息量的衍生特征
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """复合特征构造器"""

    def __init__(self):
        self.feature_names = None

    def transform(self, df):
        """
        对输入DataFrame进行复合特征构造
        输入: 预处理后的DataFrame（已完成缺失值填充和编码）
        输出: 追加了复合特征的DataFrame
        """
        df = df.copy()

        # 1. 预订稳定性指数
        # 含义: 值越接近1表示预订越稳定（修改次数少或提前期长）
        df['booking_stability_index'] = self._calc_stability_index(df)

        # 2. 总消费预估
        # 含义: ADR × 总入住晚数，反映客户机会成本
        df['total_estimated_spend'] = self._calc_total_spend(df)

        # 3. 客户忠诚度衰减指数
        # 含义: 综合历史取消记录和提前期，反映客户可靠性
        df['loyalty_decay_index'] = self._calc_loyalty_decay(df)

        # 4. 人均消费
        df['adr_per_person'] = self._calc_adr_per_person(df)

        # 5. 入住压力指数
        # 含义: 总人数/入住晚数，反映房间使用强度
        df['occupancy_pressure'] = self._calc_occupancy_pressure(df)

        # 6. 提前期分段特征
        # 将连续的 lead_time 分为短/中/长三段
        df['lead_time_category'] = self._categorize_lead_time(df)

        # 7. 预订变更频率
        df['change_frequency'] = self._calc_change_frequency(df)

        # 8. 特殊请求密度
        df['special_request_density'] = self._calc_request_density(df)

        self.feature_names = [col for col in df.columns if col not in
                              ['is_canceled', 'reservation_status', 'reservation_status_date']]
        return df

    def _calc_stability_index(self, df):
        """预订稳定性指数 = 1 - (booking_changes / (lead_time + 1))"""
        lead_time = df['lead_time'].astype(float)
        changes = df['booking_changes'].astype(float)
        return 1 - (changes / (lead_time + 1))

    def _calc_total_spend(self, df):
        """总消费预估 = adr × (周末晚数 + 工作日晚数)"""
        return df['adr'].astype(float) * (
            df['stays_in_weekend_nights'].astype(float) +
            df['stays_in_week_nights'].astype(float)
        )

    def _calc_loyalty_decay(self, df):
        """忠诚度衰减指数：综合历史取消和重复客状态"""
        prev_cancel = df['previous_cancellations'].astype(float)
        is_repeated = df['is_repeated_guest'].astype(float)
        prev_not_cancel = df['previous_bookings_not_canceled'].astype(float)
        total_history = prev_cancel + prev_not_cancel + 1e-6
        cancel_rate = prev_cancel / total_history
        # 重复客人且取消率低 → 忠诚度高；非重复客人且取消率高 → 忠诚度低
        loyalty = is_repeated * (1 - cancel_rate) + (1 - is_repeated) * (0.5 - cancel_rate * 0.5)
        return np.clip(loyalty, 0, 1)

    def _calc_adr_per_person(self, df):
        """人均房价 = adr / 总人数"""
        total_people = (df['adults'].astype(float) +
                       df['children'].fillna(0).astype(float) +
                       df['babies'].astype(float))
        total_people = total_people.replace(0, 1)  # 避免除零
        return df['adr'].astype(float) / total_people

    def _calc_occupancy_pressure(self, df):
        """入住压力指数 = 总人数 / 总入住晚数"""
        total_nights = (df['stays_in_weekend_nights'].astype(float) +
                       df['stays_in_week_nights'].astype(float))
        total_nights = total_nights.replace(0, 1)
        total_people = (df['adults'].astype(float) +
                       df['children'].fillna(0).astype(float) +
                       df['babies'].astype(float))
        return total_people / total_nights

    def _categorize_lead_time(self, df):
        """提前期分类: 0=短期(<7天), 1=中期(7-30天), 2=长期(>30天)"""
        lead_time = df['lead_time'].astype(float)
        return pd.cut(lead_time, bins=[-np.inf, 7, 30, np.inf], labels=[0, 1, 2]).astype(float)

    def _calc_change_frequency(self, df):
        """预订变更频率 = 变更次数 / 提前期"""
        lead_time = df['lead_time'].astype(float).replace(0, 1)
        return df['booking_changes'].astype(float) / lead_time

    def _calc_request_density(self, df):
        """特殊请求数量密度 = 特殊请求数 / 总入住晚数"""
        total_nights = (df['stays_in_weekend_nights'].astype(float) +
                       df['stays_in_week_nights'].astype(float)).replace(0, 1)
        return df['total_of_special_requests'].astype(float) / total_nights

    def get_new_feature_names(self):
        """返回新增的特征名称列表"""
        return [
            'booking_stability_index',
            'total_estimated_spend',
            'loyalty_decay_index',
            'adr_per_person',
            'occupancy_pressure',
            'lead_time_category',
            'change_frequency',
            'special_request_density'
        ]

    def get_feature_descriptions(self):
        """返回各复合特征的业务含义描述"""
        return {
            'booking_stability_index': '预订稳定性指数(越接近1越稳定)',
            'total_estimated_spend': '总消费预估(ADR×入住晚数)',
            'loyalty_decay_index': '客户忠诚度衰减指数',
            'adr_per_person': '人均房价',
            'occupancy_pressure': '入住压力指数(人数/晚数)',
            'lead_time_category': '提前期分类(0短1中2长期)',
            'change_frequency': '预订变更频率',
            'special_request_density': '特殊请求密度'
        }
