"""
时序聚合特征模块
基于客户历史行为窗口统计生成时间维度的衍生特征
用于捕捉客户行为的动态变化趋势
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TemporalFeatureEngineer:
    """时序聚合特征构造器"""

    def __init__(self, window_days=90):
        """
        初始化
        window_days: 历史行为窗口大小（默认90天/约3个月）
        """
        self.window_days = window_days
        self.global_stats = {}  # 全局统计量（用于冷启动填充）

    def fit(self, df):
        """
        计算全局统计量（基于全量历史数据）
        用于处理新客户（无历史记录）的冷启动问题
        """
        # 全局平均取消率
        self.global_stats['avg_cancel_rate'] = df['is_canceled'].mean()

        # 全局平均 ADR
        self.global_stats['avg_adr'] = df['adr'].mean()

        # 全局平均提前期
        self.global_stats['avg_lead_time'] = df['lead_time'].mean()

        # 全局平均入住晚数
        self.global_stats['avg_stays'] = (
            df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        ).mean()

        # 全局平均变更次数
        self.global_stats['avg_changes'] = df['booking_changes'].mean()

        # 各渠道的平均取消率
        channel_cancel = df.groupby('distribution_channel')['is_canceled'].mean()
        self.global_stats['channel_cancel_rates'] = channel_cancel.to_dict()

        # 各酒店类型的平均取消率
        hotel_cancel = df.groupby('hotel')['is_canceled'].mean()
        self.global_stats['hotel_cancel_rates'] = hotel_cancel.to_dict()

        return self

    def transform(self, df):
        """
        为每条记录计算时序聚合特征
        输入: 完整的预订DataFrame
        输出: 追加了时序特征的DataFrame
        """
        df = df.copy()

        # 确保有日期字段可用于排序（如果没有真实日期，用行号模拟时序）
        if 'arrival_date_year' in df.columns:
            # 构造排序用的伪日期（仅用于确定时序先后）
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df['_sort_date'] = (
                df['arrival_date_year'].astype(str) + '-' +
                df['arrival_date_month'].map(month_map).astype(str).str.zfill(2) + '-' +
                df['arrival_date_day_of_month'].astype(str).str.zfill(2)
            )
            df = df.sort_values('_sort_date').reset_index(drop=True)

        # ===== 时序聚合特征 =====

        # 1. 客户历史取消率（滚动窗口）
        df['customer_hist_cancel_rate'] = self._calc_rolling_cancel_rate(df)

        # 2. 客户历史平均 ADR（滚动窗口）
        df['customer_hist_avg_adr'] = self._calc_rolling_avg_adr(df)

        # 3. 客户预订频次（过去N条记录中的预订数）
        df['customer_booking_frequency'] = self._calc_booking_frequency(df)

        # 4. 客户历史平均提前期变化趋势
        df['customer_lead_time_trend'] = self._calc_lead_time_trend(df)

        # 5. 渠道风险系数（该渠道历史取消率 / 全局平均取消率）
        df['channel_risk_factor'] = self._calc_channel_risk(df)

        # 6. 酒店类型风险系数
        df['hotel_type_risk_factor'] = self._calc_hotel_risk(df)

        # 7. 客户行为一致性得分（最近行为与历史模式的偏差）
        df['behavior_consistency_score'] = self._calc_behavior_consistency(df)

        # 清理临时列
        if '_sort_date' in df.columns:
            df = df.drop(columns=['_sort_date'])

        return df

    def _calc_rolling_cancel_rate(self, df):
        """
        计算每个客户的滚动取消率
        使用 expanding window（截至当前行的所有历史记录）
        新客户使用全局均值填充
        """
        # 如果没有可区分客户的字段，用全局特征近似
        if 'country' in df.columns:
            # 按国家分组计算滚动取消率（作为客户代理分组）
            result = df.groupby('country')['is_canceled'].transform(
                lambda x: x.expanding(min_periods=1).mean().shift(1).fillna(
                    self.global_stats.get('avg_cancel_rate', 0.37)
                )
            )
        else:
            # 无分组信息，使用全局均值
            result = pd.Series([self.global_stats.get('avg_cancel_rate', 0.37)] * len(df))

        return result.fillna(self.global_stats.get('avg_cancel_rate', 0.37))

    def _calc_rolling_avg_adr(self, df):
        """客户历史平均ADR"""
        if 'country' in df.columns:
            result = df.groupby('country')['adr'].transform(
                lambda x: x.expanding(min_periods=1).mean().shift(1).fillna(
                    self.global_stats.get('avg_adr', 100)
                )
            )
        else:
            result = pd.Series([self.global_stats.get('avg_adr', 100)] * len(df))

        return result.fillna(self.global_stats.get('avg_adr', 100))

    def _calc_booking_frequency(self, df):
        """
        客户预订频次：过去N条同类型记录中的出现次数
        用 expanding count 近似
        """
        if 'country' in df.columns:
            result = df.groupby('country').cumcount()  # 该组内的累计出现次数
            # 归一化到 [0, 1] 范围
            max_count = result.max() if result.max() > 0 else 1
            result = result / max_count
        else:
            result = pd.Series([0.5] * len(df))  # 默认中等频次

        return result

    def _calc_lead_time_trend(self, df):
        """
        提前期变化趋势：最近的提前期 vs 历史平均提前期的比值
        > 1 表示近期提前期变长，< 1 表示变短
        """
        if 'country' in df.columns:
            hist_avg = df.groupby('country')['lead_time'].transform(
                lambda x: x.expanding(min_periods=3).mean().shift(1)
            )
            trend = df['lead_time'].astype(float) / hist_avg.replace(0, 1)
            # 裁剪极端值
            trend = np.clip(trend, 0.1, 10)
        else:
            trend = pd.Series([1.0] * len(df))  # 无变化

        return trend.fillna(1.0)

    def _calc_channel_risk(self, df):
        """
        渠道风险系数 = 该渠道历史取消率 / 全局平均取消率
        > 1 表示高风险渠道，< 1 表示低风险渠道
        """
        avg_cancel = self.global_stats.get('avg_cancel_rate', 0.37)
        if avg_cancel == 0:
            avg_cancel = 0.37

        if 'distribution_channel' in df.columns:
            channel_rates = self.global_stats.get('channel_cancel_rates', {})
            risk = df['distribution_channel'].map(lambda x: channel_rates.get(x, avg_cancel) / avg_cancel)
        else:
            risk = pd.Series([1.0] * len(df))

        return risk.fillna(1.0)

    def _calc_hotel_risk(self, df):
        """酒店类型风险系数（同理）"""
        avg_cancel = self.global_stats.get('avg_cancel_rate', 0.37)
        if avg_cancel == 0:
            avg_cancel = 0.37

        if 'hotel' in df.columns:
            hotel_rates = self.global_stats.get('hotel_cancel_rates', {})
            risk = df['hotel'].map(lambda x: hotel_rates.get(x, avg_cancel) / avg_cancel)
        else:
            risk = pd.Series([1.0] * len(df))

        return risk.fillna(1.0)

    def _calc_behavior_consistency(self, df):
        """
        行为一致性得分：
        综合多个维度判断当前行为是否与历史模式一致
        得分范围 [0, 1]，越高表示越一致（越稳定）
        """
        consistency = pd.Series([1.0] * len(df))

        # ADR 一致性
        if 'country' in df.columns and 'adr' in df.columns:
            hist_std = df.groupby('country')['adr'].transform(
                lambda x: x.expanding(min_periods=3).std().shift(1).fillna(50)
            )
            adr_deviation = abs(df['adr'].astype(float) - self._calc_rolling_avg_adr(df)) / (hist_std + 1)
            adr_consistency = 1 / (1 + adr_deviation)
            consistency = consistency * adr_consistency

        # 提前期一致性
        if 'country' in df.columns and 'lead_time' in df.columns:
            lead_trend = self._calc_lead_time_trend(df)
            lead_consistency = 1 / (1 + abs(lead_trend - 1))
            consistency = consistency * lead_consistency

        return np.clip(consistency.fillna(0.5), 0, 1)

    def get_feature_descriptions(self):
        """返回各时序特征的业务含义描述"""
        return {
            'customer_hist_cancel_rate': '客户历史取消率(滚动窗口)',
            'customer_hist_avg_adr': '客户历史平均房价',
            'customer_booking_frequency': '客户预订频次',
            'customer_lead_time_trend': '提前期变化趋势(>1变长)',
            'channel_risk_factor': '渠道风险系数(相对值)',
            'hotel_type_risk_factor': '酒店类型风险系数',
            'behavior_consistency_score': '行为一致性得分'
        }
