"""
动态超售策略推荐引擎
基于机器学习预测结果和房态数据，计算最优超售策略
帮助酒店在控制 walk 风险的前提下最大化收益
"""

import numpy as np
import pandas as pd
from datetime import datetime

class OverbookingEngine:
    """超售策略推荐引擎"""

    def __init__(self):
        # Walk 成本参数（每间被 walk 的客房的赔偿成本）
        self.walk_cost_per_room = 300  # 默认 ¥300/间

        # 空房成本参数（每间未售出客房的机会成本）
        # 通常为 ADR 的 60-80%
        self.vacancy_cost_ratio = 0.7

        # 风险偏好配置
        self.risk_profiles = {
            'conservative': {'max_walk_rate': 0.01, 'service_level': 0.99},   # 最多1% walk率
            'moderate': {'max_walk_rate': 0.03, 'service_level': 0.97},      # 最多3% walk率
            'aggressive': {'max_walk_rate': 0.05, 'service_level': 0.95}      # 最多5% walk率
        }

    def calculate_overbooking_recommendation(
        self,
        bookings_with_predictions,
        total_rooms_available,
        avg_adr=None,
        risk_preference='moderate'
    ):
        """
        计算超售建议

        参数:
            bookings_with_predictions: 包含预测概率的预订列表
                格式: [{'booking_id': int, 'cancel_probability': float, 'adr': float, ...}, ...]
            total_rooms_available: 当日可用房间总数
            avg_adr: 平均房价（用于计算机会成本），如果为None则从预订中计算
            risk_preference: 风险偏好 ('conservative' | 'moderate' | 'aggressive')

        返回:
            包含建议和详细分析的字典
        """
        if not bookings_with_predictions:
            return {
                'recommendation': 0,
                'total_bookings': 0,
                'rooms_available': total_rooms_available,
                'message': '无预订数据',
                'risk_analysis': {}
            }

        # 转换为 DataFrame 方便计算
        df = pd.DataFrame(bookings_with_predictions)

        # 基础统计
        n_bookings = len(df)
        confirmed_count = n_bookings  # 当前确认的预订数
        avg_adr = avg_adr or df['adr'].mean() if 'adr' in df.columns else 100

        # 各预订的取消概率
        cancel_probs = df['cancel_probability'].values if 'cancel_probability' in df.columns else np.full(n_bookings, 0.37)

        # === 核心算法：基于期望值优化的超售量计算 ===

        # 方法：二分搜索找到最优超售量
        # 目标：在满足风险约束的前提下最大化期望收益

        risk_profile = self.risk_profiles.get(risk_preference, self.risk_profiles['moderate'])
        max_walk_rate = risk_profile['max_walk_rate']

        best_overbook = 0
        best_expected_profit = -np.inf
        analysis_results = []

        # 尝试不同的超售数量（0 到 房间数的50%）
        max_overbook_candidates = max(0, int(total_rooms_available * 0.5))

        for overbook_qty in range(0, max_overbook_candidates + 1):
            total_accepted = n_bookings + overbook_qty

            # 计算期望取消数
            expected_cancellations = np.sum(cancel_probs)

            # 期望净入住数 = 接受预订数 - 期望取消数
            expected_net_checkins = total_accepted - expected_cancellations

            # 期望超额预订数（如果净入住 > 可用房）
            expected_overbooking = max(0, expected_net_checkins - total_rooms_available)

            # Walk 概率（使用正态近似或蒙特卡洛模拟）
            walk_prob = self._estimate_walk_probability(
                cancel_probs,
                total_rooms_available,
                overbook_qty
            )

            # 检查风险约束
            if walk_prob > max_walk_rate:
                continue

            # 计算期望收益
            revenue_per_room = avg_adr
            vacancy_cost = avg_adr * self.vacancy_cost_ratio

            # 期望入住收益
            expected_revenue = min(expected_net_checkins, total_rooms_available) * revenue_per_room

            # 期望空房损失（如果有空房的话）
            expected_vacancy_loss = max(0, total_rooms_available - expected_net_checkins) * vacancy_cost

            # 期望 Walk 成本
            expected_walk_cost = expected_overbooking * self.walk_cost_per_room

            # 净期望利润
            expected_profit = expected_revenue - expected_vacancy_loss - expected_walk_cost

            analysis_results.append({
                'overbook_quantity': overbook_qty,
                'total_accepted': total_accepted,
                'expected_cancellations': round(expected_cancellations, 1),
                'expected_net_checkins': round(expected_net_checkins, 1),
                'expected_walk_prob': round(walk_prob, 4),
                'expected_overbooking': round(expected_overbooking, 2),
                'expected_revenue': round(expected_revenue, 2),
                'expected_vacancy_loss': round(expected_vacancy_loss, 2),
                'expected_walk_cost': round(expected_walk_cost, 2),
                'expected_profit': round(expected_profit, 2),
                'passes_risk_constraint': walk_prob <= max_walk_rate
            })

            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_overbook = overbook_qty

        # 选择最优方案
        best_result = None
        for r in analysis_results:
            if r['overbook_quantity'] == best_overbook:
                best_result = r
                break

        # 如果没有可行方案（所有方案都超出风险约束），选择最保守的
        if best_result is None:
            feasible = [r for r in analysis_results if r['passes_risk_constraint']]
            if feasible:
                best_result = feasible[0]
                best_overbook = best_result['overbook_quantity']
            else:
                best_result = {'overbook_quantity': 0, 'expected_profit': 0}
                best_overbook = 0

        # 识别高风险订单（取消概率 > 70%）
        high_risk_threshold = 0.70
        if 'cancel_probability' in df.columns:
            high_risk_mask = df['cancel_probability'] >= high_risk_threshold
            high_risk_bookings = df[high_risk_mask][['booking_id', 'cancel_probability', 'adr']].to_dict('records')
        else:
            high_risk_bookings = []

        # 与不超售的对比
        baseline_expected_cancellations = np.sum(cancel_probs)
        baseline_net_checkins = n_bookings - baseline_expected_cancellations
        baseline_revenue = min(baseline_net_checkins, total_rooms_available) * avg_adr
        baseline_vacancy_loss = max(0, total_rooms_available - baseline_net_checkins) * avg_adr * self.vacancy_cost_ratio
        baseline_profit = baseline_revenue - baseline_vacancy_loss

        result = {
            'recommendation': best_overbook,
            'total_bookings': n_bookings,
            'rooms_available': total_rooms_available,
            'total_acceptable': n_bookings + best_overbook,
            'avg_adr': round(float(avg_adr), 2),
            'risk_preference': risk_preference,
            'risk_profile_used': risk_profile,

            # 核心建议
            'suggested_overbooking': best_overbook,
            'expected_additional_revenue': round(best_expected_profit - baseline_profit, 2) if best_result else 0,
            'revenue_increase_pct': round(
                ((best_expected_profit - baseline_profit) / max(baseline_profit, 1)) * 100, 1
            ) if best_result else 0,

            # 风险评估
            'walk_risk': round(best_result.get('expected_walk_prob', 0), 4) if best_result else 0,
            'is_within_risk_limit': True,

            # 详细分析
            'analysis': best_result,
            'all_candidates': sorted(
                [r for r in analysis_results if r['passes_risk_constraint']],
                key=lambda x: x['expected_profit'],
                reverse=True
            )[:10],  # 返回 top-10 可行方案

            # 高风险订单
            'high_risk_threshold': high_risk_threshold,
            'high_risk_booking_count': len(high_risk_bookings),
            'high_risk_bookings': high_risk_bookings[:20],  # 最多返回20条

            # 对比基准线
            'baseline': {
                'no_overbooking_profit': round(baseline_profit, 2),
                'expected_cancellations': round(baseline_expected_cancellations, 1),
                'expected_net_checkins': round(baseline_net_checkins, 1)
            },

            'generated_at': datetime.now().isoformat()
        }

        return result

    def _estimate_walk_probability(self, cancel_probs, rooms_available, overbook_qty):
        """
        估算 Walk 概率
        使用蒙特卡洛模拟（简化版：正态近似）
        """
        n = len(cancel_probs)
        total_accepted = n + overbook_qty

        # 取消数的期望和方差
        p_cancel = cancel_probs
        E_cancellations = np.sum(p_cancel)
        Var_cancellations = np.sum(p_cancel * (1 - p_cancel))  # 二项分布方差之和

        # 净入住数 = 总接受数 - 取消数
        E_net = total_accepted - E_cancellations
        Var_net = Var_cancellations  # 净入住方差 ≈ 取消方差

        # Walk 发生条件: 净入住 > 可用房
        # P(Walk) = P(Net > Rooms)
        # 使用正态近似
        if Var_net > 0:
            std_net = np.sqrt(Var_net)
            z_score = (rooms_available - E_net) / std_net
            # P(Net > Rooms) = P(Z > z_score) = 1 - Phi(z_score)
            from scipy.stats import norm
            walk_prob = 1 - norm.cdf(z_score)
        else:
            # 无方差时确定性判断
            walk_prob = 1.0 if E_net > rooms_available else 0.0

        return max(0, min(1, walk_prob))

    def get_scenario_comparison(self, bookings_with_predictions, total_rooms, avg_adr=None):
        """
        对比不同风险偏好的超售建议
        返回三种策略的对比结果
        """
        results = {}
        for profile in ['conservative', 'moderate', 'aggressive']:
            results[profile] = self.calculate_overbooking_recommendation(
                bookings_with_predictions,
                total_rooms,
                avg_adr,
                risk_preference=profile
            )
        return results


class RevenueOptimizer:
    """收益优化器（扩展功能）"""

    def __init__(self):
        self.engine = OverbookingEngine()

    def analyze_date(self, date_str, bookings, room_inventory, avg_adr=None):
        """
        分析特定日期的超售策略
        """
        return self.engine.calculate_overbooking_recommendation(
            bookings_with_predictions=bookings,
            total_rooms_available=room_inventory,
            avg_adr=avg_adr,
            risk_preference='moderate'
        )
