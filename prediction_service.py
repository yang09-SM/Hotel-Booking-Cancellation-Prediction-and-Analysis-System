
import joblib
import pandas as pd
import numpy as np
import os
from shap_analyzer import SHAPAnalyzer
from feature_engineering import FeatureEngineer
from temporal_features import TemporalFeatureEngineer
from external_data import ExternalDataIntegrator


class PredictionService:
    def __init__(self, models_dir='models'):
        self.models = {}
        self.label_encoders = None
        self.scaler = None
        self.feature_names = None
        self.model_results = None
        self.shap_analyzer = SHAPAnalyzer(models_dir)  # 初始化SHAP分析器

        # 新增：融合模型
        self.ensemble_models = {}

        self.load_models(models_dir)
        self._init_ensemble_models()  # 初始化融合模型
    
    def load_models(self, models_dir):
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
                self.models[model_name] = joblib.load(os.path.join(models_dir, model_file))
            
            if os.path.exists(os.path.join(models_dir, 'label_encoders.pkl')):
                self.label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
            
            if os.path.exists(os.path.join(models_dir, 'scaler.pkl')):
                self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            
            if os.path.exists(os.path.join(models_dir, 'feature_names.pkl')):
                self.feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
            
            if os.path.exists(os.path.join(models_dir, 'model_results.pkl')):
                self.model_results = joblib.load(os.path.join(models_dir, 'model_results.pkl'))

    def _init_ensemble_models(self):
        """初始化融合模型"""
        try:
            from sklearn.ensemble import VotingClassifier, StackingClassifier

            # 获取可用的基模型列表
            base_estimators = []
            model_name_map = {}  # 用于映射名称

            for name, model in self.models.items():
                safe_name = name.replace(' ', '_').replace('-', '_')
                base_estimators.append((safe_name, model))
                model_name_map[safe_name] = name

            if len(base_estimators) >= 2:
                # 1. Voting Classifier (软投票)
                self.ensemble_models['Voting Ensemble'] = VotingClassifier(
                    estimators=base_estimators,
                    voting='soft'  # 使用概率投票
                )

                print(f"已初始化 {len(base_estimators)} 个基模型的融合策略")
        except Exception as e:
            print(f"融合模型初始化失败: {e}")
    
    def preprocess_input(self, booking_data):
        df = pd.DataFrame([booking_data])
        
        # 去除标签泄漏特征
        leak_features = ['reservation_status', 'reservation_status_date']
        for feature in leak_features:
            if feature in df.columns:
                df = df.drop(columns=[feature])
        
        # 确保有所有需要的特征
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
        
        # 处理缺失值
        if 'children' in df.columns:
            df['children'] = df['children'].fillna(0)
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('Unknown')
        if 'agent' in df.columns:
            df['agent'] = df['agent'].fillna(0)
        if 'company' in df.columns:
            df['company'] = df['company'].fillna(0)
        
        # ===== 新增: 外部数据整合（推理模式）=====
        integrator = ExternalDataIntegrator()
        # 将 DataFrame 转为字典进行单条数据整合
        booking_dict = df.iloc[0].to_dict()
        enriched_dict = integrator.enrich_booking_data(booking_dict)
        # 将整合后的外部特征合并回 DataFrame
        for key, value in enriched_dict.items():
            if key.startswith('weather_') or key.startswith('holiday_'):
                df[key] = value
        # ========================================
        
        # 类别编码
        if self.label_encoders:
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    # 处理未见过的类别
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    if 'Unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'Unknown')
                    df[col] = le.transform(df[col])

        # ===== 复合特征构造 =====
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        # ========================

        # ===== 新增: 时序聚合特征（推理模式，使用默认全局统计量）=====
        temp_engineer = TemporalFeatureEngineer(window_days=90)
        # 推理时使用默认全局统计量（冷启动处理）
        temp_engineer.global_stats = {
            'avg_cancel_rate': 0.37,
            'avg_adr': 100,
            'avg_lead_time': 100,
            'avg_stays': 3,
            'avg_changes': 0.1,
            'channel_cancel_rates': {},
            'hotel_cancel_rates': {}
        }
        df = temp_engineer.transform(df)
        # ============================================================

        # 标准化数值特征
        if self.scaler and self.feature_names:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            existing_numeric = [col for col in numeric_cols if col in df.columns]
            if existing_numeric:
                df[existing_numeric] = self.scaler.transform(df[existing_numeric])
        
        # 按训练时的特征顺序排列
        if self.feature_names:
            df = df[self.feature_names]
        
        return df
    
    def predict(self, booking_data, model_name='Random Forest'):
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}

        processed_data = self.preprocess_input(booking_data)
        model = self.models[model_name]

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0] if hasattr(model, 'predict_proba') else None

        # 获取SHAP解释（包含Top-5特征贡献度）
        explanation = None
        try:
            shap_result = self.shap_analyzer.explain_single_prediction(booking_data, model_name)
            if 'error' not in shap_result:
                # 只提取关键信息，避免返回过多数据
                explanation = {
                    'top_features': shap_result.get('top_features', []),
                    'base_value': shap_result.get('base_value'),
                    'prediction_confidence': shap_result.get('probability')
                }
        except Exception as e:
            print(f"SHAP解释生成失败: {e}")
            explanation = None

        result = {
            'model': model_name,
            'prediction': int(prediction),
            'prediction_label': 'Canceled' if prediction == 1 else 'Not Canceled',
            'probability': {
                'canceled': float(probability[1]) if probability is not None else None,
                'not_canceled': float(probability[0]) if probability is not None else None
            },
            'explanation': explanation  # 新增SHAP解释字段
        }

        return result

    def get_shap_explanation(self, booking_data, model_name='XGBoost'):
        """
        获取单样本预测的详细SHAP解释

        参数:
            booking_data: 原始预订数据字典
            model_name: 模型名称

        返回:
            dict: 包含完整SHAP分析结果的字典
        """
        return self.shap_analyzer.explain_single_prediction(booking_data, model_name)

    def get_global_feature_importance(self, model_name=None):
        """
        获取全局特征重要性（Summary Plot数据）

        参数:
            model_name: 模型名称（可选）

        返回:
            dict: 特征重要性数据
        """
        return self.shap_analyzer.get_global_feature_importance(model_name)

    def get_dependence_plot_data(self, feature_name, model_name='XGBoost'):
        """
        获取特征依赖关系数据（Dependence Plot）

        参数:
            feature_name: 特征名称
            model_name: 模型名称

        返回:
            dict: 依赖关系数据
        """
        return self.shap_analyzer.get_dependence_plot_data(feature_name, model_name)
    
    def predict_all_models(self, booking_data):
        results = {}

        # 各基模型预测
        for model_name in self.models.keys():
            results[model_name] = self.predict(booking_data, model_name)

        # 新增：融合模型预测
        try:
            results['Voting Ensemble'] = self.predict_ensemble(booking_data, 'voting')
            results['Stacking Ensemble'] = self.predict_ensemble(booking_data, 'stacking')
        except Exception as e:
            results['Ensemble Error'] = str(e)

        return results

    def predict_ensemble(self, booking_data, ensemble_type='voting'):
        """
        使用融合模型进行预测
        参数:
            booking_data: 预订数据字典
            ensemble_type: 'voting' 或 'stacking'
        返回:
            预测结果字典
        """
        processed_data = self.preprocess_input(booking_data)

        # 先获取各基模型的独立预测
        base_predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(processed_data)[0]
                proba = model.predict_proba(processed_data)[0] if hasattr(model, 'predict_proba') else None
                base_predictions[model_name] = {
                    'prediction': int(pred),
                    'probability': {
                        'canceled': float(proba[1]) if proba is not None else None,
                        'not_canceled': float(proba[0]) if proba is not None else None
                    }
                }
            except Exception as e:
                base_predictions[model_name] = {'error': str(e)}

        # === 融合策略 ===

        if ensemble_type == 'voting':
            result = self._voting_fusion(base_predictions, processed_data)
            result['method'] = 'Soft Voting'
        elif ensemble_type == 'stacking':
            result = self._stacking_fusion(base_predictions, processed_data)
            result['method'] = 'Weighted Stacking'
        else:
            return {'error': f'不支持的融合类型: {ensemble_type}'}

        result['base_predictions'] = base_predictions

        # 计算置信区间（基于基模型预测方差）
        probabilities = [
            bp.get('probability', {}).get('canceled')
            for bp in base_predictions.values()
            if isinstance(bp.get('probability'), dict) and bp['probability'].get('canceled') is not None
        ]

        if probabilities:
            result['confidence_interval'] = {
                'mean': float(np.mean(probabilities)),
                'std': float(np.std(probabilities)),
                'min': float(np.min(probabilities)),
                'max': float(np.max(probabilities)),
                'agreement': float(sum(1 for p in probabilities if p >= 0.5) / len(probabilities))  # 模型一致性
            }

        return result

    def _voting_fusion(self, base_predictions, processed_data):
        """软投票融合：对各模型的取消概率取平均"""
        probabilities = []

        for model_name, pred_info in base_predictions.items():
            prob = pred_info.get('probability', {}).get('canceled')
            if prob is not None:
                probabilities.append((model_name, prob))

        if not probabilities:
            return {'prediction': 0, 'prediction_label': 'Not Canceled', 'ensemble_probability': None}

        # 简单平均
        avg_prob = np.mean([p for _, p in probabilities])

        # 加权平均（可选：根据历史性能加权）
        # 这里使用等权简单平均
        final_prediction = 1 if avg_prob >= 0.5 else 0

        return {
            'prediction': int(final_prediction),
            'prediction_label': 'Canceled' if final_prediction == 1 else 'Not Canceled',
            'ensemble_probability': {
                'canceled': float(avg_prob),
                'not_canceled': float(1 - avg_prob)
            },
            'voting_details': [{'model': name, 'weight': 1.0/len(probabilities), 'probability': float(p)}
                              for name, p in probabilities]
        }

    def _stacking_fusion(self, base_predictions, processed_data):
        """
        堆叠融合：使用加权方案模拟元学习器效果
        权重可以基于各模型的历史 ROC-AUC 分配
        """
        # 如果有模型性能数据，使用性能加权的堆叠
        weights = {}
        if self.model_results:
            for model_name in base_predictions.keys():
                if model_name in self.model_results:
                    # 使用 AUC 作为权重
                    auc = self.model_results[model_name].get('roc_auc', 0.5)
                    weights[model_name] = auc
                else:
                    weights[model_name] = 0.5
        else:
            # 无历史数据时使用等权
            for model_name in base_predictions.keys():
                weights[model_name] = 1.0

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # 加权平均概率
        weighted_prob = 0.0
        stacking_details = []

        for model_name, pred_info in base_predictions.items():
            prob = pred_info.get('probability', {}).get('canceled')
            if prob is not None:
                w = weights.get(model_name, 0)
                weighted_prob += w * prob
                stacking_details.append({
                    'model': model_name,
                    'weight': float(w),
                    'probability': float(prob),
                    'weighted_contribution': float(w * prob)
                })

        final_prediction = 1 if weighted_prob >= 0.5 else 0

        return {
            'prediction': int(final_prediction),
            'prediction_label': 'Canceled' if final_prediction == 1 else 'Not Canceled',
            'ensemble_probability': {
                'canceled': float(weighted_prob),
                'not_canceled': float(1 - weighted_prob)
            },
            'stacking_details': stacking_details,
            'weights_used': {k: float(v) for k, v in weights.items()}
        }

    def get_available_ensembles(self):
        """获取可用的融合方法列表"""
        return ['Voting Ensemble', 'Stacking Ensemble']
    
    def get_model_performance(self):
        if self.model_results:
            return self.model_results
        return None
    
    def get_available_models(self):
        return list(self.models.keys())

# 全局预测服务实例
prediction_service = None

def get_prediction_service():
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service

