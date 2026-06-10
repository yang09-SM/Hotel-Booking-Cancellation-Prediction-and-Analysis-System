import shap
import joblib
import os
import base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


class SHAPAnalyzer:
    """SHAP可解释性分析器"""

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.explainers = {}
        self.shap_values_cache = {}
        self.models = {}
        self.label_encoders = None
        self.scaler = None
        self.feature_names = None
        self.background_data = None

        # 加载模型和预处理组件
        self._load_models_and_build_explainers()

    def _load_models_and_build_explainers(self):
        """加载所有模型和预处理组件，构建SHAP解释器"""
        if not os.path.exists(self.models_dir):
            print(f"模型目录不存在: {self.models_dir}")
            return

        # 加载所有模型文件
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]

        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
            try:
                self.models[model_name] = joblib.load(os.path.join(self.models_dir, model_file))
                print(f"成功加载模型: {model_name}")
            except Exception as e:
                print(f"加载模型 {model_name} 失败: {e}")

        # 加载预处理组件
        if os.path.exists(os.path.join(self.models_dir, 'label_encoders.pkl')):
            self.label_encoders = joblib.load(os.path.join(self.models_dir, 'label_encoders.pkl'))

        if os.path.exists(os.path.join(self.models_dir, 'scaler.pkl')):
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))

        if os.path.exists(os.path.join(self.models_dir, 'feature_names.pkl')):
            self.feature_names = joblib.load(os.path.join(self.models_dir, 'feature_names.pkl'))

        # 生成背景数据（用于KernelExplainer）
        self._generate_background_data()

        # 为每个模型构建解释器
        self._build_explainers()

    def _generate_background_data(self):
        """生成背景数据用于SHAP解释"""
        if self.feature_names is None:
            return

        # 使用零均值或随机生成的背景数据
        n_samples = 100
        background_data = np.zeros((n_samples, len(self.feature_names)))

        # 为数值特征添加一些变化
        for i in range(n_samples):
            for j, feature in enumerate(self.feature_names):
                # 添加一些随机噪声使背景数据更真实
                background_data[i][j] = np.random.normal(0, 0.5)

        self.background_data = background_data

    def _build_explainers(self):
        """为每个模型构建合适的SHAP解释器"""
        tree_models = ['XGBoost', 'LightGBM']

        for model_name, model in self.models.items():
            try:
                if model_name in tree_models:
                    # 树模型使用TreeExplainer（更快更准确）
                    self.explainers[model_name] = shap.TreeExplainer(model)
                    print(f"为 {model_name} 创建了 TreeExplainer")
                else:
                    # 其他模型使用KernelExplainer（采样加速）
                    if self.background_data is not None:
                        # 使用预测函数而非模型对象
                        predict_fn = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
                        self.explainers[model_name] = shap.KernelExplainer(
                            predict_fn,
                            self.background_data[:50]  # 使用50个背景样本加速
                        )
                        print(f"为 {model_name} 创建了 KernelExplainer")
            except Exception as e:
                print(f"为 {model_name} 创建解释器失败: {e}")

    def get_global_feature_importance(self, model_name=None):
        """
        获取全局特征重要性（Summary Plot 数据）

        参数:
            model_name: 模型名称，如果为None则返回第一个可用模型的结果

        返回:
            dict: {
                feature_names: [],
                importance_values: [],
                feature_importance_ranking: dict
            }
        """
        if model_name is None:
            model_name = list(self.explainers.keys())[0] if self.explainers else None

        if model_name not in self.explainers:
            return {'error': f'模型 {model_name} 的解释器不可用'}

        # 检查缓存
        cache_key = f"{model_name}_global"
        if cache_key in self.shap_values_cache:
            return self.shap_values_cache[cache_key]

        try:
            explainer = self.explainers[model_name]
            model = self.models[model_name]

            # 计算SHAP值
            if isinstance(explainer, shap.TreeExplainer):
                # TreeExplainer可以直接处理背景数据或使用训练数据的子集
                if self.background_data is not None:
                    shap_values = explainer.shap_values(self.background_data)
                else:
                    # 如果没有背景数据，生成简单的测试数据
                    test_data = np.zeros((10, len(self.feature_names)))
                    shap_values = explainer.shap_values(test_data)
            else:
                # KernelExplainer需要显式调用shap_values方法
                if self.background_data is not None:
                    shap_values = explainer.shap_values(self.background_data)
                else:
                    return {'error': '无法计算SHAP值：缺少背景数据'}

            # 确保shap_values是数组格式
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 对于二分类问题，取正类的SHAP值
            elif isinstance(shap_values, shap.Explanation):
                shap_values = shap_values.values

            # 计算特征重要性（绝对值的平均）
            if len(shap_values.shape) > 1:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            else:
                mean_abs_shap = np.abs(shap_values)

            # 特征重要性排序
            feature_importance_idx = np.argsort(mean_abs_shap)[::-1]
            sorted_features = [self.feature_names[i] for i in feature_importance_idx]
            sorted_importance = [mean_abs_shap[i] for i in feature_importance_idx]

            result = {
                'feature_names': sorted_features,
                'importance_values': sorted_importance,
                'feature_importance_ranking': {
                    feature: float(importance)
                    for feature, importance in zip(sorted_features, sorted_importance)
                }
            }

            # 缓存结果
            self.shap_values_cache[cache_key] = result

            return result

        except Exception as e:
            return {'error': f'计算特征重要性失败: {str(e)}'}

    def explain_single_prediction(self, booking_data, model_name='XGBoost'):
        """
        单样本预测解释（Force Plot / Waterfall Plot 数据）

        参数:
            booking_data: 原始预订数据字典
            model_name: 模型名称

        返回:
            dict: {
                base_value: float,
                prediction: int,
                probability: float,
                feature_contributions: [
                    {
                        feature: str,
                        contribution: float,
                        value: str
                    }
                ],
                top_features: [...]  # Top-5 关键特征
            }
        """
        if model_name not in self.explainers:
            return {'error': f'模型 {model_name} 的解释器不可用'}

        if model_name not in self.models:
            return {'error': f'模型 {model_name} 不存在'}

        try:
            model = self.models[model_name]
            explainer = self.explainers[model_name]

            # 预处理输入数据
            processed_data = self._preprocess_input(booking_data)

            # 获取预测结果
            prediction = model.predict(processed_data)[0]
            probability = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0]
                probability = float(proba[1])  # 取取消概率

            # 计算SHAP值
            if isinstance(explainer, shap.TreeExplainer):
                shap_values = explainer.shap_values(processed_data)
            else:
                shap_values = explainer.shap_values(processed_data)

            # 处理不同格式的SHAP值输出
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 二分类取正类
            elif isinstance(shap_values, shap.Explanation):
                shap_values = shap_values.values
                base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0.5
            else:
                base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0.5

            if not isinstance(base_value, (int, float)):
                if isinstance(base_value, np.ndarray):
                    base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
                else:
                    base_value = 0.5

            # 获取特征贡献度
            shap_values_flat = shap_values.flatten() if hasattr(shap_values, 'flatten') else shap_values

            feature_contributions = []
            for i, feature in enumerate(self.feature_names):
                contribution = float(shap_values_flat[i])
                original_value = booking_data.get(feature, processed_data.iloc[0, i])

                # 格式化原始值以便显示
                if isinstance(original_value, (int, float)):
                    value_str = f"{original_value:.2f}"
                else:
                    value_str = str(original_value)

                feature_contributions.append({
                    'feature': feature,
                    'contribution': contribution,
                    'value': value_str
                })

            # 按绝对贡献度排序，获取Top-5关键特征
            sorted_contributions = sorted(
                feature_contributions,
                key=lambda x: abs(x['contribution']),
                reverse=True
            )
            top_features = sorted_contributions[:5]

            result = {
                'base_value': float(base_value),
                'prediction': int(prediction),
                'probability': probability,
                'feature_contributions': feature_contributions,
                'top_features': top_features
            }

            return result

        except Exception as e:
            return {'error': f'预测解释失败: {str(e)}'}

    def get_dependence_plot_data(self, feature_name, model_name='XGBoost'):
        """
        获取特征依赖关系数据（Dependence Plot）

        参数:
            feature_name: 特征名称
            model_name: 模型名称

        返回:
            dict: {
                feature_values: [],
                shap_values: [],
                interaction_feature: str
            }
        """
        if model_name not in self.explainers:
            return {'error': f'模型 {model_name} 的解释器不可用'}

        if feature_name not in self.feature_names:
            return {'error': f'特征 {feature_name} 不存在'}

        # 检查缓存
        cache_key = f"{model_name}_dependence_{feature_name}"
        if cache_key in self.shap_values_cache:
            return self.shap_values_cache[cache_key]

        try:
            explainer = self.explainers[model_name]

            # 计算SHAP值
            if self.background_data is not None:
                if isinstance(explainer, shap.TreeExplainer):
                    all_shap_values = explainer.shap_values(self.background_data)
                else:
                    all_shap_values = explainer.shap_values(self.background_data)

                # 处理格式
                if isinstance(all_shap_values, list):
                    all_shap_values = all_shap_values[1]
                elif isinstance(all_shap_values, shap.Explanation):
                    all_shap_values = all_shap_values.values

                # 找到目标特征的索引
                feature_idx = self.feature_names.index(feature_name)

                # 提取该特征的值和对应的SHAP值
                feature_values = self.background_data[:, feature_idx]
                shap_vals_for_feature = all_shap_values[:, feature_idx]

                # 自动检测交互特征（选择与目标特征交互最强的特征）
                interaction_feature = self._find_interaction_feature(
                    all_shap_values, feature_idx
                )

                result = {
                    'feature_values': feature_values.tolist(),
                    'shap_values': shap_vals_for_feature.tolist(),
                    'interaction_feature': interaction_feature
                }

                # 缓存结果
                self.shap_values_cache[cache_key] = result

                return result
            else:
                return {'error': '缺少背景数据'}

        except Exception as e:
            return {'error': f'获取依赖关系数据失败: {str(e)}'}

    def _find_interaction_feature(self, shap_values, target_feature_idx):
        """自动查找与目标特征交互最强的特征"""
        try:
            # 计算与其他特征的交互强度
            interactions = []
            for i in range(len(self.feature_names)):
                if i != target_feature_idx:
                    # 使用简单的相关性度量
                    correlation = np.abs(np.corrcoef(
                        shap_values[:, target_feature_idx],
                        shap_values[:, i]
                    )[0, 1])
                    interactions.append((self.feature_names[i], correlation))

            # 选择交互最强的特征
            if interactions:
                interactions.sort(key=lambda x: x[1], reverse=True)
                return interactions[0][0]

            return self.feature_names[0]  # 默认返回第一个特征
        except:
            return self.feature_names[0] if self.feature_names else 'unknown'

    def generate_summary_plot_base64(self, model_name='XGBoost', max_display=20):
        """
        生成 Summary Plot 的 Base64 编码图片

        参数:
            model_name: 模型名称
            max_display: 显示的最大特征数量

        返回:
            str: Base64编码的图片字符串
        """
        if model_name not in self.explainers:
            return None

        try:
            explainer = self.explainers[model_name]

            # 计算SHAP值
            if self.background_data is not None:
                if isinstance(explainer, shap.TreeExplainer):
                    shap_values = explainer.shap_values(self.background_data)
                else:
                    shap_values = explainer.shap_values(self.background_data)

                # 处理格式
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                elif isinstance(shap_values, shap.Explanation):
                    shap_values = shap_values.values

                # 创建图形
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values,
                    self.background_data,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False
                )

                # 转换为Base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

                return img_base64

            return None

        except Exception as e:
            print(f"生成Summary Plot失败: {e}")
            return None

    def generate_force_plot_base64(self, booking_data, model_name='XGBoost'):
        """
        生成 Force Plot 的 Base64 编码图片

        参数:
            booking_data: 原始预订数据字典
            model_name: 模型名称

        返回:
            str: Base64编码的图片字符串
        """
        if model_name not in self.explainers:
            return None

        try:
            model = self.models[model_name]
            explainer = self.explainers[model_name]

            # 预处理输入数据
            processed_data = self._preprocess_input(booking_data)

            # 计算SHAP值
            if isinstance(explainer, shap.TreeExplainer):
                shap_values = explainer.shap_values(processed_data)
            else:
                shap_values = explainer.shap_values(processed_data)

            # 处理格式
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif isinstance(shap_values, shap.Explanation):
                shap_values = shap_values.values

            # 创建图形
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                expected_value,
                shap_values[0],
                processed_data.iloc[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )

            # 转换为Base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return img_base64

        except Exception as e:
            print(f"生成Force Plot失败: {e}")
            return None

    def _preprocess_input(self, booking_data):
        """预处理输入数据（复用prediction_service的逻辑）"""
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

    def clear_cache(self):
        """清除SHAP缓存"""
        self.shap_values_cache.clear()


# 全局SHAP分析器实例
shap_analyzer_instance = None


def get_shap_analyzer():
    """获取全局SHAP分析器实例（单例模式）"""
    global shap_analyzer_instance
    if shap_analyzer_instance is None:
        shap_analyzer_instance = SHAPAnalyzer()
    return shap_analyzer_instance
