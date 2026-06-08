
import joblib
import pandas as pd
import numpy as np
import os

class PredictionService:
    def __init__(self, models_dir='models'):
        self.models = {}
        self.label_encoders = None
        self.scaler = None
        self.feature_names = None
        self.model_results = None
        
        self.load_models(models_dir)
    
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
    
    def predict(self, booking_data, model_name='Random Forest'):
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        processed_data = self.preprocess_input(booking_data)
        model = self.models[model_name]
        
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'model': model_name,
            'prediction': int(prediction),
            'prediction_label': 'Canceled' if prediction == 1 else 'Not Canceled',
            'probability': {
                'canceled': float(probability[1]) if probability is not None else None,
                'not_canceled': float(probability[0]) if probability is not None else None
            }
        }
        
        return result
    
    def predict_all_models(self, booking_data):
        results = {}
        
        for model_name in self.models.keys():
            results[model_name] = self.predict(booking_data, model_name)
        
        return results
    
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

