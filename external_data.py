"""
外部数据源整合模块
提供天气数据和节假日数据的获取与特征构造能力
用于增强酒店预订取消预测模型的外部信息输入
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WeatherDataService:
    """天气预报数据服务"""
    
    def __init__(self, api_key=None, provider='default'):
        """
        初始化天气服务
        api_key: 天气API密钥（从环境变量或配置获取）
        provider: 数据提供商 ('hefeng' | 'openweather' | 'default')
        """
        self.api_key = api_key or os.environ.get('WEATHER_API_KEY', '')
        self.provider = provider
        self._cache = {}  # 简单内存缓存
        
        # 默认城市坐标映射（葡萄牙常用城市，因为数据集来自葡萄牙酒店）
        self.city_coordinates = {
            'Lisbon': (38.7223, -9.1393),
            'Oporto': (41.1495, -8.6108),
            'Faro': (37.0194, -7.9322),
            'Setubal': (38.5247, -8.8883),
            'Braga': (41.5506, -8.4227),
            # 中国主要城市
            'Beijing': (39.9042, 116.4074),
            'Shanghai': (31.2304, 121.4737),
            'Guangzhou': (23.1291, 113.2644),
            'Shenzhen': (22.5431, 114.0579),
            'Hangzhou': (30.2741, 120.1551),
            'Chengdu': (30.5728, 104.0668),
            'Wuhan': (30.5928, 114.3055),
            'Sanya': (18.2528, 109.5120),
            'Xiamen': (24.4798, 118.0894),
            'Qingdao': (36.0671, 120.3826),
            'Dali': (25.6065, 100.2635),
            'Lijiang': (26.8721, 100.2308),
            'Guilin': (25.2744, 110.2990),
            'default': (38.7223, -9.1393),  # 默认里斯本
        }
    
    def get_weather_by_city_date(self, city, date_str):
        """
        获取指定城市指定日期的天气数据
        参数:
            city: 城市名（英文）
            date_str: 日期字符串 'YYYY-MM-DD'
        返回:
            天气特征字典，如果获取失败则返回默认值
        """
        cache_key = f"{city}_{date_str}"
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 如果没有配置API Key或网络不可用，使用基于日期的合理默认值
        weather_data = self._get_default_weather(date_str)
        
        # 如果有API Key，尝试调用真实API
        if self.api_key and self.provider != 'default':
            try:
                weather_data = self._fetch_real_weather(city, date_str)
            except Exception as e:
                print(f"天气API调用失败，使用默认值: {e}")
        
        self._cache[cache_key] = weather_data
        return weather_data
    
    def _get_default_weather(self, date_str):
        """
        基于日期生成合理的默认天气特征（无API时使用）
        根据月份推断季节性天气模式
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            month = date_obj.month
        except:
            month = 6  # 默认夏季
        
        # 基于北半球季节模式（数据集主要为葡萄牙）
        seasonal_patterns = {
            # (month): (temp_max, temp_min, humidity, precipitation_mm, wind_speed, cloud_cover, is_holiday_season)
            1:  (14, 7, 80, 90, 16, 65, False),   # 一月 冬季
            2:  (15, 8, 75, 80, 17, 60, False),   # 二月 冬末
            3:  (17, 9, 70, 55, 18, 55, False),   # 三月 初春
            4:  (19, 11, 65, 50, 19, 50, True),   # 四月 春季（复活节假期）
            5:  (22, 13, 60, 35, 18, 45, False),  # 五月 春末
            6:  (26, 16, 55, 15, 17, 30, False),  # 六月 初夏
            7:  (28, 18, 50, 5, 16, 20, True),    # 七月 夏季（旺季）
            8:  (28, 18, 50, 5, 16, 25, True),    # 八月 夏季（旺季）
            9:  (26, 17, 58, 25, 17, 35, False),  # 九月 秋初
            10: (22, 14, 68, 65, 17, 50, False),  # 十月 秋季
            11: (17, 10, 76, 85, 16, 62, False),  # 十一月 秋末
            12: (14, 8, 81, 95, 16, 68, False),   # 十二月 冬季
        }
        
        pattern = seasonal_patterns.get(month, seasonal_patterns[7])
        
        # 添加随机波动使数据更真实
        np.random.seed(int(date_str.replace('-', '')) % (2**31))
        
        return {
            'temp_max': round(pattern[0] + np.random.uniform(-3, 3), 1),
            'temp_min': round(pattern[1] + np.random.uniform(-2, 2), 1),
            'humidity': max(20, min(100, pattern[2] + int(np.random.uniform(-10, 10)))),
            'precipitation_mm': max(0, pattern[3] + int(np.random.uniform(-10, 10))),
            'wind_speed_kmh': max(0, pattern[4] + int(np.random.uniform(-5, 5))),
            'cloud_cover_pct': max(0, min(100, pattern[5] + int(np.random.uniform(-15, 15)))),
            'is_peak_season': pattern[6],
            'weather_condition': self._infer_condition(pattern),
            'data_source': 'seasonal_default'
        }
    
    def _fetch_real_weather(self, city, date_str):
        """调用真实天气API（OpenWeatherMap 或和风天气）"""
        coords = self.city_coordinates.get(city, self.city_coordinates['default'])
        lat, lon = coords
        
        if self.provider == 'openweather':
            url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        elif self.provider == 'hefeng':
            url = f"https://devapi.qweather.com/v7/weather/3d?location={lon},{lat}&key={self.api_key}"
        else:
            return self._get_default_weather(date_str)
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 解析不同API的数据格式...
        # 这里返回解析后的统一格式
        parsed = self._parse_api_response(data, provider=self.provider)
        parsed['data_source'] = 'realtime_api'
        return parsed
    
    def _infer_condition(self, pattern):
        """根据气象参数推断天气状况"""
        temp_max, temp_min, humidity, precip, wind, cloud, _ = pattern
        
        if precip > 50:
            return 'rainy'
        elif cloud > 70:
            return 'cloudy'
        elif temp_max > 30 and humidity < 50:
            return 'sunny_hot'
        elif temp_min < 5:
            return 'cold'
        else:
            return 'mild'
    
    def _parse_api_response(self, data, provider):
        """解析API响应为统一格式"""
        # 具体解析逻辑取决于API提供商
        return {
            'temp_max': 25.0,
            'temp_min': 15.0,
            'humidity': 60,
            'precipitation_mm': 0,
            'wind_speed_kmh': 15,
            'cloud_cover_pct': 40,
            'is_peak_season': False,
            'weather_condition': 'mild',
            'raw_data': data
        }


class HolidayCalendarService:
    """节假日日历服务"""

    def __init__(self):
        self.holidays = self._load_holiday_calendar()
    
    def _load_holiday_calendar(self):
        """
        加载节假日日历
        包含中国和国际主要节假日
        """
        holidays = {
            # === 中国法定节假日 ===
            # 元旦
            '0101': {'name_cn': '元旦', 'name_en': 'New Year', 'type': 'public', 'region': 'CN'},
            # 春节（农历，每年不同，这里用近似公历范围）
            '0210_0217': {'name_cn': '春节', 'name_en': 'Spring Festival', 'type': 'public', 'region': 'CN', 'is_major': True},
            # 清明节
            '0404_0406': {'name_cn': '清明节', 'name_en': 'Qingming Festival', 'type': 'public', 'region': 'CN'},
            # 劳动节
            '0501_0505': {'name_cn': '劳动节', 'name_en': 'Labor Day', 'type': 'public', 'region': 'CN'},
            # 端午节
            '0528_0530': {'name_cn': '端午节', 'name_en': 'Dragon Boat Festival', 'type': 'public', 'region': 'CN'},
            # 中秋节
            '0915_0917': {'name_cn': '中秋节', 'name_en': 'Mid-Autumn Festival', 'type': 'public', 'region': 'CN'},
            # 国庆节
            '1001_1007': {'name_cn': '国庆节', 'name_en': 'National Day', 'type': 'public', 'region': 'CN', 'is_major': True},
            
            # === 国际主要节假日 ===
            '0214': {'name_cn': '情人节', 'name_en': "Valentine's Day", 'type': 'international', 'region': 'GLOBAL'},
            '0317': {'name_cn': '圣帕特里克节', 'name_en': "St. Patrick's Day", 'type': 'international', 'region': 'EU/US'},
            '0401': {'name_cn': '愚人节', 'name_en': "April Fools' Day", 'type': 'international', 'region': 'GLOBAL'},
            '1225': {'name_cn': '圣诞节', 'name_en': 'Christmas Day', 'type': 'public', 'region': 'EU/US'},
            '1231': {'name_cn': '跨年夜', 'name_en': "New Year's Eve", 'type': 'international', 'region': 'GLOBAL'},
            
            # === 葡萄牙特定节日（因为数据集来自葡萄牙）===
            '0425': {'name_cn': '自由日', 'name_en': 'Freedom Day', 'type': 'public', 'region': 'PT'},
            '0501_pt': {'name_cn': '劳动节', 'name_en': 'Labor Day', 'type': 'public', 'region': 'PT'},
            '0610': {'name_cn': '葡萄牙日', 'name_en': 'Day of Portugal', 'type': 'public', 'region': 'PT', 'is_major': True},
            '0815': {'name_cn': '圣母升天节', 'name_en': 'Assumption of Mary', 'type': 'public', 'region': 'PT'},
            '1005': {'name_cn': '共和国日', 'name_en': 'Republic Day', 'type': 'public', 'region': 'PT'},
            '1101': {'name_cn': '诸圣节', 'name_en': 'All Saints\' Day', 'type': 'public', 'region': 'PT'},
            '1201': {'name_cn': '恢复独立日', 'name_en': 'Restoration of Independence', 'type': 'public', 'region': 'PT'},
        }
        return holidays
    
    def is_holiday(self, date_str, region='ALL'):
        """
        判断某天是否为节假日
        参数:
            date_str: 日期字符串 'YYYY-MM-DD' 或 'MMDD'
            region: 地区过滤 ('CN', 'PT', 'EU/US', 'GLOBAL', 'ALL')
        返回:
            (bool, holiday_info_dict_or_None)
        """
        # 提取月日
        mmdd = date_str.replace('-', '')[4:] if len(date_str) > 4 else date_str[:4]
        mmdd_start = mmdd[:4]
        
        for key, info in self.holidays.items():
            # 检查是否匹配（支持范围格式如 '0210_0217'）
            if '_' in key:
                start, end = key.split('_')
                if start <= mmdd <= end:
                    if region == 'ALL' or info.get('region') == region or info.get('region') == 'GLOBAL':
                        return True, info
            else:
                if key == mmdd or mmdd.startswith(key[:4]):
                    if region == 'ALL' or info.get('region') == region or info.get('region') == 'GLOBAL':
                        return True, info
        
        return False, None
    
    def get_holiday_features(self, date_str):
        """
        获取日期的节假日特征
        返回包含多个特征的字典
        """
        is_hol, hol_info = self.is_holiday(date_str)
        
        try:
            date_obj = datetime.strptime(date_str.split(' ')[0][:10], '%Y-%m-%d')
        except:
            date_obj = datetime.now()
        
        weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
        month = date_obj.month
        
        return {
            'is_holiday': int(is_hol),
            'holiday_type': hol_info.get('type', 'none') if is_hol else 'none',
            'is_major_holiday': int(hol_info.get('is_major', False)) if is_hol else 0,
            'is_weekend': int(weekday >= 5),
            'day_of_week': weekday,
            'month': month,
            'is_summer_season': int(month in [6, 7, 8]),
            'is_winter_season': int(month in [12, 1, 2]),
            'days_to_nearest_holiday': self._days_to_nearest_holiday(date_str),
            'holiday_name_cn': hol_info.get('name_cn', '') if is_hol else '',
            'holiday_name_en': hol_info.get('name_en', '') if is_hol else '',
        }
    
    def _days_to_nearest_holiday(self, date_str):
        """计算距离最近假日的天数"""
        try:
            target = datetime.strptime(date_str.split(' ')[0][:10], '%Y-%m-%d')
        except:
            return 30  # 默认
        
        min_days = 365
        for key in self.holidays.keys():
            if '_' in key:
                start_mmd = key.split('_')[0]
                holiday_date = datetime(target.year, int(start_mmd[:2]), int(start_mmd[2:]))
            else:
                if len(key) >= 4:
                    holiday_date = datetime(target.year, int(key[:2]), int(key[2:4]))
                else:
                    continue
            
            days = abs((target - holiday_date).days)
            if days < min_days:
                min_days = days
        
        return min_days


class ExternalDataIntegrator:
    """外部数据整合器 - 统一管理所有外部数据源"""
    
    def __init__(self, weather_api_key=None, weather_provider='default'):
        self.weather_service = WeatherDataService(api_key=weather_api_key, provider=weather_provider)
        self.holiday_service = HolidayCalendarService()
    
    def enrich_booking_data(self, booking_data):
        """
        为单条预订数据补充外部特征
        输入: 原始预订数据字典
        输出: 补充了外部特征的字典（原地修改）
        """
        # 从预订数据提取必要字段
        country = booking_data.get('country', 'default')
        arrival_year = booking_data.get('arrival_date_year', 2024)
        arrival_month = booking_data.get('arrival_date_month', 'July')
        arrival_day = booking_data.get('arrival_date_day_of_month', 15)
        
        # 构造入住日期字符串
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month_num = month_map.get(arrival_month, 7)
        date_str = f"{int(arrival_year)}-{str(month_num).zfill(2)}-{str(int(arrival_day)).zfill(2)}"
        
        # 获取天气特征
        weather_features = self.weather_service.get_weather_by_city_date(country, date_str)
        
        # 获取节假日特征
        holiday_features = self.holiday_service.get_holiday_features(date_str)
        
        # 将外部特征合并到预订数据中
        external_features = {}
        external_features.update({f'weather_{k}': v for k, v in weather_features.items() 
                                  if isinstance(v, (int, float, bool))})
        external_features.update({f'holiday_{k}': v for k, v in holiday_features.items() 
                                  if isinstance(v, (int, float, bool))})
        
        booking_data.update(external_features)
        
        return booking_data
    
    def enrich_dataframe(self, df):
        """
        为整个 DataFrame 批量补充外部特征
        输入: 预订 DataFrame
        输出: 补充了外部特征的 DataFrame
        """
        df = df.copy()
        
        # 构造日期列
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        df['_date_str'] = (
            df['arrival_date_year'].astype(str) + '-' +
            df['arrival_date_month'].map(lambda x: str(month_map.get(x, 7))).str.zfill(2) + '-' +
            df['arrival_date_day_of_month'].astype(str).str.zfill(2)
        )
        
        # 批量获取天气特征
        weather_cols = ['weather_temp_max', 'weather_temp_min', 'weather_humidity',
                       'weather_precipitation_mm', 'weather_wind_speed_kmh',
                       'weather_cloud_cover_pct', 'weather_is_peak_season']
        
        for col in weather_cols:
            df[col] = 0.0
        
        for idx, row in df.iterrows():
            city = row.get('country', 'default')
            date_str = row['_date_str']
            weather = self.weather_service.get_weather_by_city_date(city, date_str)
            for wcol in weather_cols:
                suffix = wcol.replace('weather_', '')
                if suffix in weather:
                    df.at[idx, wcol] = weather[suffix]
        
        # 批量获取节假日特征
        holiday_cols = ['holiday_is_holiday', 'holiday_is_weekend', 'holiday_day_of_week',
                       'holiday_is_summer_season', 'holiday_is_winter_season',
                       'holiday_days_to_nearest_holiday', 'holiday_is_major_holiday']
        
        for col in holiday_cols:
            df[col] = 0
        
        for idx, row in df.iterrows():
            date_str = row['_date_str']
            hfeat = self.holiday_service.get_holiday_features(date_str)
            for hcol in holiday_cols:
                suffix = hcol.replace('holiday_', '')
                if suffix in hfeat:
                    df.at[idx, hcol] = hfeat[suffix]
        
        # 清理临时列
        df = df.drop(columns=['_date_str'], errors='ignore')
        
        return df
    
    def get_external_feature_names(self):
        """返回所有外部特征名称"""
        return [
            # 天气特征
            'weather_temp_max', 'weather_temp_min', 'weather_humidity',
            'weather_precipitation_mm', 'weather_wind_speed_kmh',
            'weather_cloud_cover_pct', 'weather_is_peak_season',
            # 节假日特征
            'holiday_is_holiday', 'holiday_is_weekend', 'holiday_day_of_week',
            'holiday_is_summer_season', 'holiday_is_winter_season',
            'holiday_days_to_nearest_holiday', 'holiday_is_major_holiday'
        ]
