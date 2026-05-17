"""
特征提取模块
基于PDF文档要求，提取多模态特征用于随机森林诊断
"""
import numpy as np
from typing import Dict, List, Optional
import math
from datetime import datetime, timedelta


class FeatureExtractor:
    """变压器特征提取类"""
    
    # 气体常量
    GAS_NAMES = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2']
    
    def __init__(self):
        """初始化特征提取器"""
        # 用于存储历史数据计算增长率
        self.historical_data = []
        self.max_history_hours = 24  # 最多保存24小时的历史数据
        
    def calculate_iec_ratios(self, gas_values: Dict[str, float]) -> Dict[str, float]:
        """
        计算IEC三比值
        
        Args:
            gas_values: 气体浓度值字典 {'H2': 15.0, 'CH4': 10.0, ...}
            
        Returns:
            IEC三比值字典
        """
        h2 = gas_values.get('H2', 0)
        ch4 = gas_values.get('CH4', 0)
        c2h2 = gas_values.get('C2H2', 0)
        c2h4 = gas_values.get('C2H4', 0)
        c2h6 = gas_values.get('C2H6', 0)
        
        # 避免除零错误
        ratio_ch4_h2 = ch4 / h2 if h2 > 0 else 0
        ratio_c2h2_c2h4 = c2h2 / c2h4 if c2h4 > 0 else 0
        ratio_c2h4_c2h6 = c2h4 / c2h6 if c2h6 > 0 else 0
        
        return {
            'IEC_CH4_H2': ratio_ch4_h2,
            'IEC_C2H2_C2H4': ratio_c2h2_c2h4,
            'IEC_C2H4_C2H6': ratio_c2h4_c2h6
        }
    
    def calculate_co_co2_ratio(self, gas_values: Dict[str, float]) -> Dict[str, float]:
        """
        计算CO/CO2比值
        
        Args:
            gas_values: 气体浓度值字典
            
        Returns:
            CO/CO2相关特征
        """
        co = gas_values.get('CO', 0)
        co2 = gas_values.get('CO2', 0)
        
        # CO2/CO比值（用户特别要求）
        ratio_co2_co = co2 / co if co > 0 else 0
        
        # CO/(CO+CO2)比值 - 另一个有用的特征
        ratio_co_total = co / (co + co2) if (co + co2) > 0 else 0
        
        return {
            'CO2_CO_Ratio': ratio_co2_co,
            'CO_Total_Ratio': ratio_co_total,
            'CO': co,
            'CO2': co2
        }
    
    def calculate_total_hydrocarbon(self, gas_values: Dict[str, float]) -> Dict[str, float]:
        """
        计算总烃（Total Hydrocarbon, THC）
        
        Args:
            gas_values: 气体浓度值字典
            
        Returns:
            总烃相关特征
        """
        ch4 = gas_values.get('CH4', 0)
        c2h2 = gas_values.get('C2H2', 0)
        c2h4 = gas_values.get('C2H4', 0)
        c2h6 = gas_values.get('C2H6', 0)
        
        thc = ch4 + c2h2 + c2h4 + c2h6
        
        return {
            'THC': thc,
            'CH4': ch4,
            'C2H2': c2h2,
            'C2H4': c2h4,
            'C2H6': c2h6
        }
    
    def calculate_thc_growth_rate(self, current_thc: float) -> Dict[str, float]:
        """
        计算总烃增长率
        
        Args:
            current_thc: 当前总烃值
            
        Returns:
            总烃增长率特征
        """
        # 获取当前时间
        now = datetime.now()
        
        # 添加当前数据到历史记录
        self.historical_data.append({'time': now, 'thc': current_thc})
        
        # 清理超过24小时的历史数据
        cutoff_time = now - timedelta(hours=self.max_history_hours)
        self.historical_data = [d for d in self.historical_data if d['time'] >= cutoff_time]
        
        growth_rate_24h = 0.0
        growth_rate_1h = 0.0
        
        if len(self.historical_data) >= 2:
            # 计算24小时增长率
            earliest = self.historical_data[0]
            time_diff_hours = (now - earliest['time']).total_seconds() / 3600
            if time_diff_hours > 0 and earliest['thc'] > 0:
                growth_rate_24h = (current_thc - earliest['thc']) / earliest['thc'] * 100 / time_diff_hours
            
            # 计算1小时增长率
            one_hour_ago = now - timedelta(hours=1)
            closest_1h = min(self.historical_data, key=lambda d: abs((d['time'] - one_hour_ago).total_seconds()))
            time_diff_1h = (now - closest_1h['time']).total_seconds() / 3600
            if time_diff_1h > 0 and closest_1h['thc'] > 0:
                growth_rate_1h = (current_thc - closest_1h['thc']) / closest_1h['thc'] * 100 / time_diff_1h
        
        return {
            'THC_Growth_Rate_24h': growth_rate_24h,
            'THC_Growth_Rate_1h': growth_rate_1h
        }
    
    def calculate_moisture_relative_humidity(self, moisture_ppm: float, oil_temp: float) -> Dict[str, float]:
        """
        将微水绝对含量转换为相对饱和度
        
        Args:
            moisture_ppm: 微水含量（ppm）
            oil_temp: 油温（°C）
            
        Returns:
            微水相关特征
        """
        # 计算饱和蒸汽压（简化公式，单位：ppm）
        # 基于经验公式：饱和浓度(ppm) = 10^(2.5 + 0.025*T)  其中T为温度(°C)
        # 实际应使用更精确的公式，但这里简化处理
        
        # 更精确的公式（基于IEC 60814）
        # 油中水的饱和浓度(ppm) ≈ 10^(5.86 - 719/T_K) * 10^6
        # T_K为绝对温度(K)
        t_k = oil_temp + 273.15
        
        # 简化公式（适用于0-100°C范围）
        # 饱和浓度(ppm) ≈ 200 * exp(0.035*T - 1.5)
        saturation_ppm = 200 * math.exp(0.035 * oil_temp - 1.5)
        
        # 相对饱和度（%）
        relative_humidity = (moisture_ppm / saturation_ppm) * 100 if saturation_ppm > 0 else 0
        relative_humidity = min(relative_humidity, 100.0)  # 限制在100%以内
        
        # 计算危险度指标
        # 正常：<30%，注意：30-50%，危险：>50%
        danger_level = 0
        if relative_humidity > 50:
            danger_level = 2  # 危险
        elif relative_humidity > 30:
            danger_level = 1  # 注意
        
        return {
            'Moisture_ppm': moisture_ppm,
            'Moisture_Relative_Humidity': relative_humidity,
            'Moisture_Danger_Level': danger_level,
            'Oil_Temperature': oil_temp
        }
    
    def calculate_temperature_features(self, winding_temp: float, oil_temp: float, 
                                      historical_oil_temps: List[float] = None) -> Dict[str, float]:
        """
        计算温度相关特征
        
        Args:
            winding_temp: 绕组温度（°C）
            oil_temp: 当前油温（°C）
            historical_oil_temps: 过去1小时的油温历史数据
            
        Returns:
            温度相关特征
        """
        # 温升特征：ΔT = T绕组 - T油
        delta_t_winding_oil = winding_temp - oil_temp
        
        # 计算温度变化率（用过去1小时的平均温度变化率替代环境温度）
        temp_change_rate = 0.0
        if historical_oil_temps and len(historical_oil_temps) >= 2:
            # 计算过去1小时的平均变化率
            temp_diff = historical_oil_temps[-1] - historical_oil_temps[0]
            time_diff_hours = 1.0  # 假设是1小时的数据
            temp_change_rate = temp_diff / time_diff_hours
        
        # 计算温度绝对值特征
        temp_max = winding_temp
        temp_min = oil_temp
        temp_avg = (winding_temp + oil_temp) / 2
        
        return {
            'Delta_T_Winding_Oil': delta_t_winding_oil,
            'Temp_Change_Rate': temp_change_rate,
            'Winding_Temp': winding_temp,
            'Oil_Temp': oil_temp,
            'Temp_Max': temp_max,
            'Temp_Min': temp_min,
            'Temp_Avg': temp_avg
        }
    
    def extract_all_features(self, gas_values: Dict[str, float], moisture_ppm: float, 
                           oil_temp: float, winding_temp: float, 
                           historical_oil_temps: List[float] = None) -> Dict[str, float]:
        """
        提取所有特征
        
        Args:
            gas_values: 气体浓度值字典
            moisture_ppm: 微水含量（ppm）
            oil_temp: 油温（°C）
            winding_temp: 绕组温度（°C）
            historical_oil_temps: 过去1小时的油温历史数据
            
        Returns:
            所有特征的字典
        """
        features = {}
        
        # 1. IEC三比值
        features.update(self.calculate_iec_ratios(gas_values))
        
        # 2. CO/CO2比值（用户特别要求）
        features.update(self.calculate_co_co2_ratio(gas_values))
        
        # 3. 总烃特征
        thc_features = self.calculate_total_hydrocarbon(gas_values)
        features.update(thc_features)
        
        # 4. 总烃增长率
        growth_features = self.calculate_thc_growth_rate(thc_features['THC'])
        features.update(growth_features)
        
        # 5. 微水相对饱和度
        features.update(self.calculate_moisture_relative_humidity(moisture_ppm, oil_temp))
        
        # 6. 温度特征
        features.update(self.calculate_temperature_features(winding_temp, oil_temp, historical_oil_temps))
        
        # 7. 添加原始气体值（可选，用于调试）
        features.update({f'Raw_{gas}': gas_values.get(gas, 0) for gas in self.GAS_NAMES})
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表（与extract_all_features输出对应）
        
        Returns:
            特征名称列表
        """
        feature_names = [
            # IEC三比值
            'IEC_CH4_H2', 'IEC_C2H2_C2H4', 'IEC_C2H4_C2H6',
            # CO/CO2比值
            'CO2_CO_Ratio', 'CO_Total_Ratio', 'CO', 'CO2',
            # 总烃
            'THC', 'CH4', 'C2H2', 'C2H4', 'C2H6',
            # 增长率
            'THC_Growth_Rate_24h', 'THC_Growth_Rate_1h',
            # 微水
            'Moisture_ppm', 'Moisture_Relative_Humidity', 'Moisture_Danger_Level', 'Oil_Temperature',
            # 温度
            'Delta_T_Winding_Oil', 'Temp_Change_Rate', 'Winding_Temp', 'Oil_Temp',
            'Temp_Max', 'Temp_Min', 'Temp_Avg'
        ]
        
        # 添加原始气体值特征名
        feature_names.extend([f'Raw_{gas}' for gas in self.GAS_NAMES])
        
        return feature_names


def test_feature_extractor():
    """测试特征提取器"""
    extractor = FeatureExtractor()
    
    # 模拟数据
    gas_values = {
        'H2': 15.0,
        'CH4': 10.0,
        'C2H2': 0.5,
        'C2H4': 8.0,
        'C2H6': 20.0,
        'CO': 120.0,
        'CO2': 800.0
    }
    moisture_ppm = 12.0
    oil_temp = 55.0
    winding_temp = 70.0
    
    # 提取特征
    features = extractor.extract_all_features(gas_values, moisture_ppm, oil_temp, winding_temp)
    
    print("提取的特征:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n特征数量: {len(features)}")
    print(f"特征名称列表: {extractor.get_feature_names()}")
    
    return features


if __name__ == "__main__":
    test_feature_extractor()