"""
随机森林诊断模块
基于IEC 60599标准和多模态特征进行变压器故障诊断
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List, Tuple, Optional
import warnings


class RandomForestDiagnosis:
    """基于随机森林的变压器故障诊断类"""
    
    # 故障类型映射
    FAULT_TYPES = {
        0: '正常老化',
        1: '局部放电',
        2: '低能量放电',
        3: '高能量放电',
        4: '热故障 <150°C',
        5: '热故障 150~300°C',
        6: '热故障 300~700°C',
        7: '热故障 >700°C'
    }
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, model_path: str = None):
        """
        初始化随机森林诊断器
        
        Args:
            n_estimators: 森林中树的数量
            random_state: 随机种子，确保结果可复现
            model_path: 预训练模型路径（可选）
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def generate_iec60599_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据IEC 60599标准生成模拟训练数据
        
        Args:
            n_samples: 每个故障类型的样本数量
            
        Returns:
            特征矩阵X和标签y
        """
        X = []
        y = []
        
        # 定义每种故障类型的特征分布
        fault_distributions = {
            0: {  # 正常老化
                'IEC_CH4_H2': (0.05, 0.02),  # <0.1 (IEC 60599 Code 0)
                'IEC_C2H2_C2H4': (0.05, 0.02),  # <0.1
                'IEC_C2H4_C2H6': (0.5, 0.1),  # <1
                'CO2_CO_Ratio': (13, 2),  # >11 (IEC 60599 正常老化典型值)
                'Moisture_Relative_Humidity': (25, 5),  # <30%
                'Delta_T_Winding_Oil': (15, 3)  # 正常温升
            },
            1: {  # 局部放电
                'IEC_CH4_H2': (0.5, 0.15),  # 0.1~1
                'IEC_C2H2_C2H4': (0.03, 0.01),  # <0.1
                'IEC_C2H4_C2H6': (0.8, 0.2),  # <1
                'CO2_CO_Ratio': (8, 1.5),  # 3~10
                'Moisture_Relative_Humidity': (35, 5),  # 30~50%
                'Delta_T_Winding_Oil': (18, 4)
            },
            2: {  # 低能量放电
                'IEC_CH4_H2': (0.8, 0.2),  # 0.1~1
                'IEC_C2H2_C2H4': (1.5, 0.5),  # 0.1~3
                'IEC_C2H4_C2H6': (2.0, 0.5),  # 1~3
                'CO2_CO_Ratio': (6, 1.5),  # 3~10
                'Moisture_Relative_Humidity': (40, 8),  # 30~50%
                'Delta_T_Winding_Oil': (22, 5)
            },
            3: {  # 高能量放电
                'IEC_CH4_H2': (1.5, 0.3),  # >1
                'IEC_C2H2_C2H4': (5.0, 1.0),  # >3
                'IEC_C2H4_C2H6': (4.0, 1.0),  # >3
                'CO2_CO_Ratio': (5, 1.2),  # 3~10
                'Moisture_Relative_Humidity': (45, 10),  # 30~50%
                'Delta_T_Winding_Oil': (28, 6)
            },
            4: {  # 热故障 <150°C
                'IEC_CH4_H2': (0.5, 0.15),  # 0.1~1
                'IEC_C2H2_C2H4': (0.05, 0.02),  # <0.1
                'IEC_C2H4_C2H6': (1.5, 0.3),  # 1~3
                'CO2_CO_Ratio': (4, 1),  # 3~10
                'Moisture_Relative_Humidity': (50, 10),  # >50%
                'Delta_T_Winding_Oil': (20, 4)
            },
            5: {  # 热故障 150~300°C
                'IEC_CH4_H2': (0.7, 0.2),  # 0.1~1
                'IEC_C2H2_C2H4': (0.15, 0.05),  # 0.1~3
                'IEC_C2H4_C2H6': (2.5, 0.5),  # 1~3
                'CO2_CO_Ratio': (3.5, 0.8),  # 3~10
                'Moisture_Relative_Humidity': (55, 12),  # >50%
                'Delta_T_Winding_Oil': (25, 5)
            },
            6: {  # 热故障 300~700°C
                'IEC_CH4_H2': (1.2, 0.3),  # >1
                'IEC_C2H2_C2H4': (0.2, 0.08),  # 0.1~3
                'IEC_C2H4_C2H6': (3.5, 0.8),  # >3
                'CO2_CO_Ratio': (2.5, 0.6),  # <3
                'Moisture_Relative_Humidity': (60, 15),  # >50%
                'Delta_T_Winding_Oil': (35, 8)
            },
            7: {  # 热故障 >700°C
                'IEC_CH4_H2': (1.5, 0.4),  # >1
                'IEC_C2H2_C2H4': (0.3, 0.1),  # >3
                'IEC_C2H4_C2H6': (4.0, 1.0),  # >3
                'CO2_CO_Ratio': (2.0, 0.5),  # <3
                'Moisture_Relative_Humidity': (65, 18),  # >50%
                'Delta_T_Winding_Oil': (40, 10)
            }
        }
        
        # 生成特征名称（与特征提取器输出对应）
        self.feature_names = [
            'IEC_CH4_H2', 'IEC_C2H2_C2H4', 'IEC_C2H4_C2H6',
            'CO2_CO_Ratio', 'Moisture_Relative_Humidity', 'Delta_T_Winding_Oil'
        ]
        
        # 为每种故障类型生成样本
        for fault_type, distributions in fault_distributions.items():
            for _ in range(n_samples):
                features = []
                for feature_name in self.feature_names:
                    mean, std = distributions[feature_name]
                    value = np.random.normal(mean, std)
                    features.append(max(0, value))  # 确保非负
                
                X.append(features)
                y.append(fault_type)
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None, test_size: float = 0.2):
        """
        训练随机森林模型
        
        Args:
            X: 特征矩阵，如果为None则使用IEC 60599标准生成的数据
            y: 标签数组，如果为None则使用IEC 60599标准生成的标签
            test_size: 测试集比例
        """
        # 如果没有提供数据，则生成IEC 60599标准数据
        if X is None or y is None:
            print("使用IEC 60599标准生成模拟训练数据...")
            X, y = self.generate_iec60599_training_data(n_samples=500)
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )
        
        # 创建随机森林分类器
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 训练模型
        print("开始训练随机森林模型...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"模型训练完成！")
        print(f"测试集准确率: {accuracy:.4f}")
        print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=list(self.FAULT_TYPES.values()))}")
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            print("\n特征重要性:")
            for i, (name, importance) in enumerate(zip(self.feature_names, importances)):
                print(f"  {name}: {importance:.4f}")
        
        self.is_trained = True
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        使用训练好的模型进行预测
        
        Args:
            features: 特征字典（与特征提取器输出格式相同）
            
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            return {
                'fault_type': '未知',
                'fault_code': -1,
                'confidence': 0.0,
                'probabilities': {},
                'error': '模型未训练'
            }
        
        # 按照训练时的特征顺序提取特征值
        feature_values = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_values.append(features[feature_name])
            else:
                # 如果缺少特征，使用默认值0
                feature_values.append(0.0)
        
        # 转换为numpy数组并标准化
        X = np.array([feature_values])
        X_scaled = self.scaler.transform(X)
        
        # 预测
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # 获取置信度
        confidence = np.max(probabilities)
        
        # 构建概率字典
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            if prob > 0.01:  # 只显示概率大于1%的故障类型
                prob_dict[self.FAULT_TYPES.get(i, f'未知{i}')] = round(prob, 4)
        
        return {
            'fault_type': self.FAULT_TYPES.get(prediction, f'未知{prediction}'),
            'fault_code': int(prediction),
            'confidence': round(float(confidence), 4),
            'probabilities': prob_dict,
            'input_features': features,
            'error': None
        }
    
    def save_model(self, model_path: str):
        """保存模型到文件"""
        if not self.is_trained:
            print("模型未训练，无法保存")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'fault_types': self.FAULT_TYPES
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str):
        """从文件加载模型"""
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"模型已从 {model_path} 加载")
        return True


def test_random_forest():
    """测试随机森林诊断器"""
    # 创建诊断器实例
    diagnosis = RandomForestDiagnosis(n_estimators=100)
    
    # 训练模型
    diagnosis.train()
    
    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
    diagnosis.save_model(model_path)
    
    # 测试预测
    test_features = {
        'IEC_CH4_H2': 0.5,
        'IEC_C2H2_C2H4': 0.05,
        'IEC_C2H4_C2H6': 0.8,
        'CO2_CO_Ratio': 8.0,
        'Moisture_Relative_Humidity': 25.0,
        'Delta_T_Winding_Oil': 15.0
    }
    
    print("\n测试预测:")
    print(f"输入特征: {test_features}")
    result = diagnosis.predict(test_features)
    print(f"预测结果: {result}")
    
    return diagnosis


if __name__ == "__main__":
    test_random_forest()