"""
数据预处理模块
实现3-Sigma异常值检测与插值修复 + Savitzky-Golay平滑滤波
"""
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple
import warnings


class DataPreprocessor:
    """变压器数据预处理类"""
    
    def __init__(self, window_size: int = 5, poly_order: int = 2):
        """
        初始化预处理器
        
        Args:
            window_size: S-G滤波窗口大小，必须为奇数
            poly_order: 多项式阶数，必须小于window_size
        """
        self.window_size = window_size
        self.poly_order = poly_order
        
    def detect_outliers_3sigma(self, data: List[float], threshold: float = 3.0) -> Tuple[List[bool], List[float]]:
        """
        使用3-Sigma准则检测异常值
        
        Args:
            data: 输入数据序列
            threshold: Sigma阈值，通常为3.0
            
        Returns:
            is_outlier: 布尔列表，标记异常值位置
            fixed_data: 修复后的数据序列（异常值用线性插值替换）
        """
        if len(data) < 3:
            return [False] * len(data), data.copy()
        
        data_array = np.array(data, dtype=float)
        is_outlier = [False] * len(data)
        
        # 计算均值和标准差
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std < 1e-10:  # 标准差接近0，没有异常值
            return is_outlier, data.copy()
        
        # 标记异常值
        for i in range(len(data_array)):
            z_score = abs(data_array[i] - mean) / std
            if z_score > threshold:
                is_outlier[i] = True
        
        # 使用线性插值修复异常值
        fixed_data = data.copy()
        outlier_indices = [i for i, is_out in enumerate(is_outlier) if is_out]
        
        if outlier_indices:
            valid_indices = [i for i, is_out in enumerate(is_outlier) if not is_out]
            valid_values = [data[i] for i in valid_indices]
            
            # 对每个异常值进行线性插值
            for idx in outlier_indices:
                # 找到前后最近的非异常值
                left_idx = right_idx = None
                
                for i in range(idx - 1, -1, -1):
                    if not is_outlier[i]:
                        left_idx = i
                        break
                
                for i in range(idx + 1, len(data)):
                    if not is_outlier[i]:
                        right_idx = i
                        break
                
                # 线性插值
                if left_idx is not None and right_idx is not None:
                    # 两侧都有有效值
                    weight = (idx - left_idx) / (right_idx - left_idx)
                    fixed_data[idx] = data[left_idx] + weight * (data[right_idx] - data[left_idx])
                elif left_idx is not None:
                    # 只有左侧有效值，使用最近的值
                    fixed_data[idx] = data[left_idx]
                elif right_idx is not None:
                    # 只有右侧有效值，使用最近的值
                    fixed_data[idx] = data[right_idx]
        
        return is_outlier, fixed_data
    
    def apply_sg_filter(self, data: List[float]) -> List[float]:
        """
        应用Savitzky-Golay平滑滤波
        
        Args:
            data: 输入数据序列
            
        Returns:
            filtered_data: 滤波后的数据序列
        """
        if len(data) < self.window_size:
            return data.copy()
        
        data_array = np.array(data, dtype=float)
        
        # 确保window_size是奇数
        window = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
        
        # 应用S-G滤波
        filtered = savgol_filter(data_array, window, self.poly_order)
        
        return filtered.tolist()
    
    def preprocess_pipeline(self, data: List[float], apply_filter: bool = True) -> Dict:
        """
        完整的预处理管道
        
        Args:
            data: 原始数据序列
            apply_filter: 是否应用S-G滤波
            
        Returns:
            包含预处理结果的字典
        """
        if not data:
            return {
                'original': [],
                'outlier_mask': [],
                'interpolated': [],
                'filtered': [],
                'stats': {}
            }
        
        # 步骤1：3-Sigma异常值检测与插值修复
        outlier_mask, interpolated = self.detect_outliers_3sigma(data)
        
        # 步骤2：S-G平滑滤波
        filtered = self.apply_sg_filter(interpolated) if apply_filter else interpolated
        
        # 计算统计信息
        original_array = np.array(data, dtype=float)
        filtered_array = np.array(filtered, dtype=float)
        
        stats_info = {
            'original_mean': float(np.mean(original_array)),
            'original_std': float(np.std(original_array)),
            'filtered_mean': float(np.mean(filtered_array)),
            'filtered_std': float(np.std(filtered_array)),
            'outlier_count': sum(outlier_mask),
            'outlier_ratio': sum(outlier_mask) / len(data) if data else 0
        }
        
        return {
            'original': data,
            'outlier_mask': outlier_mask,
            'interpolated': interpolated,
            'filtered': filtered,
            'stats': stats_info
        }
    
    def preprocess_batch(self, data_dict: Dict[str, List[float]], apply_filter: bool = True) -> Dict[str, Dict]:
        """
        批量预处理多个数据通道
        
        Args:
            data_dict: {通道名: 数据序列} 的字典
            apply_filter: 是否应用S-G滤波
            
        Returns:
            {通道名: 预处理结果} 的字典
        """
        results = {}
        for channel_name, data in data_dict.items():
            results[channel_name] = self.preprocess_pipeline(data, apply_filter)
        return results


def test_preprocessor():
    """测试预处理器功能"""
    import matplotlib.pyplot as plt
    
    # 生成模拟数据
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_signal = 5 * np.sin(x) + 2 * x
    
    # 添加噪声和异常值
    noise = np.random.normal(0, 0.5, 100)
    data = true_signal + noise
    
    # 添加一些异常值
    data[10] = 20
    data[30] = -15
    data[70] = 25
    
    # 预处理
    preprocessor = DataPreprocessor(window_size=11, poly_order=3)
    result = preprocessor.preprocess_pipeline(data.tolist(), apply_filter=True)
    
    print(f"原始数据点数: {len(data)}")
    print(f"检测到异常值: {result['stats']['outlier_count']}个 ({result['stats']['outlier_ratio']*100:.1f}%)")
    print(f"原始均值: {result['stats']['original_mean']:.3f}, 标准差: {result['stats']['original_std']:.3f}")
    print(f"滤波后均值: {result['stats']['filtered_mean']:.3f}, 标准差: {result['stats']['filtered_std']:.3f}")
    
    return result


if __name__ == "__main__":
    test_preprocessor()