"""
决策融合模块
将 Duval 三角形法和随机森林模型的诊断结果进行融合
"""
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

# IEEE C57.104-2019 Level 1 正常浓度上限（ppm）
# 所有气体均低于此值时视为变压器处于正常状态，跳过 Duval，直接使用 RF 置信度
_NORMAL_LIMITS: Dict[str, float] = {
    'H2':   100.0,
    'CH4':  120.0,
    'C2H2':   3.0,
    'C2H4':  50.0,
    'C2H6':  65.0,
    'CO':   350.0,
    'CO2': 2500.0,
}


def _to_cartesian(ch4_pct: float, c2h4_pct: float):
    """三元坐标转直角坐标（与 Grafana 面板 toCartesian 函数完全一致）"""
    return (c2h4_pct + ch4_pct * 0.5, ch4_pct * 0.866025)


def _point_in_polygon(px: float, py: float, polygon) -> bool:
    """射线法判断点是否在多边形内"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# Duval 三角形 I 区域顶点定义（来自 Grafana 面板，格式 [CH4%, C2H4%]，C2H2% = 100 - CH4 - C2H4）
# 与面板完全保持一致，确保 Python 判定结果与可视化点位一致
_DUVAL_ZONE_VERTICES = [
    ('PD', [(98, 0),  (100, 0), (98, 2)]),
    ('T1', [(98, 2),  (80, 20), (76, 20), (96, 0),  (98, 0)]),
    ('T2', [(80, 20), (50, 50), (46, 50), (76, 20)]),
    ('T3', [(50, 50), (0, 100), (0, 85),  (35, 50), (46, 50)]),
    ('D1', [(87, 0),  (64, 23), (0, 23),  (0, 0)]),
    ('D2', [(64, 23), (74, 13), (47, 40), (31, 40), (0, 71), (0, 23)]),
    ('DT', [(96, 0),  (76, 20), (46, 50), (35, 50), (0, 85), (0, 71),
            (31, 40), (47, 40), (64, 23), (87, 0)]),
]

# 预计算直角坐标，避免运行时重复转换
_DUVAL_CARTESIAN = [
    (name, [_to_cartesian(ch4, c2h4) for ch4, c2h4 in verts])
    for name, verts in _DUVAL_ZONE_VERTICES
]


class FusionDiagnosis:
    """变压器故障融合诊断类"""

    DUVAL_FAULT_MAP = {
        'PD': 1,
        'D1': 2,
        'D2': 3,
        'DT': 7,
        'T1': 4,
        'T2': 5,
        'T3': 6,
        'N':  0,
    }

    FAULT_TYPES = {
        0: '正常老化',
        1: '局部放电',
        2: '低能量放电',
        3: '高能量放电',
        4: '热故障 <150°C',
        5: '热故障 150~300°C',
        6: '热故障 300~700°C',
        7: '放电 + 热故障',
    }

    def __init__(self, duval_weight: float = 0.4, rf_weight: float = 0.6):
        self.duval_weight = duval_weight
        self.rf_weight = rf_weight

    def _all_normal(self, gas_values: Dict[str, float]) -> bool:
        """判断所有气体是否均在 IEEE C57.104-2019 Level 1 正常范围内"""
        return all(
            gas_values.get(gas, 0.0) < limit
            for gas, limit in _NORMAL_LIMITS.items()
        )

    # ------------------------------------------------------------------
    # Duval 三角形
    # ------------------------------------------------------------------

    def duval_triangle_diagnosis(self, ch4_pct: float, c2h4_pct: float, c2h2_pct: float) -> Dict:
        """
        基于 IEC Duval 三角形 I 进行故障诊断。
        使用与 Grafana 面板相同的多边形顶点，通过射线法判断点所在区域，
        保证前端可视化与后端判定结果一致。
        """
        total = ch4_pct + c2h4_pct + c2h2_pct
        if total <= 0:
            region = 'N'
        else:
            ch4  = ch4_pct  / total * 100
            c2h4 = c2h4_pct / total * 100
            px, py = _to_cartesian(ch4, c2h4)

            region = 'N'
            for name, polygon in _DUVAL_CARTESIAN:
                if _point_in_polygon(px, py, polygon):
                    region = name
                    break

        fault_code = self.DUVAL_FAULT_MAP.get(region, 0)
        return {
            'method': 'Duval Triangle I',
            'fault_type': self.FAULT_TYPES.get(fault_code, '未知'),
            'fault_code': fault_code,
            'confidence': 0.7,
            'details': {
                'ch4_pct':  round(ch4_pct,  2),
                'c2h4_pct': round(c2h4_pct, 2),
                'c2h2_pct': round(c2h2_pct, 2),
                'region':   region,
            },
        }

    # ------------------------------------------------------------------
    # 随机森林
    # ------------------------------------------------------------------

    def random_forest_diagnosis(self, rf_model, features: Dict) -> Dict:
        if rf_model is None or not hasattr(rf_model, 'predict'):
            return {
                'method': 'Random Forest',
                'fault_type': '未知',
                'fault_code': -1,
                'confidence': 0.0,
                'error': '模型不可用',
            }
        result = rf_model.predict(features)
        return {
            'method': 'Random Forest',
            'fault_type':    result.get('fault_type', '未知'),
            'fault_code':    result.get('fault_code', -1),
            'confidence':    result.get('confidence', 0.0),
            'probabilities': result.get('probabilities', {}),
            'error':         result.get('error'),
        }

    # ------------------------------------------------------------------
    # 融合
    # ------------------------------------------------------------------

    def fuse_diagnoses(self, duval_result: Dict, rf_result: Dict,
                       gas_values: Optional[Dict[str, float]] = None) -> Dict:
        """
        融合诊断：
        - 所有气体均在正常范围内（IEEE C57.104-2019 Level 1）→ 直接采用 RF 结果
        - 任意气体超标 → Duval + RF 加权投票
        """
        duval_code = duval_result.get('fault_code', 0)
        rf_code    = rf_result.get('fault_code', 0)

        # 任一方法失败则直接返回另一方
        if duval_code == -1:
            return rf_result
        if rf_code == -1:
            return duval_result

        # 正常状态：跳过 Duval，直接使用 RF 置信度
        if gas_values and self._all_normal(gas_values):
            rf_conf = rf_result.get('confidence', 0.0)
            return {
                'method':     'RF Only (Normal State)',
                'fault_type': rf_result.get('fault_type', '未知'),
                'fault_code': rf_code,
                'confidence': round(rf_conf, 4),
                'weights':    {'duval': 0.0, 'random_forest': 1.0},
                'sub_results': {
                    'duval':         duval_result,
                    'random_forest': rf_result,
                },
                'timestamp': datetime.now().isoformat(),
            }

        # 异常状态：Duval + RF 加权投票
        duval_conf = duval_result.get('confidence', 0.0)
        rf_conf    = rf_result.get('confidence', 0.0)

        votes = np.zeros(len(self.FAULT_TYPES))
        votes[duval_code] += self.duval_weight * duval_conf
        votes[rf_code]    += self.rf_weight    * rf_conf

        final_code = int(np.argmax(votes))
        final_conf = float(votes[final_code])

        return {
            'method':     'Fusion (Duval + RF)',
            'fault_type': self.FAULT_TYPES.get(final_code, '未知') if final_conf >= 0.3 else '不确定',
            'fault_code': final_code,
            'confidence': round(final_conf, 4),
            'weights':    {'duval': self.duval_weight, 'random_forest': self.rf_weight},
            'sub_results': {
                'duval':         duval_result,
                'random_forest': rf_result,
            },
            'timestamp': datetime.now().isoformat(),
        }
