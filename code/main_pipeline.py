"""
主数据处理管道
独立运行（与 simulate.py 并行），从 InfluxDB 读取原始数据，
经过预处理 → 特征提取 → 随机森林 + Duval 融合诊断后，
将诊断结果写回 InfluxDB 供 Grafana 展示。

停止方式：终端按 Enter
"""

import os
import sys
import signal
import time
from collections import deque
from datetime import datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from preprocessor import DataPreprocessor
from feature_extractor import FeatureExtractor
from random_forest_diagnosis import RandomForestDiagnosis
from fusion_diagnosis import FusionDiagnosis

# ==================== 配置 ====================
URL    = os.getenv('INFLUXDB_URL',   'http://localhost:8086')
TOKEN  = os.getenv('INFLUXDB_TOKEN', 'my-influxdb-super-secret-token')
ORG    = os.getenv('INFLUXDB_ORG',   'lab')
BUCKET = os.getenv('INFLUXDB_BUCKET','transformer')

# simulate.py 每 3 秒写一条，S-G 滤波窗口需要 11 个点（约 33 秒数据）
SG_WINDOW   = 11
# pipeline 每隔多少秒处理一次（建议 ≥ 30，给 simulate.py 足够时间积累窗口数据）
POLL_INTERVAL = 30

MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')

# 需要采集并预处理的字段
GAS_FIELDS  = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2']
TEMP_FIELDS = ['顶层油温', '绕组温度', '微水含量']
ALL_FIELDS  = GAS_FIELDS + TEMP_FIELDS


class DataPipeline:

    def __init__(self):
        # InfluxDB
        try:
            self.client    = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            print('✅ InfluxDB 连接成功')
        except Exception as e:
            print(f'❌ InfluxDB 连接失败: {e}')
            sys.exit(1)

        # 预处理（S-G 窗口 11，多项式阶 3）
        self.preprocessor = DataPreprocessor(window_size=SG_WINDOW, poly_order=3)

        # 特征提取
        self.extractor = FeatureExtractor()

        # 随机森林
        self.rf = RandomForestDiagnosis()
        os.makedirs(MODEL_DIR, exist_ok=True)
        if os.path.exists(MODEL_PATH):
            self.rf.load_model(MODEL_PATH)
            print('✅ 随机森林模型加载成功')
        else:
            print('⚠️  未找到预训练模型，开始训练…')
            self.rf.train()
            self.rf.save_model(MODEL_PATH)
            print(f'✅ 模型已保存到 {MODEL_PATH}')

        # 融合诊断
        self.fusion = FusionDiagnosis(duval_weight=0.4, rf_weight=0.6)

        # 每个字段的历史原始值缓冲（滑动窗口，用于预处理）
        # maxlen = SG_WINDOW 即可；simulate.py 写 3s/条，30s 内约 10 条，足够窗口
        self.buffers: dict[str, deque] = {
            field: deque(maxlen=SG_WINDOW) for field in ALL_FIELDS
        }

        self.is_running = False

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------

    def _read_latest(self) -> dict | None:
        """从 InfluxDB 读取最新一条 transformer_001 数据"""
        query = '''
        from(bucket: "transformer")
          |> range(start: -2m)
          |> filter(fn: (r) => r._measurement == "transformer_001")
          |> last()
        '''
        try:
            tables = self.query_api.query(query, org=ORG)
            data = {}
            for table in tables:
                for record in table.records:
                    data[record.get_field()] = record.get_value()
            return data if data else None
        except Exception as e:
            print(f'❌ 读取数据失败: {e}')
            return None

    # ------------------------------------------------------------------
    # 预处理（真正使用滑动窗口 + S-G 滤波）
    # ------------------------------------------------------------------

    def _preprocess(self, raw: dict) -> dict | None:
        """
        将最新原始值追加到各字段的滑动窗口缓冲，
        对窗口内数据执行 3-Sigma 异常检测 + S-G 平滑，
        返回经过滤波后的最新值。
        窗口数据不足时仍返回原始值（退化为直通模式）。
        """
        # 更新缓冲
        for field in ALL_FIELDS:
            if field in raw:
                self.buffers[field].append(float(raw[field]))

        processed = {}
        for field in ALL_FIELDS:
            buf = list(self.buffers[field])
            if not buf:
                continue
            if len(buf) >= SG_WINDOW:
                result = self.preprocessor.preprocess_pipeline(buf, apply_filter=True)
                processed[field] = result['filtered'][-1]
            else:
                # 窗口数据不足，直接用原始值
                processed[field] = buf[-1]

        return processed if processed else None

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------

    def _extract_features(self, data: dict) -> dict | None:
        gas_values = {g: data.get(g, 0.0) for g in GAS_FIELDS}
        moisture   = data.get('微水含量', 0.0)
        oil_temp   = data.get('顶层油温', 0.0)
        winding_temp = data.get('绕组温度', 0.0)
        historical_oil_temps = list(self.buffers['顶层油温'])

        try:
            return self.extractor.extract_all_features(
                gas_values, moisture, oil_temp, winding_temp, historical_oil_temps
            )
        except Exception as e:
            print(f'❌ 特征提取失败: {e}')
            return None

    # ------------------------------------------------------------------
    # 融合诊断
    # ------------------------------------------------------------------

    def _diagnose(self, features: dict, preprocessed: dict) -> dict | None:
        ch4  = preprocessed.get('CH4',  0.0)
        c2h4 = preprocessed.get('C2H4', 0.0)
        c2h2 = preprocessed.get('C2H2', 0.0)
        total = ch4 + c2h4 + c2h2

        if total > 0:
            ch4_pct  = ch4  / total * 100
            c2h4_pct = c2h4 / total * 100
            c2h2_pct = c2h2 / total * 100
        else:
            ch4_pct = c2h4_pct = c2h2_pct = 0.0

        duval_result = self.fusion.duval_triangle_diagnosis(ch4_pct, c2h4_pct, c2h2_pct)
        rf_result    = self.fusion.random_forest_diagnosis(self.rf, features)
        gas_values   = {g: preprocessed.get(g, 0.0) for g in GAS_FIELDS}
        return self.fusion.fuse_diagnoses(duval_result, rf_result, gas_values=gas_values)

    # ------------------------------------------------------------------
    # 写回 InfluxDB
    # ------------------------------------------------------------------

    def _write(self, diagnosis: dict):
        sub = diagnosis.get('sub_results', {})
        point = (
            Point('transformer_001_diagnosis')
            .tag('method', 'fusion')
            .field('fault_code',      int(diagnosis.get('fault_code', -1)))
            .field('fault_type_name', diagnosis.get('fault_type', '未知'))
            .field('confidence',      float(diagnosis.get('confidence', 0.0)))
            .field('duval_fault_code',int(sub.get('duval', {}).get('fault_code', -1)))
            .field('duval_region',    sub.get('duval', {}).get('details', {}).get('region', 'N'))
            .field('rf_fault_code',   int(sub.get('random_forest', {}).get('fault_code', -1)))
            .field('rf_confidence',   float(sub.get('random_forest', {}).get('confidence', 0.0)))
        )
        self.write_api.write(bucket=BUCKET, record=point)

    # ------------------------------------------------------------------
    # 单次处理循环
    # ------------------------------------------------------------------

    def process_once(self):
        print(f"\n{'='*52}")
        print(f"⏰  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        raw = self._read_latest()
        if not raw:
            print('⚠️  无数据，跳过本次处理')
            return

        preprocessed = self._preprocess(raw)
        if not preprocessed:
            print('⚠️  预处理失败，跳过')
            return

        buf_len = len(self.buffers['H2'])
        mode = 'S-G 滤波' if buf_len >= SG_WINDOW else f'直通（缓冲 {buf_len}/{SG_WINDOW}）'
        print(f'🔧  预处理模式: {mode}')

        features = self._extract_features(preprocessed)
        if not features:
            return

        diagnosis = self._diagnose(features, preprocessed)
        if not diagnosis:
            return

        sub   = diagnosis.get('sub_results', {})
        duval = sub.get('duval', {})
        rf    = sub.get('random_forest', {})

        # 传感器数据
        H2   = preprocessed.get('H2',   0.0)
        CH4  = preprocessed.get('CH4',  0.0)
        C2H2 = preprocessed.get('C2H2', 0.0)
        C2H4 = preprocessed.get('C2H4', 0.0)
        C2H6 = preprocessed.get('C2H6', 0.0)
        CO   = preprocessed.get('CO',   0.0)
        CO2  = preprocessed.get('CO2',  0.0)
        oil  = preprocessed.get('顶层油温', 0.0)
        wind = preprocessed.get('绕组温度', 0.0)
        mois = preprocessed.get('微水含量', 0.0)

        # IEC 三比值
        r1 = CH4  / H2   if H2   > 0 else 0
        r2 = C2H2 / C2H4 if C2H4 > 0 else 0
        r3 = C2H4 / C2H6 if C2H6 > 0 else 0

        print(
            f"📊  气体(ppm)  H2:{H2:.1f}  CH4:{CH4:.1f}  C2H2:{C2H2:.3f}  "
            f"C2H4:{C2H4:.2f}  C2H6:{C2H6:.1f}  CO:{CO:.1f}  CO2:{CO2:.1f}"
        )
        print(
            f"🌡️   温度/湿度  油温:{oil:.1f}°C  绕组:{wind:.1f}°C  微水:{mois:.2f}ppm"
        )
        print(
            f"📐  IEC三比值  CH4/H2:{r1:.4f}  C2H2/C2H4:{r2:.4f}  C2H4/C2H6:{r3:.4f}"
        )
        method = diagnosis.get('method', '')
        mode   = '仅RF·气体正常' if 'Normal State' in method else 'Duval+RF'
        print(
            f"🏥  融合诊断:  {diagnosis['fault_type']} "
            f"(置信度 {diagnosis['confidence']:.0%})  [{mode}]  |  "
            f"Duval:{duval.get('details', {}).get('region', '?')}(conf {duval.get('confidence', 0):.0%})  |  "
            f"RF:{rf.get('fault_type', '?')}(conf {rf.get('confidence', 0):.0%})"
        )

        try:
            self._write(diagnosis)
            print('✅  结果已写入 InfluxDB')
        except Exception as e:
            print(f'❌  写入失败: {e}')

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'is_running', False))
        signal.signal(signal.SIGINT,  lambda *_: setattr(self, 'is_running', False))

        print('\n' + '='*52)
        print('🚀  变压器数据处理管道启动')
        print(f'📡  InfluxDB: {URL}')
        print(f'⏱️   轮询间隔: {POLL_INTERVAL}s')
        print('🛑  停止方式: SIGTERM / Ctrl-C')
        print('='*52 + '\n')

        self.is_running = True

        while self.is_running:
            try:
                self.process_once()
            except Exception as e:
                import traceback
                print(f'❌  处理异常: {e}')
                traceback.print_exc()

            for _ in range(POLL_INTERVAL):
                if not self.is_running:
                    break
                time.sleep(1)

        print('\n✅  正在关闭管道…')
        self.client.close()
        print('👋  管道已安全退出')

if __name__ == '__main__':
    DataPipeline().run()
