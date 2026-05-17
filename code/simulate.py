import time
import random
import math
import threading
import sys
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ================= 配置区域 =================
URL         = "http://localhost:8086"
TOKEN       = "my-influxdb-super-secret-token"
ORG         = "lab"
BUCKET      = "transformer"
NOISE_SCALE = 0.3   # 传感器噪声缩放系数，1.0 为原始强度，调小使曲线更平滑

# ================= 故障场景定义 =================
# 每个场景的基准气体值（ppm）均来自 IEC 60599 典型值范围，
# 确保 IEC 三比值编码和随机森林特征均落入对应故障类的训练分布中心。
#
# 场景格式：(名称, {气体基准值}, 顶层油温基准, 绕组温度基准, CO基准, CO2基准, 微水基准)
SCENARIOS = {
    '0': {
        'name':    '正常老化',
        'desc':    'H2↑ CH4低, 三比值均 <阈值，CO2/CO≈10，绝缘正常老化',
        # IEC: CH4/H2=0.08(code0), C2H2/C2H4=0.05(0), C2H4/C2H6=0.5(0) → 000
        # H2=250 使比值=0.08，远离 0.1 的临界点，避免落入局部放电编码
        'H2':    50.0, 'CH4':   4.0, 'C2H2': 0.05, 'C2H4':  0.7, 'C2H6': 10.0,
        'CO':   100.0, 'CO2': 1000.0, 'moisture': 12.0,
        'oil_temp': 50.0, 'winding_temp_extra': 15.0,
    },
    '1': {
        'name':    '局部放电',
        'desc':    'H2极高，CH4中等，C2H2/C2H4和C2H4/C2H6均低，CH4/H2≈0.5',
        # IEC: CH4/H2=0.5(1), C2H2/C2H4=0.04(0), C2H4/C2H6=0.83(0) → 100
        # Duval: CH4%≈99% → PD区域
        'H2':  1000.0, 'CH4': 500.0, 'C2H2':  0.2, 'C2H4':  5.0, 'C2H6':  6.0,
        'CO':   125.0, 'CO2': 1000.0, 'moisture': 20.0,
        'oil_temp': 45.0, 'winding_temp_extra': 18.0,
    },
    '2': {
        'name':    '低能量放电',
        'desc':    'C2H2显著升高，CH4/H2和C2H2/C2H4均在中等范围，C2H4/C2H6<1',
        # IEC: CH4/H2=0.8(1), C2H2/C2H4=1.5(1), C2H4/C2H6=2.0(1) → 111
        'H2':   500.0, 'CH4': 400.0, 'C2H2': 150.0, 'C2H4': 100.0, 'C2H6': 50.0,
        'CO':   150.0, 'CO2':  900.0, 'moisture': 25.0,
        'oil_temp': 55.0, 'winding_temp_extra': 22.0,
    },
    '3': {
        'name':    '高能量放电',
        'desc':    'C2H2极高，CH4/H2>1，C2H2/C2H4>3，C2H4/C2H6>3，放电能量大',
        # IEC: CH4/H2=1.5(2), C2H2/C2H4=5.0(2), C2H4/C2H6=4.0(2) → 222 → 高能量放电
        'H2':   500.0, 'CH4': 750.0, 'C2H2':1000.0, 'C2H4': 200.0, 'C2H6': 50.0,
        'CO':   180.0, 'CO2':  900.0, 'moisture': 30.0,
        'oil_temp': 60.0, 'winding_temp_extra': 28.0,
    },
    '4': {
        'name':    '热故障 <150°C',
        'desc':    'CH4和C2H4升高，C2H4/C2H6在1~3，C2H2极低，轻度过热',
        # IEC: CH4/H2=0.5(1), C2H2/C2H4=0.05(0), C2H4/C2H6=1.5(1) → 101 → 热故障
        'H2':   200.0, 'CH4': 100.0, 'C2H2':  5.0, 'C2H4': 100.0, 'C2H6': 67.0,
        'CO':   200.0, 'CO2':  800.0, 'moisture': 28.0,
        'oil_temp': 55.0, 'winding_temp_extra': 20.0,
    },
    '5': {
        'name':    '热故障 150~300°C',
        'desc':    'CH4/H2≈0.7，C2H4/C2H6≈2.5，中度过热，CO开始上升',
        # IEC: CH4/H2=0.7(1), C2H2/C2H4=0.15(1), C2H4/C2H6=2.5(1) → 111 (热故障方向)
        'H2':   400.0, 'CH4': 280.0, 'C2H2': 45.0, 'C2H4': 300.0, 'C2H6': 120.0,
        'CO':   250.0, 'CO2':  875.0, 'moisture': 32.0,
        'oil_temp': 65.0, 'winding_temp_extra': 25.0,
    },
    '6': {
        'name':    '热故障 300~700°C',
        'desc':    'CH4/H2>1，C2H4/C2H6>3，CO2/CO<3，严重过热',
        # IEC: CH4/H2=1.2(2), C2H2/C2H4=0.2(1), C2H4/C2H6=3.5(2) → 212 → 热故障高温
        'H2':   300.0, 'CH4': 360.0, 'C2H2':100.0, 'C2H4': 500.0, 'C2H6': 143.0,
        'CO':   350.0, 'CO2':  875.0, 'moisture': 38.0,
        'oil_temp': 75.0, 'winding_temp_extra': 35.0,
    },
    '7': {
        'name':    '热故障 >700°C',
        'desc':    'CH4/H2>1，C2H4极高，CO2/CO<2，绕组严重过热',
        # IEC: CH4/H2=1.5(2), C2H2/C2H4=0.3(1), C2H4/C2H6=4.0(2) → 212 → 高温热故障
        'H2':   400.0, 'CH4': 600.0, 'C2H2':300.0, 'C2H4':1000.0, 'C2H6': 250.0,
        'CO':   450.0, 'CO2':  900.0, 'moisture': 45.0,
        'oil_temp': 85.0, 'winding_temp_extra': 40.0,
    },
}

# ================= 选择场景 =================
print("=" * 52)
print("  变压器 DGA 仿真程序")
print("=" * 52)
print("请选择本次仿真的故障场景：\n")
for key, s in SCENARIOS.items():
    print(f"  [{key}] {s['name']}")
    print(f"       {s['desc']}")
print()
choice = input("输入编号（直接回车默认 0=正常老化）：").strip()
if choice not in SCENARIOS:
    choice = '0'

scene = SCENARIOS[choice]
print(f"\n✅ 已选择场景：{scene['name']}\n")

# ================= 初始化 InfluxDB =================
try:
    client    = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
except Exception as e:
    print(f"数据库连接失败：{e}")
    sys.exit(1)

# ================= 基准值（从选定场景读取） =================
base_H2       = scene['H2']
base_CH4      = scene['CH4']
base_C2H2     = scene['C2H2']
base_C2H4     = scene['C2H4']
base_C2H6     = scene['C2H6']
base_CO       = scene['CO']
base_CO2      = scene['CO2']
base_moisture = scene['moisture']
base_oil_temp = scene['oil_temp']
winding_extra = scene['winding_temp_extra']

is_running = True

def listen_for_stop():
    global is_running
    input()
    is_running = False

print("=" * 52)
print(f"🚀 开始模拟：{scene['name']}")
print("📈 每 3 秒写入一条数据，带昼夜负载/温度周期波动")
print("🛑 停止方式：按 Enter 安全退出")
print("=" * 52 + "\n")

threading.Thread(target=listen_for_stop, daemon=True).start()

step = 0

try:
    while is_running:
        t = step * 0.1

        # 负载率和环境温度周期波动（与场景无关）
        load_factor  = 0.65 + 0.25 * math.sin(t)
        ambient_temp = base_oil_temp + 10.0 * math.sin(t - 1.0)

        temperature_top     = ambient_temp + (35.0 * load_factor) + random.uniform(-0.5, 0.5)
        temperature_winding = temperature_top + (winding_extra * load_factor) + random.uniform(-0.5, 0.5)

        # 气体极缓慢增长（模拟老化积累），各场景增长速率等比例调整
        drift = scene['H2'] / 15.0  # 以正常老化场景为基准归一化漂移速率
        base_H2   += random.uniform(0.0, 0.01 * drift)
        base_CH4  += random.uniform(0.0, 0.005 * drift)
        base_C2H6 += random.uniform(0.0, 0.005 * drift)
        base_CO   += random.uniform(0.0, 0.05 * drift)
        base_CO2  += random.uniform(0.0, 0.20 * drift)

        # 传感器测量噪声
        noise = drift ** 0.5 * NOISE_SCALE
        H2   = max(0.0, base_H2   + random.uniform(-0.5, 0.5) * noise)
        CH4  = max(0.0, base_CH4  + random.uniform(-0.5, 0.5) * noise)
        C2H4 = max(0.0, base_C2H4 + random.uniform(-0.3, 0.3) * noise)
        C2H6 = max(0.0, base_C2H6 + random.uniform(-0.5, 0.5) * noise)
        C2H2 = max(0.0, base_C2H2 + random.uniform(-0.02, 0.05) * noise)
        CO   = max(0.0, base_CO   + random.uniform(-2.0, 2.0) * noise)
        CO2  = max(0.0, base_CO2  + random.uniform(-5.0, 5.0) * noise)
        moisture = max(0.0, base_moisture + random.uniform(-0.2, 0.2))

        point = (
            Point("transformer_001")
            .field("顶层油温",  round(temperature_top, 1))
            .field("绕组温度",  round(temperature_winding, 1))
            .field("微水含量",  round(moisture, 2))
            .field("H2",        round(H2, 1))
            .field("CH4",       round(CH4, 1))
            .field("C2H4",      round(C2H4, 1))
            .field("C2H6",      round(C2H6, 1))
            .field("C2H2",      round(C2H2, 2))
            .field("CO",        round(CO, 1))
            .field("CO2",       round(CO2, 1))
        )
        write_api.write(bucket=BUCKET, record=point)

        print(f"[{step:04d}][{scene['name']}] "
              f"油温:{temperature_top:4.1f}°C 绕组:{temperature_winding:4.1f}°C | "
              f"H2:{H2:6.1f} CH4:{CH4:6.1f} C2H4:{C2H4:6.1f} "
              f"C2H6:{C2H6:6.1f} C2H2:{C2H2:5.2f} | "
              f"CO:{CO:6.1f} CO2:{CO2:7.1f} 水分:{moisture:4.1f}")

        step += 1
        time.sleep(3)

except Exception as e:
    print(f"\n❌ 运行中发生错误: {e}")
finally:
    print("\n✅ 收到停止指令，正在安全关闭数据库连接...")
    client.close()
    print("👋 数据库连接已断开，模拟程序已安全退出。")
