import time
import random
import math
import threading
import sys
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ================= 配置区域 =================
URL    = "http://localhost:8086"
TOKEN  = "my-influxdb-super-secret-token"
ORG    = "lab"
BUCKET = "transformer"

# 初始化数据库客户端
try:
    client    = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
except Exception as e:
    print(f"数据库连接失败，请检查 InfluxDB 是否启动以及 Token 是否正确。\n错误信息: {e}")
    sys.exit(1)

# ================= 变压器正常运行基准值 =================
# 气体基数（单位：ppm），正常情况下非常稳定
base_H2   = 15.0
base_CH4  = 10.0
base_C2H4 = 8.0
base_C2H6 = 20.0  # 乙烷：正常值约 10~50 ppm
base_C2H2 = 0.0   # 乙炔：正常变压器应趋近于 0
base_CO   = 120.0
base_CO2  = 800.0  # 二氧化碳：正常值约 500~3000 ppm，与纸绝缘老化相关
base_moisture = 12.0

# 全局运行状态标志
is_running = True

# ================= 优雅停止监听线程 =================
def listen_for_stop():
    global is_running
    # input() 会阻塞当前线程，但不会阻塞主线程的数据模拟
    input()
    is_running = False

# ================= 主程序 =================
print("==================================================")
print("🚀 开始模拟变压器实际运行工况...")
print("📈 特征：带有昼夜负载/温度周期波动，气体数值稳定且含传感器噪声。")
print("🛑 停止方式：随时在终端按下 【回车键 (Enter)】 安全退出。")
print("==================================================\n")

# 启动后台监听线程 (daemon=True 确保主程序退出时线程自动销毁)
threading.Thread(target=listen_for_stop, daemon=True).start()

step = 0

try:
    while is_running:
        # t 模拟时间流逝，控制周期循环。
        t = step * 0.1

        # 1. 模拟电网负载率波动 (范围约 0.4 到 0.9)
        load_factor = 0.65 + 0.25 * math.sin(t)

        # 2. 模拟环境温度波动 (范围约 15°C 到 35°C，存在一定相位差)
        ambient_temp = 25.0 + 10.0 * math.sin(t - 1.0)

        # 3. 计算物理相关的温度值
        # 顶层油温 = 环境温度 + (负载带来的温升) + 传感器微小噪声
        temperature_top = ambient_temp + (35.0 * load_factor) + random.uniform(-0.5, 0.5)
        # 绕组温度 = 顶层油温 + (绕组与油的温差，受负载直接影响) + 传感器微小噪声
        temperature_winding = temperature_top + (15.0 * load_factor) + random.uniform(-0.5, 0.5)

        # 4. 模拟气体和水分的极缓慢老化与传感器测量噪声
        base_H2   += random.uniform(0.0, 0.01)
        base_CH4  += random.uniform(0.0, 0.005)
        base_C2H6 += random.uniform(0.0, 0.005)  # 乙烷：缓慢老化积累
        base_CO   += random.uniform(0.0, 0.05)
        base_CO2  += random.uniform(0.0, 0.20)   # CO2：随纸绝缘老化缓慢增长

        # 实际采集值 = 基础值 + 传感器高频测量噪声 (使用 max(0, x) 避免出现负数)
        H2   = max(0.0, base_H2   + random.uniform(-0.5, 0.5))
        CH4  = max(0.0, base_CH4  + random.uniform(-0.5, 0.5))
        C2H4 = max(0.0, base_C2H4 + random.uniform(-0.3, 0.3))
        C2H6 = max(0.0, base_C2H6 + random.uniform(-0.5, 0.5))
        C2H2 = max(0.0, base_C2H2 + random.uniform(-0.02, 0.05))  # 乙炔保持极低
        CO   = max(0.0, base_CO   + random.uniform(-2.0, 2.0))
        CO2  = max(0.0, base_CO2  + random.uniform(-5.0, 5.0))
        moisture = max(0.0, base_moisture + random.uniform(-0.2, 0.2))

        # 5. 封装并写入 InfluxDB
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

        # 6. 终端打印输出
        print(f"[{step:04d}] 负载率: {load_factor*100:2.0f}% | "
              f"油温: {temperature_top:4.1f}°C | 绕组: {temperature_winding:4.1f}°C | "
              f"H2: {H2:4.1f} | CH4: {CH4:4.1f} | C2H4: {C2H4:4.1f} | "
              f"C2H6: {C2H6:4.1f} | C2H2: {C2H2:4.2f} | "
              f"CO: {CO:5.1f} | CO2: {CO2:6.1f} | 水分: {moisture:4.1f}")

        step += 1
        time.sleep(3)

except Exception as e:
    print(f"\n❌ 运行中发生错误: {e}")
finally:
    # 捕获到退出信号后的收尾工作
    print("\n✅ 收到停止指令，正在安全关闭数据库连接...")
    client.close()
    print("👋 数据库连接已断开，模拟程序已安全退出。")
