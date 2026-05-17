# 变压器油中溶解气体实时监测系统

基于 DGA（溶解气体分析）的变压器运行状态监测平台。通过采集变压器油中的特征气体浓度、油温、绕组温度等参数，实时写入时序数据库，并在 Grafana 看板中可视化展示，用于辅助判断变压器潜伏性故障。

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| InfluxDB | 2.7.1 | 时序数据存储 |
| Grafana | 10.2.3 | 数据可视化 |
| Python | 3.x | 数据采集 / 诊断 |

## 项目结构

```
.
├── docker-compose.yml              # InfluxDB + Grafana 容器编排
├── requirements.txt                # Python 依赖
├── code/
│   ├── simulate.py                 # 传感器数据模拟（对接真实传感器时替换此文件）
│   ├── main_pipeline.py            # 数据处理与诊断管道
│   ├── preprocessor.py             # 3-Sigma 异常检测 + S-G 平滑滤波
│   ├── feature_extractor.py        # IEC 三比值、CO/CO2、总烃等特征提取
│   ├── random_forest_diagnosis.py  # 随机森林故障分类器
│   └── fusion_diagnosis.py         # Duval 三角形 + 随机森林融合诊断
└── grafana/
    ├── plugins/                    # 预装插件（volkovlabs-echarts-panel）
    └── provisioning/
        ├── datasources/            # InfluxDB 数据源自动配置
        └── dashboards/             # 看板自动配置与 JSON 定义
```

## 两个 Python 程序的作用

本项目的 Python 部分由两个**独立运行**的程序组成，互不依赖，通过 InfluxDB 传递数据。

---

### `simulate.py` — 数据来源

```bash
python3 code/simulate.py
```

启动后会出现场景选择菜单，选择本次仿真的故障类型：

```
[0] 正常老化        [1] 局部放电        [2] 低能量放电      [3] 高能量放电
[4] 热故障 <150°C   [5] 热故障 150~300°C [6] 热故障 300~700°C [7] 热故障 >700°C
```

直接回车默认选 0（正常老化）。

**作用：** 模拟变压器传感器，每 3 秒生成一条与所选故障类型匹配的带噪声运行数据（油温、绕组温度、H₂/CH₄/C₂H₂/C₂H₄/C₂H₆/CO/CO₂/微水），写入 InfluxDB 的 `transformer_001` 测量表。各场景的气体基准值参照 IEC 60599 典型值设定，确保 IEC 三比值特征和 Duval 三角形坐标均落入对应故障区域。

**适用场景：** 没有真实传感器时用于演示和开发。接入真实传感器后，**替换此文件**即可，其余代码无需改动——只要传感器程序按相同格式向 `transformer_001` 写数据即可。

**单独运行时的效果：** Grafana 看板中的原始数据图表（油温、气体浓度、微水、Duval 三角形、IEC 三比值）正常刷新，但**没有**融合诊断结果（"融合诊断结果"面板无数据）。

---

### `main_pipeline.py` — 诊断管道

```bash
python3 code/main_pipeline.py
```

**作用：** 每 30 秒从 InfluxDB 读取最新原始数据，经过以下流程后将诊断结论写回 InfluxDB 的 `transformer_001_diagnosis` 测量表：

```
原始数据 → 3-Sigma 异常检测 + S-G 平滑滤波
        → IEC 三比值 / CO/CO₂ / 总烃 / 微水 / 温度特征提取
        → 随机森林分类 + Duval 三角形 I 判断
        → 加权融合（RF 60% + Duval 40%）
        → 写入诊断结果（故障类型 + 置信度）
```

**注意：** 首次运行会自动训练随机森林模型（约 10 秒），训练完成后保存到 `code/models/random_forest_model.pkl`，之后直接加载。

**单独运行时的效果：** 若 InfluxDB 中已有 `simulate.py` 或真实传感器写入的数据，管道正常工作并更新诊断结果；若无原始数据，管道会提示"无数据，跳过"并等待下一轮。

---

### 两者同时运行 — 完整功能

推荐用 tmux 开两个窗格：

```bash
# 新建 tmux 会话
tmux new -s transformer

# 上下分屏：Ctrl+b 然后 "
# 上方窗格
python3 code/simulate.py

# 切换到下方窗格：Ctrl+b 然后 ↓
python3 code/main_pipeline.py
```

此时 Grafana 看板所有面板均有数据：原始监测数据实时滚动，融合诊断结果每 30 秒更新一次。

---

## 快速开始

### 1. 启动数据库与看板服务

```bash
docker compose up -d
```

首次启动自动完成 InfluxDB 初始化和 Grafana 数据源/看板注入。

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> **WSL / Linux** 若提示 PEP 668 错误，改用：
> ```bash
> pip install --break-system-packages -r requirements.txt
> ```

### 3. 启动程序

按需选择：

| 目的 | 命令 |
|------|------|
| 仅查看原始监测数据 | `python3 code/simulate.py` |
| 完整功能（监测 + 诊断） | 同时运行两个程序（见上方 tmux 用法） |

### 4. 查看监测看板

浏览器打开 [http://localhost:3000](http://localhost:3000)，账号 `admin` / `admin`。

## 监测指标

| 指标 | 说明 |
|------|------|
| 顶层油温 / 绕组温度 | 随负载率和环境温度周期波动 |
| H₂、CH₄、C₂H₄、C₂H₆、C₂H₂ | 油中特征溶解气体（ppm） |
| CO、CO₂ | 纸绝缘老化相关气体，随运行时间缓慢增长 |
| 微水含量 | 绝缘油水分（ppm） |
| 融合诊断结论 | 故障类型 + 置信度（需运行 main_pipeline.py） |

## 默认配置

| 项目 | 值 |
|------|----|
| InfluxDB 地址 | http://localhost:8086 |
| InfluxDB Token | `my-influxdb-super-secret-token` |
| Grafana 地址 | http://localhost:3000 |
| Grafana 账号 | admin / admin |
