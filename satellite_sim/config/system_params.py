"""系统参数配置"""

import numpy as np

# 卫星参数
SATELLITE_PARAMS = {
    'height': 500000,  # 卫星高度(m)
    'num_beams': 7,    # 波束数量
    'tx_power': 40,    # 发射功率(dBW)
    'antenna_gain': 30,  # 天线增益(dBi)
    'frequency': 20e9,  # 载波频率(Hz)
}

# 地面小区参数
CELL_PARAMS = {
    'radius': 22600,     # 小区半径(m)
    'num_users': 10,    # 每个小区的用户数
    'min_elevation': 10,  # 最小仰角(度)
}

# 信道参数
CHANNEL_PARAMS = {
    'noise_figure': 3,     # 噪声系数(dB)
    'temperature': 290,    # 系统温度(K)
    'bandwidth': 50e6,     # 系统带宽(Hz)
    'rain_rate': 0,        # 降雨率(mm/h)
}

# 仿真参数
SIM_PARAMS = {
    'num_slots': 1000,     # 仿真时隙数
    'slot_duration': 1e-3,  # 时隙长度(s)
}

# 常量
CONSTANTS = {
    'c': 3e8,              # 光速(m/s)
    'k': 1.38e-23,         # 玻尔兹曼常数
    'pi': np.pi,
}

# 坐标范围（经纬度）
COORD_RANGE = {
    'lat_min': 30,         # 最小纬度
    'lat_max': 40,         # 最大纬度
    'lon_min': 110,        # 最小经度
    'lon_max': 120,        # 最大经度
} 