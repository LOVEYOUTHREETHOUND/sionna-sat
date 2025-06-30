"""几何计算工具"""

import numpy as np
from typing import List, Tuple, Union
from .coordinates import calculate_distance, calculate_elevation_azimuth

def create_hexagon_grid(center_lat: float, center_lon: float, radius: float, 
                       num_rings: int) -> List[Tuple[float, float]]:
    """创建六边形网格
    
    Args:
        center_lat: 中心纬度(度)
        center_lon: 中心经度(度)
        radius: 小区半径(米)
        num_rings: 环数
        
    Returns:
        centers: 所有小区中心点的坐标列表[(lat, lon), ...]
    """
    centers = [(center_lat, center_lon)]  # 中心小区
    
    # 六边形网格的基本向量
    angles = np.arange(0, 360, 60)  # 六个方向
    
    # 计算经纬度的缩放因子
    lat_scale = 111320  # 1度纬度对应的米数
    lon_scale = 111320 * np.cos(np.radians(center_lat))  # 1度经度对应的米数
    
    for ring in range(1, num_rings + 1):
        # 对于每一环
        for i in range(6):  # 六个方向
            # 计算这个方向上的点
            for j in range(ring):
                angle = np.radians(angles[i])
                # 计算偏移距离
                dx = radius * (2 * ring - j) * np.cos(angle)
                dy = radius * (2 * ring - j) * np.sin(angle)
                
                # 转换为经纬度偏移
                dlat = dy / lat_scale
                dlon = dx / lon_scale
                
                # 添加新的中心点
                centers.append((center_lat + dlat, center_lon + dlon))
                
                # 如果不是最后一个点，添加六边形边上的点
                if j < ring - 1:
                    next_angle = np.radians(angles[(i + 1) % 6])
                    dx = radius * (2 * ring - j - 1) * np.cos(next_angle)
                    dy = radius * (2 * ring - j - 1) * np.sin(next_angle)
                    dlat = dy / lat_scale
                    dlon = dx / lon_scale
                    centers.append((center_lat + dlat, center_lon + dlon))
    
    return centers

def calculate_beam_coverage(sat_pos: Tuple[float, float, float],
                          cell_positions: List[Tuple[float, float, float]],
                          min_elevation: float = 10.0) -> List[int]:
    """计算卫星波束覆盖的小区
    
    Args:
        sat_pos: 卫星ECEF坐标(x,y,z)
        cell_positions: 小区中心ECEF坐标列表[(x,y,z),...]
        min_elevation: 最小仰角(度)
        
    Returns:
        covered_cells: 被覆盖的小区索引列表
    """
    covered_cells = []
    
    for i, cell_pos in enumerate(cell_positions):
        # 计算仰角
        elevation, _ = calculate_elevation_azimuth(sat_pos, cell_pos)
        
        # 如果仰角大于最小仰角，认为小区被覆盖
        if elevation >= min_elevation:
            covered_cells.append(i)
            
    return covered_cells

def calculate_beam_gain(sat_pos: Tuple[float, float, float],
                       ground_pos: Tuple[float, float, float],
                       max_gain: float,
                       beam_width: float) -> float:
    """计算波束增益
    
    使用简化的抛物面天线模型
    
    Args:
        sat_pos: 卫星ECEF坐标(x,y,z)
        ground_pos: 地面点ECEF坐标(x,y,z)
        max_gain: 波束中心最大增益(dB)
        beam_width: 3dB波束宽度(度)
        
    Returns:
        gain: 波束增益(dB)
    """
    # 计算到波束中心的角度
    elevation, azimuth = calculate_elevation_azimuth(sat_pos, ground_pos)
    
    # 使用简化的抛物面天线模型
    theta = np.sqrt(elevation**2 + azimuth**2)  # 到波束中心的角度
    gain = max_gain - 12 * (theta / beam_width)**2
    
    # 限制最小增益
    min_gain = max_gain - 30  # 假设旁瓣电平为-30dB
    gain = max(gain, min_gain)
    
    return gain

def calculate_path_loss(distance: float, frequency: float, 
                       rain_rate: float = 0.0) -> float:
    """计算路径损耗
    
    包括自由空间损耗和雨衰
    
    Args:
        distance: 距离(米)
        frequency: 频率(Hz)
        rain_rate: 降雨率(mm/h)
        
    Returns:
        path_loss: 路径损耗(dB)
    """
    # 自由空间损耗
    c = 3e8  # 光速
    wavelength = c / frequency
    fspl = 20 * np.log10(4 * np.pi * distance / wavelength)
    
    # 雨衰(使用简化模型)
    if rain_rate > 0:
        # 简化的ITU-R P.838模型
        k = 0.124  # k系数(与频率和极化方式有关)
        alpha = 1.6  # alpha系数
        rain_loss = k * rain_rate**alpha * distance/1000  # 转换为km
    else:
        rain_loss = 0
        
    return fspl + rain_loss

def calculate_snr(tx_power: float, tx_gain: float, rx_gain: float,
                 path_loss: float, noise_figure: float,
                 temperature: float, bandwidth: float) -> float:
    """计算信噪比
    
    Args:
        tx_power: 发射功率(dBW)
        tx_gain: 发射增益(dB)
        rx_gain: 接收增益(dB)
        path_loss: 路径损耗(dB)
        noise_figure: 噪声系数(dB)
        temperature: 系统温度(K)
        bandwidth: 带宽(Hz)
        
    Returns:
        snr: 信噪比(dB)
    """
    # 计算接收功率
    rx_power = tx_power + tx_gain + rx_gain - path_loss
    
    # 计算噪声功率
    k = 1.38e-23  # 玻尔兹曼常数
    noise_power = 10 * np.log10(k * temperature * bandwidth) + noise_figure
    
    # 计算SNR
    snr = rx_power - noise_power
    
    return snr 