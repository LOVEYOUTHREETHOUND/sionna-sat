"""坐标转换工具"""

import numpy as np
from typing import Tuple, Union

def geodetic_to_ecef(lat: float, lon: float, h: float) -> Tuple[float, float, float]:
    """将大地坐标系(纬度,经度,高度)转换为地心地固坐标系(ECEF)
    
    Args:
        lat: 纬度(度)
        lon: 经度(度)
        h: 高度(米)
        
    Returns:
        x, y, z: ECEF坐标(米)
    """
    # WGS84椭球体参数
    a = 6378137.0  # 长半轴(m)
    f = 1/298.257223563  # 扁率
    b = a * (1 - f)  # 短半轴
    e2 = 1 - (b/a)**2  # 第一偏心率平方
    
    # 转换为弧度
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # 计算卯酉圈曲率半径
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # 计算ECEF坐标
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + h) * np.sin(lat_rad)
    
    return x, y, z

def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """将ECEF坐标转换为大地坐标系
    
    Args:
        x, y, z: ECEF坐标(米)
        
    Returns:
        lat: 纬度(度)
        lon: 经度(度)
        h: 高度(米)
    """
    # WGS84椭球体参数
    a = 6378137.0
    f = 1/298.257223563
    b = a * (1 - f)
    e2 = 1 - (b/a)**2
    
    # 计算经度
    lon = np.arctan2(y, x)
    
    # 迭代计算纬度和高度
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    for _ in range(5):  # 通常5次迭代足够
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N/(N + h)))
    
    return np.degrees(lat), np.degrees(lon), h

def enu_to_ecef(e: float, n: float, u: float, 
                ref_lat: float, ref_lon: float, ref_h: float) -> Tuple[float, float, float]:
    """将局部东北天(ENU)坐标转换为ECEF坐标
    
    Args:
        e, n, u: 东、北、天坐标(米)
        ref_lat: 参考点纬度(度)
        ref_lon: 参考点经度(度)
        ref_h: 参考点高度(米)
        
    Returns:
        x, y, z: ECEF坐标(米)
    """
    # 转换为弧度
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    # 旋转矩阵
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # ENU到ECEF的转换矩阵
    R = np.array([
        [-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon],
        [cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon],
        [0, cos_lat, sin_lat]
    ])
    
    # 计算ECEF坐标增量
    enu = np.array([e, n, u])
    xyz_delta = R @ enu
    
    # 获取参考点的ECEF坐标
    x_ref, y_ref, z_ref = geodetic_to_ecef(ref_lat, ref_lon, ref_h)
    
    # 计算最终ECEF坐标
    x = x_ref + xyz_delta[0]
    y = y_ref + xyz_delta[1]
    z = z_ref + xyz_delta[2]
    
    return x, y, z

def ecef_to_enu(x: float, y: float, z: float,
                ref_lat: float, ref_lon: float, ref_h: float) -> Tuple[float, float, float]:
    """将ECEF坐标转换为局部东北天(ENU)坐标
    
    Args:
        x, y, z: ECEF坐标(米)
        ref_lat: 参考点纬度(度)
        ref_lon: 参考点经度(度)
        ref_h: 参考点高度(米)
        
    Returns:
        e, n, u: 东、北、天坐标(米)
    """
    # 获取参考点的ECEF坐标
    x_ref, y_ref, z_ref = geodetic_to_ecef(ref_lat, ref_lon, ref_h)
    
    # 计算相对坐标
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref
    
    # 转换为弧度
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    # 旋转矩阵
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # ECEF到ENU的转换矩阵
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    ])
    
    # 计算ENU坐标
    xyz_delta = np.array([dx, dy, dz])
    enu = R @ xyz_delta
    
    return enu[0], enu[1], enu[2]

def calculate_elevation_azimuth(sat_pos: Tuple[float, float, float],
                              ground_pos: Tuple[float, float, float]) -> Tuple[float, float]:
    """计算地面站到卫星的仰角和方位角
    
    Args:
        sat_pos: 卫星ECEF坐标(x,y,z)(米)
        ground_pos: 地面站ECEF坐标(x,y,z)(米)
        
    Returns:
        elevation: 仰角(度)
        azimuth: 方位角(度)
    """
    # 转换为地面站的局部ENU坐标系
    ground_lat, ground_lon, ground_h = ecef_to_geodetic(*ground_pos)
    e, n, u = ecef_to_enu(*sat_pos, ground_lat, ground_lon, ground_h)
    
    # 计算仰角
    r = np.sqrt(e**2 + n**2 + u**2)
    elevation = np.arcsin(u/r)
    
    # 计算方位角
    azimuth = np.arctan2(e, n)
    if azimuth < 0:
        azimuth += 2*np.pi
    
    return np.degrees(elevation), np.degrees(azimuth)

def calculate_distance(pos1: Tuple[float, float, float],
                      pos2: Tuple[float, float, float]) -> float:
    """计算两点间的欧几里得距离
    
    Args:
        pos1: 第一个点的坐标(x,y,z)
        pos2: 第二个点的坐标(x,y,z)
        
    Returns:
        distance: 距离(米)
    """
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2))) 