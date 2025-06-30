"""信道模型"""

import numpy as np
from typing import List, Tuple, Dict
from ..utils.coordinates import calculate_distance, calculate_elevation_azimuth
from ..utils.geometry import calculate_path_loss, calculate_beam_gain

class ChannelModel:
    """卫星信道模型类"""
    
    def __init__(self,
                 frequency: float,
                 bandwidth: float,
                 noise_figure: float,
                 temperature: float,
                 rain_rate: float = 0.0):
        """初始化信道模型
        
        Args:
            frequency: 载波频率(Hz)
            bandwidth: 系统带宽(Hz)
            noise_figure: 噪声系数(dB)
            temperature: 系统温度(K)
            rain_rate: 降雨率(mm/h)
        """
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.noise_figure = noise_figure
        self.temperature = temperature
        self.rain_rate = rain_rate
        
        # 常量
        self.k = 1.38e-23  # 玻尔兹曼常数
        self.c = 3e8      # 光速
        
        # 计算噪声功率
        self.noise_power = self.k * self.temperature * self.bandwidth * 10**(self.noise_figure/10)
        
    def calculate_channel_matrix(self,
                               sat_pos: Tuple[float, float, float],
                               ground_pos: Tuple[float, float, float],
                               sat_gain: float,
                               ground_gain: float,
                               beam_width: float) -> Dict[str, float]:
        """计算信道矩阵
        
        Args:
            sat_pos: 卫星ECEF坐标(x,y,z)
            ground_pos: 地面站ECEF坐标(x,y,z)
            sat_gain: 卫星天线最大增益(dB)
            ground_gain: 地面站天线增益(dB)
            beam_width: 波束宽度(度)
            
        Returns:
            channel: 信道参数字典
        """
        # 计算距离
        distance = calculate_distance(sat_pos, ground_pos)
        
        # 计算仰角和方位角
        elevation, azimuth = calculate_elevation_azimuth(sat_pos, ground_pos)
        
        # 计算波束增益
        beam_gain = calculate_beam_gain(
            sat_pos,
            ground_pos,
            sat_gain,
            beam_width
        )
        
        # 计算路径损耗
        path_loss = calculate_path_loss(
            distance,
            self.frequency,
            self.rain_rate
        )
        
        # 计算接收天线增益
        rx_gain = ground_gain
        
        # 计算噪声功率
        noise_power = 10 * np.log10(self.k * self.temperature * self.bandwidth) + \
            self.noise_figure
            
        return {
            'distance': distance,
            'elevation': elevation,
            'azimuth': azimuth,
            'beam_gain': beam_gain,
            'path_loss': path_loss,
            'rx_gain': rx_gain,
            'noise_power': noise_power
        }
        
    def calculate_channel_response(self,
                                 satellite_position: np.ndarray,
                                 user_positions: List[np.ndarray],
                                 satellite_gain: float,
                                 user_gain: float,
                                 beam_width: float,
                                 tx_power: float) -> np.ndarray:
        """计算信道响应
        
        Args:
            satellite_position: 卫星位置[x,y,z]
            user_positions: 用户位置列表[[x,y,z],...]
            satellite_gain: 卫星天线增益(dB)
            user_gain: 用户天线增益(dB)
            beam_width: 波束宽度(度)
            tx_power: 发射功率(W)
            
        Returns:
            H: 信道矩阵[num_users, num_subcarriers]
        """
        num_users = len(user_positions)
        num_subcarriers = int(self.bandwidth / 15e3)  # 假设15kHz子载波间隔
        
        # 初始化信道矩阵
        H = np.zeros((num_users, num_subcarriers), dtype=np.complex64)
        
        # 对每个用户计算信道响应
        for i, user_pos in enumerate(user_positions):
            # 1. 计算路径损耗
            distance = np.linalg.norm(satellite_position - user_pos)
            path_loss = calculate_path_loss(
                distance=distance,
                frequency=self.frequency,
                rain_rate=self.rain_rate
            )
            
            # 2. 计算天线增益
            # 简化模型：如果用户在波束覆盖范围内，使用最大增益
            # 实际应该根据角度计算精确的天线方向图
            total_gain = satellite_gain + user_gain  # dB
            
            # 3. 计算接收信号功率
            rx_power = tx_power * 10**((total_gain - path_loss)/10)
            
            # 4. 生成信道系数
            # 对每个子载波添加随机相位
            phases = np.random.uniform(0, 2*np.pi, num_subcarriers)
            H[i, :] = np.sqrt(rx_power) * np.exp(1j * phases)
            
        return H
        
    def add_doppler_effect(self,
                          H: np.ndarray,
                          satellite_velocity: np.ndarray,
                          user_positions: List[np.ndarray],
                          satellite_position: np.ndarray) -> np.ndarray:
        """添加多普勒效应
        
        Args:
            H: 原始信道矩阵[num_users, num_subcarriers]
            satellite_velocity: 卫星速度向量[vx,vy,vz]
            user_positions: 用户位置列表
            satellite_position: 卫星位置
            
        Returns:
            H: 添加多普勒效应后的信道矩阵
        """
        c = 3e8  # 光速
        
        for i, user_pos in enumerate(user_positions):
            # 计算卫星到用户的单位向量
            direction = (user_pos - satellite_position)
            direction = direction / np.linalg.norm(direction)
            
            # 计算多普勒频移
            doppler_freq = np.dot(satellite_velocity, direction) * self.frequency / c
            
            # 添加相位旋转
            t = np.arange(H.shape[1]) / (self.bandwidth / H.shape[1])  # 时间点
            phase_rotation = np.exp(1j * 2 * np.pi * doppler_freq * t)
            
            # 应用多普勒效应
            H[i, :] = H[i, :] * phase_rotation
            
        return H
        
    def add_phase_noise(self,
                       H: np.ndarray,
                       phase_noise_var: float = 0.1) -> np.ndarray:
        """添加相位噪声
        
        Args:
            H: 信道响应矩阵
            phase_noise_var: 相位噪声方差(弧度^2)
            
        Returns:
            H: 添加相位噪声后的信道响应
        """
        phase_noise = np.random.normal(0, np.sqrt(phase_noise_var), H.shape)
        return H * np.exp(1j * phase_noise)
        
    def calculate_snr(self,
                     H: np.ndarray,
                     tx_power: float) -> np.ndarray:
        """计算信噪比
        
        Args:
            H: 信道矩阵[num_users, num_subcarriers]
            tx_power: 发射功率(W)
            
        Returns:
            snr: 每个用户的信噪比[num_users]
        """
        # 计算接收信号功率
        rx_power = np.mean(np.abs(H)**2, axis=1) * tx_power
        
        # 计算SNR
        snr = rx_power / self.noise_power
        
        return 10 * np.log10(snr)  # 转换为dB 