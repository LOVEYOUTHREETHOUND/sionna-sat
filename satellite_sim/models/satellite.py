"""卫星模型"""

import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
from ..utils.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    calculate_elevation_azimuth,
    calculate_distance
)
from ..utils.geometry import (
    calculate_beam_coverage,
    calculate_beam_gain,
    calculate_path_loss
)
from ..utils.stk_interface import STKDataReader

class Satellite:
    """卫星类"""
    
    def __init__(self,
                 sat_id: str,
                 stk_data_file: str,
                 num_beams: int,
                 tx_power: float,
                 antenna_gain: float,
                 frequency: float):
        """初始化卫星对象
        
        Args:
            sat_id: 卫星ID
            stk_data_file: STK轨道数据文件路径
            num_beams: 波束数量
            tx_power: 发射功率(dBW)
            antenna_gain: 天线增益(dBi)
            frequency: 载波频率(Hz)
        """
        self.sat_id = sat_id
        self.num_beams = num_beams
        self.tx_power = tx_power
        self.antenna_gain = antenna_gain
        self.frequency = frequency
        
        # 加载STK数据
        self.stk_reader = STKDataReader(stk_data_file)
        
        # 初始化波束配置
        self._init_beams()
        
    def _init_beams(self):
        """初始化波束配置"""
        # 波束参数
        self.beam_width = 1.0  # 3dB波束宽度(度)
        self.beam_directions = []  # 波束指向
        
        # 根据波束数量计算波束指向
        if self.num_beams == 7:
            # 7波束配置：1个中心波束，6个外围波束
            # 中心波束
            self.beam_directions.append((0, 0))  # (俯仰角,方位角)
            
            # 6个外围波束，相位差60度
            for i in range(6):
                angle = i * 60  # 方位角
                self.beam_directions.append((self.beam_width, angle))
                
    def update_position(self, time: datetime):
        """更新卫星位置
        
        Args:
            time: 时间点
        """
        self.current_time = time
        self.position = self.stk_reader.get_satellite_position(time)
        self.velocity = self.stk_reader.get_satellite_velocity(time)
        self.attitude = self.stk_reader.get_satellite_attitude(time)
        
    def get_beam_coverage(self, cell_positions: List[Tuple[float, float, float]],
                         min_elevation: float = 10.0) -> Dict[int, List[int]]:
        """计算每个波束覆盖的小区
        
        Args:
            cell_positions: 小区中心ECEF坐标列表
            min_elevation: 最小仰角(度)
            
        Returns:
            coverage: {beam_id: [cell_id, ...], ...}波束覆盖字典
        """
        coverage = {}
        
        # 对每个波束计算覆盖
        for beam_id in range(self.num_beams):
            coverage[beam_id] = calculate_beam_coverage(
                self.position,
                cell_positions,
                min_elevation
            )
            
        return coverage
        
    def calculate_link_budget(self, ground_pos: Tuple[float, float, float],
                            beam_id: int,
                            rx_gain: float,
                            noise_figure: float,
                            temperature: float,
                            bandwidth: float,
                            rain_rate: float = 0.0) -> Dict[str, float]:
        """计算链路预算
        
        Args:
            ground_pos: 地面站ECEF坐标
            beam_id: 波束ID
            rx_gain: 接收天线增益(dB)
            noise_figure: 噪声系数(dB)
            temperature: 系统温度(K)
            bandwidth: 带宽(Hz)
            rain_rate: 降雨率(mm/h)
            
        Returns:
            budget: 链路预算结果字典
        """
        # 计算距离
        distance = calculate_distance(self.position, ground_pos)
        
        # 计算波束增益
        tx_gain = calculate_beam_gain(
            self.position,
            ground_pos,
            self.antenna_gain,
            self.beam_width
        )
        
        # 计算路径损耗
        path_loss = calculate_path_loss(
            distance,
            self.frequency,
            rain_rate
        )
        
        # 计算接收功率
        rx_power = self.tx_power + tx_gain + rx_gain - path_loss
        
        # 计算噪声功率
        k = 1.38e-23  # 玻尔兹曼常数
        noise_power = 10 * np.log10(k * temperature * bandwidth) + noise_figure
        
        # 计算SNR
        snr = rx_power - noise_power
        
        # 返回链路预算结果
        return {
            'distance': distance,
            'tx_gain': tx_gain,
            'path_loss': path_loss,
            'rx_power': rx_power,
            'noise_power': noise_power,
            'snr': snr
        }
        
    def get_coverage_stats(self, cell_positions: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """获取覆盖统计信息
        
        Args:
            cell_positions: 小区中心ECEF坐标列表
            
        Returns:
            stats: 覆盖统计信息字典
        """
        # 获取波束覆盖
        coverage = self.get_beam_coverage(cell_positions)
        
        # 统计覆盖的小区数
        total_cells = len(cell_positions)
        covered_cells = set()
        for beam_cells in coverage.values():
            covered_cells.update(beam_cells)
        num_covered = len(covered_cells)
        
        # 计算覆盖率
        coverage_rate = num_covered / total_cells * 100
        
        # 计算每个波束的平均仰角
        beam_elevations = {}
        for beam_id, beam_cells in coverage.items():
            if beam_cells:
                elevations = []
                for cell_id in beam_cells:
                    cell_pos = cell_positions[cell_id]
                    elevation, _ = calculate_elevation_azimuth(self.position, cell_pos)
                    elevations.append(elevation)
                beam_elevations[beam_id] = np.mean(elevations)
            else:
                beam_elevations[beam_id] = 0
                
        return {
            'total_cells': total_cells,
            'covered_cells': num_covered,
            'coverage_rate': coverage_rate,
            'beam_elevations': beam_elevations
        } 