"""STK数据接口"""

import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime, timedelta

class STKDataReader:
    """STK数据读取接口"""
    
    def __init__(self, data_file: str):
        """初始化STK数据读取器
        
        Args:
            data_file: STK导出的数据文件路径
        """
        self.data_file = data_file
        self.data = None
        self._load_data()
        
    def _load_data(self):
        """加载STK数据文件"""
        try:
            # 假设STK导出的是CSV格式
            self.data = pd.read_csv(self.data_file)
            
            # 转换时间列
            if 'Time' in self.data.columns:
                self.data['Time'] = pd.to_datetime(self.data['Time'])
                
        except Exception as e:
            print(f"Error loading STK data: {e}")
            self.data = None
            
    def get_satellite_position(self, time: datetime) -> Tuple[float, float, float]:
        """获取指定时间的卫星位置
        
        Args:
            time: 时间点
            
        Returns:
            position: (x, y, z)卫星ECEF坐标(米)
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        # 找到最接近的时间点
        idx = (self.data['Time'] - time).abs().idxmin()
        row = self.data.iloc[idx]
        
        # 假设STK数据中的列名为'X', 'Y', 'Z'
        return (row['X'], row['Y'], row['Z'])
        
    def get_satellite_velocity(self, time: datetime) -> Tuple[float, float, float]:
        """获取指定时间的卫星速度
        
        Args:
            time: 时间点
            
        Returns:
            velocity: (vx, vy, vz)卫星ECEF速度(米/秒)
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        idx = (self.data['Time'] - time).abs().idxmin()
        row = self.data.iloc[idx]
        
        # 假设STK数据中的列名为'VX', 'VY', 'VZ'
        return (row['VX'], row['VY'], row['VZ'])
        
    def get_satellite_attitude(self, time: datetime) -> Tuple[float, float, float]:
        """获取指定时间的卫星姿态
        
        Args:
            time: 时间点
            
        Returns:
            attitude: (roll, pitch, yaw)卫星姿态角(度)
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        idx = (self.data['Time'] - time).abs().idxmin()
        row = self.data.iloc[idx]
        
        # 假设STK数据中的列名为'Roll', 'Pitch', 'Yaw'
        return (row['Roll'], row['Pitch'], row['Yaw'])
        
    def get_time_range(self) -> Tuple[datetime, datetime]:
        """获取数据的时间范围
        
        Returns:
            start_time, end_time: 数据起止时间
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        return (self.data['Time'].min(), self.data['Time'].max())
        
    def get_positions_in_range(self, start_time: datetime, 
                             end_time: datetime) -> List[Tuple[datetime, Tuple[float, float, float]]]:
        """获取时间范围内的所有卫星位置
        
        Args:
            start_time: 起始时间
            end_time: 结束时间
            
        Returns:
            positions: [(time, (x,y,z)), ...]位置列表
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        mask = (self.data['Time'] >= start_time) & (self.data['Time'] <= end_time)
        data_in_range = self.data[mask]
        
        positions = []
        for _, row in data_in_range.iterrows():
            time = row['Time']
            pos = (row['X'], row['Y'], row['Z'])
            positions.append((time, pos))
            
        return positions
        
    def interpolate_position(self, time: datetime) -> Tuple[float, float, float]:
        """插值计算指定时间的卫星位置
        
        Args:
            time: 时间点
            
        Returns:
            position: (x,y,z)插值计算的卫星位置
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        # 找到时间点前后的数据
        after_mask = self.data['Time'] >= time
        before_mask = self.data['Time'] < time
        
        if not (any(before_mask) and any(after_mask)):
            raise ValueError("Time out of data range")
            
        before_time = self.data[before_mask]['Time'].max()
        after_time = self.data[after_mask]['Time'].min()
        
        before_pos = self.get_satellite_position(before_time)
        after_pos = self.get_satellite_position(after_time)
        
        # 线性插值
        dt_total = (after_time - before_time).total_seconds()
        dt = (time - before_time).total_seconds()
        t = dt / dt_total
        
        x = before_pos[0] + t * (after_pos[0] - before_pos[0])
        y = before_pos[1] + t * (after_pos[1] - before_pos[1])
        z = before_pos[2] + t * (after_pos[2] - before_pos[2])
        
        return (x, y, z) 