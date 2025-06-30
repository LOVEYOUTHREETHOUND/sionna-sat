"""卫星波束模型"""

import numpy as np
from typing import List, Tuple, Dict
from ..utils.coordinates import calculate_elevation_azimuth
from ..utils.geometry import calculate_beam_gain

class Beam:
    """卫星波束类"""
    
    def __init__(self,
                 beam_id: int,
                 max_gain: float,
                 beam_width: float,
                 pointing_elevation: float = 0,
                 pointing_azimuth: float = 0):
        """初始化波束对象
        
        Args:
            beam_id: 波束ID
            max_gain: 波束中心最大增益(dB)
            beam_width: 3dB波束宽度(度)
            pointing_elevation: 指向俯仰角(度)
            pointing_azimuth: 指向方位角(度)
        """
        self.beam_id = beam_id
        self.max_gain = max_gain
        self.beam_width = beam_width
        self.pointing_elevation = pointing_elevation
        self.pointing_azimuth = pointing_azimuth
        
        # 初始化覆盖的小区和用户
        self.covered_cells = []
        self.covered_users = []
        
    def update_pointing(self,
                       elevation: float,
                       azimuth: float):
        """更新波束指向
        
        Args:
            elevation: 新的俯仰角(度)
            azimuth: 新的方位角(度)
        """
        self.pointing_elevation = elevation
        self.pointing_azimuth = azimuth
        
    def calculate_gain(self,
                      sat_pos: Tuple[float, float, float],
                      ground_pos: Tuple[float, float, float]) -> float:
        """计算波束增益
        
        Args:
            sat_pos: 卫星ECEF坐标(x,y,z)
            ground_pos: 地面点ECEF坐标(x,y,z)
            
        Returns:
            gain: 波束增益(dB)
        """
        return calculate_beam_gain(
            sat_pos,
            ground_pos,
            self.max_gain,
            self.beam_width
        )
        
    def is_point_covered(self,
                        sat_pos: Tuple[float, float, float],
                        ground_pos: Tuple[float, float, float],
                        min_gain: float = -30) -> bool:
        """判断点是否被波束覆盖
        
        Args:
            sat_pos: 卫星ECEF坐标
            ground_pos: 地面点ECEF坐标
            min_gain: 最小增益阈值(dB)
            
        Returns:
            covered: 是否被覆盖
        """
        # 计算到地面点的仰角和方位角
        elevation, azimuth = calculate_elevation_azimuth(sat_pos, ground_pos)
        
        # 计算与波束指向的角度差
        delta_elevation = abs(elevation - self.pointing_elevation)
        delta_azimuth = abs(azimuth - self.pointing_azimuth)
        if delta_azimuth > 180:
            delta_azimuth = 360 - delta_azimuth
            
        # 计算总的角度偏差
        angle_offset = np.sqrt(delta_elevation**2 + delta_azimuth**2)
        
        # 计算波束增益
        gain = self.calculate_gain(sat_pos, ground_pos)
        
        # 判断是否被覆盖
        return gain >= min_gain and angle_offset <= 2*self.beam_width
        
    def update_coverage(self,
                       sat_pos: Tuple[float, float, float],
                       cells: List[Dict],
                       min_gain: float = -30):
        """更新波束覆盖的小区和用户
        
        Args:
            sat_pos: 卫星ECEF坐标
            cells: 小区列表，每个小区是包含id、position和users的字典
            min_gain: 最小增益阈值(dB)
        """
        self.covered_cells = []
        self.covered_users = []
        
        for cell in cells:
            cell_pos = cell['position']
            if self.is_point_covered(sat_pos, cell_pos, min_gain):
                self.covered_cells.append(cell['id'])
                
                # 检查小区内的用户
                for user in cell['users']:
                    user_pos = user['position']
                    if self.is_point_covered(sat_pos, user_pos, min_gain):
                        self.covered_users.append(user['id'])
                        
    def get_coverage_stats(self) -> Dict[str, int]:
        """获取覆盖统计信息
        
        Returns:
            stats: 覆盖统计信息字典
        """
        return {
            'num_cells': len(self.covered_cells),
            'num_users': len(self.covered_users)
        }
        
class BeamGroup:
    """波束组类"""
    
    def __init__(self,
                 num_beams: int,
                 max_gain: float,
                 beam_width: float):
        """初始化波束组
        
        Args:
            num_beams: 波束数量
            max_gain: 波束中心最大增益(dB)
            beam_width: 3dB波束宽度(度)
        """
        self.num_beams = num_beams
        self.max_gain = max_gain
        self.beam_width = beam_width
        
        # 创建波束对象
        self.beams = []
        self._init_beams()
        
    def _init_beams(self):
        """初始化波束配置"""
        if self.num_beams == 7:
            # 7波束配置：1个中心波束，6个外围波束
            # 中心波束
            center_beam = Beam(0, self.max_gain, self.beam_width, 90, 0)
            self.beams.append(center_beam)
            
            # 6个外围波束，相位差60度
            for i in range(6):
                elevation = 90 - self.beam_width  # 略微倾斜
                azimuth = i * 60
                beam = Beam(i+1, self.max_gain, self.beam_width, elevation, azimuth)
                self.beams.append(beam)
                
    def update_coverage(self,
                       sat_pos: Tuple[float, float, float],
                       cells: List[Dict],
                       min_gain: float = -30):
        """更新所有波束的覆盖
        
        Args:
            sat_pos: 卫星ECEF坐标
            cells: 小区列表
            min_gain: 最小增益阈值(dB)
        """
        for beam in self.beams:
            beam.update_coverage(sat_pos, cells, min_gain)
            
    def get_beam_by_id(self, beam_id: int) -> Beam:
        """根据ID获取波束对象
        
        Args:
            beam_id: 波束ID
            
        Returns:
            beam: 波束对象
        """
        if 0 <= beam_id < len(self.beams):
            return self.beams[beam_id]
        else:
            raise ValueError(f"Invalid beam ID: {beam_id}")
            
    def get_coverage_map(self) -> Dict[int, List[int]]:
        """获取波束覆盖映射
        
        Returns:
            coverage: {beam_id: [cell_id, ...], ...}波束覆盖字典
        """
        coverage = {}
        for beam in self.beams:
            coverage[beam.beam_id] = beam.covered_cells
        return coverage
        
    def get_user_assignment(self) -> Dict[int, List[int]]:
        """获取用户分配
        
        Returns:
            assignment: {beam_id: [user_id, ...], ...}用户分配字典
        """
        assignment = {}
        for beam in self.beams:
            assignment[beam.beam_id] = beam.covered_users
        return assignment 