"""地面小区模型"""

import numpy as np
from typing import List, Tuple, Dict
from ..utils.coordinates import geodetic_to_ecef
from ..utils.geometry import create_hexagon_grid

class Cell:
    """地面小区类"""
    
    def __init__(self,
                 cell_id: int,
                 center_lat: float,
                 center_lon: float,
                 radius: float,
                 num_users: int,
                 area_id: str = None):
        """初始化地面小区
        
        Args:
            cell_id: 小区ID
            center_lat: 中心纬度(度)
            center_lon: 中心经度(度)
            radius: 小区半径(米)
            num_users: 用户数量
            area_id: 所属区域ID
        """
        self.cell_id = cell_id
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.num_users = num_users
        self.area_id = area_id
        
        # 计算小区中心ECEF坐标
        self.center_ecef = geodetic_to_ecef(center_lat, center_lon, 0)
        
        # 生成用户位置
        self._generate_users()
        
    def _generate_users(self):
        """生成用户随机位置"""
        # 在六边形区域内随机生成用户
        self.user_positions = []
        
        for _ in range(self.num_users):
            # 随机生成极坐标
            r = self.radius * np.sqrt(np.random.random())  # 均匀分布
            theta = np.random.random() * 2 * np.pi
            
            # 转换为局部直角坐标
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # 转换为经纬度偏移
            lat_scale = 111320  # 1度纬度对应的米数
            lon_scale = 111320 * np.cos(np.radians(self.center_lat))  # 1度经度对应的米数
            
            dlat = y / lat_scale
            dlon = x / lon_scale
            
            # 计算用户经纬度
            user_lat = self.center_lat + dlat
            user_lon = self.center_lon + dlon
            
            # 转换为ECEF坐标
            user_pos = geodetic_to_ecef(user_lat, user_lon, 0)
            self.user_positions.append(user_pos)
            
    def get_user_positions(self) -> List[Tuple[float, float, float]]:
        """获取用户ECEF坐标列表
        
        Returns:
            positions: [(x,y,z), ...]用户坐标列表
        """
        return self.user_positions
        
    def get_center_position(self) -> Tuple[float, float, float]:
        """获取小区中心ECEF坐标
        
        Returns:
            position: (x,y,z)小区中心坐标
        """
        return self.center_ecef
        
class CellGrid:
    """小区网格类"""
    
    def __init__(self,
                 areas: Dict = None,
                 center_lat: float = None,
                 center_lon: float = None,
                 cell_radius: float = 22600,  # 默认22.6km，单位米
                 num_rings: int = 2,
                 users_per_cell: int = 10):
        """初始化小区网格
        
        Args:
            areas: 区域字典，格式为 {'A1': {'name': '区域名', 'bounds': {'min_lon': x, 'max_lon': y, 'min_lat': z, 'max_lat': w}, 'color': 'color'}}
            center_lat: 中心纬度(度)，当使用单一区域时
            center_lon: 中心经度(度)，当使用单一区域时
            cell_radius: 小区半径(米)
            num_rings: 环数，当使用单一区域时
            users_per_cell: 每个小区的用户数
        """
        self.cell_radius = cell_radius
        self.users_per_cell = users_per_cell
        self.areas = areas
        
        # 创建小区对象
        self.cells = []
        
        if areas:
            # 使用多区域模式
            self._create_cells_in_areas()
        else:
            # 使用单一区域模式
            self.center_lat = center_lat
            self.center_lon = center_lon
            self.num_rings = num_rings
            
            # 生成六边形网格
            self.cell_centers = create_hexagon_grid(
                center_lat,
                center_lon,
                cell_radius,
                num_rings
            )
            
            # 创建小区对象
            for i, (lat, lon) in enumerate(self.cell_centers):
                cell = Cell(i, lat, lon, cell_radius, users_per_cell)
                self.cells.append(cell)
    
    def _create_cells_in_areas(self):
        """在多个区域中创建小区"""
        cell_id = 0
        
        for area_id, area_info in self.areas.items():
            bounds = area_info['bounds']
            
            # 计算区域中心
            center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
            center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
            
            # 计算区域尺寸
            lat_span = bounds['max_lat'] - bounds['min_lat']
            lon_span = bounds['max_lon'] - bounds['min_lon']
            
            # 估算需要的环数以覆盖整个区域
            # 假设1度纬度约111km
            lat_radius_degrees = self.cell_radius / 111000  # 转换为度
            
            # 计算需要多少个小区才能覆盖整个区域
            num_cells_lat = int(np.ceil(lat_span / (lat_radius_degrees * 2)))
            num_cells_lon = int(np.ceil(lon_span / (lat_radius_degrees * 2)))
            
            # 取较大值作为环数估计
            estimated_rings = max(num_cells_lat, num_cells_lon) // 2 + 1
            
            print(f"区域 {area_id} 估计环数: {estimated_rings}")
            
            # 生成六边形网格
            cell_centers = create_hexagon_grid(
                center_lat,
                center_lon,
                self.cell_radius,
                estimated_rings
            )
            
            # 筛选在区域内的小区
            for lat, lon in cell_centers:
                if (bounds['min_lat'] <= lat <= bounds['max_lat'] and
                    bounds['min_lon'] <= lon <= bounds['max_lon']):
                    cell = Cell(cell_id, lat, lon, self.cell_radius, self.users_per_cell, area_id)
                    self.cells.append(cell)
                    cell_id += 1
            
            print(f"区域 {area_id} 创建了 {len(self.cells) - (cell_id - len(self.cells))} 个小区")
            
    def get_all_cell_centers(self) -> List[Tuple[float, float, float]]:
        """获取所有小区中心ECEF坐标
        
        Returns:
            centers: [(x,y,z), ...]小区中心坐标列表
        """
        return [cell.get_center_position() for cell in self.cells]
        
    def get_all_user_positions(self) -> List[Tuple[float, float, float]]:
        """获取所有用户ECEF坐标
        
        Returns:
            positions: [(x,y,z), ...]用户坐标列表
        """
        positions = []
        for cell in self.cells:
            positions.extend(cell.get_user_positions())
        return positions
        
    def get_cell_by_id(self, cell_id: int) -> Cell:
        """根据ID获取小区对象
        
        Args:
            cell_id: 小区ID
            
        Returns:
            cell: 小区对象
        """
        if 0 <= cell_id < len(self.cells):
            return self.cells[cell_id]
        else:
            raise ValueError(f"Invalid cell ID: {cell_id}")
            
    def get_num_cells(self) -> int:
        """获取小区总数
        
        Returns:
            num_cells: 小区数量
        """
        return len(self.cells)
        
    def get_total_users(self) -> int:
        """获取用户总数
        
        Returns:
            num_users: 用户数量
        """
        return len(self.cells) * self.users_per_cell
        
    def get_cells_by_area(self, area_id: str) -> List[Cell]:
        """获取指定区域的小区
        
        Args:
            area_id: 区域ID
            
        Returns:
            cells: 小区列表
        """
        return [cell for cell in self.cells if cell.area_id == area_id]
        
    def get_all_cells_info(self) -> List[Dict]:
        """获取所有小区的信息
        
        Returns:
            cells_info: 小区信息列表
        """
        return [
            {
                'id': cell.cell_id,
                'position': cell.get_center_position(),
                'area_id': cell.area_id,
                'users': [
                    {'id': j, 'position': pos}
                    for j, pos in enumerate(cell.get_user_positions())
                ]
            }
            for cell in self.cells
        ] 