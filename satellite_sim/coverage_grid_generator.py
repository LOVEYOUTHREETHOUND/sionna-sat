import numpy as np
import h3
import folium
from typing import List, Tuple, Dict
import json
import os
import matplotlib.pyplot as plt

class CoverageGridGenerator:
    def __init__(self):
        # 基础常量
        self.R_c = 22.6  # 覆盖半径(km)
        self.hotspot_ratio = 0.1  # 热点区域比例
        
        # 目标区域定义
        self.areas = {
            'A1': {
                'name': '低纬度典型区域',
                'bounds': {
                    'min_lon': 90,
                    'max_lon': 110,
                    'min_lat': -10,
                    'max_lat': 10
                },
                'color': 'red'
            },
            'A2': {
                'name': '高纬度典型区域',
                'bounds': {
                    'min_lon': 90,
                    'max_lon': 110,
                    'min_lat': 25,
                    'max_lat': 45
                },
                'color': 'blue'
            }
        }
        
        # 选择合适的H3分辨率
        self.h3_resolution = self._select_h3_resolution()

    def _select_h3_resolution(self) -> int:
        """根据覆盖半径选择合适的H3分辨率"""
        target_area = np.pi * (self.R_c ** 2)  # 圆形覆盖面积
        
        # 从高分辨率开始尝试，找到最合适的分辨率
        for res in range(15, -1, -1):
            hex_area = h3.cell_area(h3.latlng_to_cell(0, 0, res), unit='km^2')
            if hex_area >= target_area * 0.9:  # 允许10%的误差
                return res
        return 5  # 默认分辨率

    def get_area_cells(self, bounds: Dict) -> List[str]:
        """获取区域内的所有H3单元"""
        cells = set()
        
        # 生成网格点
        lat_steps = np.linspace(bounds['min_lat'], bounds['max_lat'], 100)
        lon_steps = np.linspace(bounds['min_lon'], bounds['max_lon'], 100)
        
        # 获取每个点所在的H3单元
        for lat in lat_steps:
            for lon in lon_steps:
                cell = h3.latlng_to_cell(lat, lon, self.h3_resolution)
                cells.add(cell)
        
        return list(cells)

    def create_results_directory(self):
        """创建结果目录结构"""
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        subdirs = ['json', 'plots', 'stats', 'html']
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        return results_dir

    def save_area_statistics(self, results_dir: str):
        """保存区域统计信息"""
        stats_file = os.path.join(results_dir, 'stats', 'area_statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            for area_id, area_info in self.areas.items():
                # 读取该区域的JSON数据
                with open(os.path.join(results_dir, 'json', f'area_{area_id}_coverage.json'), 'r') as json_file:
                    data = json.load(json_file)
                
                # 写入统计信息
                f.write(f"\n{area_info['name']}统计信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"覆盖半径: {self.R_c:.2f} km\n")
                f.write(f"H3分辨率: {self.h3_resolution}\n")
                f.write(f"小区总数: {len(data['cells'])}\n")
                f.write(f"热点小区数: {len(data['hotspot_cells'])}\n")
                
                # 计算小区面积
                sample_cell = data['cells'][0]
                cell_area = h3.cell_area(sample_cell, unit='km^2')
                f.write(f"单个小区面积: {cell_area:.2f} km²\n")
                
                # 写入所有小区编码
                f.write("\n普通小区编码:\n")
                normal_cells = set(data['cells']) - set(data['hotspot_cells'])
                for cell in sorted(normal_cells):
                    f.write(f"{cell}\n")
                
                f.write("\n热点小区编码:\n")
                for cell in sorted(data['hotspot_cells']):
                    f.write(f"{cell}\n")
                
                f.write("\n" + "=" * 40 + "\n")

    def plot_area_coverage(self, area_id: str, data: Dict, results_dir: str):
        """生成静态可视化图"""
        plt.figure(figsize=(15, 10))
        
        # 绘制区域边界
        bounds = data['bounds']
        plt.plot([bounds['min_lon'], bounds['max_lon'], bounds['max_lon'], bounds['min_lon'], bounds['min_lon']],
                [bounds['min_lat'], bounds['min_lat'], bounds['max_lat'], bounds['max_lat'], bounds['min_lat']],
                'k--', label='区域边界')
        
        # 绘制H3六边形
        for cell in data['cells']:
            boundary = h3.cell_to_boundary(cell)
            boundary_lats, boundary_lons = zip(*boundary)
            
            if cell in data['hotspot_cells']:
                plt.fill(boundary_lons, boundary_lats, 'yellow', alpha=0.3, label='热点小区' if cell == data['hotspot_cells'][0] else '')
            else:
                plt.fill(boundary_lons, boundary_lats, 'lightblue', alpha=0.2, label='普通小区' if cell == data['cells'][0] else '')
        
        # 添加标题和标签
        plt.title(f"{self.areas[area_id]['name']}覆盖分析\n"
                 f"(总小区数: {len(data['cells'])}, 热点小区数: {len(data['hotspot_cells'])})")
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.grid(True)
        plt.legend()
        
        # 保存图片
        plt.savefig(os.path.join(results_dir, 'plots', f'coverage_area_{area_id}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualization(self) -> None:
        """创建可视化结果"""
        # 创建结果目录
        results_dir = self.create_results_directory()
        print(f"\n结果将保存在 {os.path.abspath(results_dir)} 目录中")
        
        # 处理每个区域
        for area_id, area_info in self.areas.items():
            print(f"\n处理{area_info['name']}...")
            
            # 获取区域内的所有小区
            cells = self.get_area_cells(area_info['bounds'])
            print(f"小区总数: {len(cells)}")
            
            # 随机选择热点小区
            hotspot_count = int(len(cells) * self.hotspot_ratio)
            hotspot_cells = sorted(np.random.choice(cells, hotspot_count, replace=False))
            print(f"热点小区数: {len(hotspot_cells)}")
            
            # 准备数据
            coverage_data = {
                'area_name': area_info['name'],
                'cells': sorted(cells),
                'hotspot_cells': hotspot_cells,
                'bounds': area_info['bounds'],
                'resolution': self.h3_resolution,
                'coverage_radius_km': self.R_c
            }
            
            # 保存JSON数据
            print("保存JSON数据...")
            json_file = os.path.join(results_dir, 'json', f'area_{area_id}_coverage.json')
            with open(json_file, 'w') as f:
                json.dump(coverage_data, f, indent=2)
            
            # 生成静态图
            print("生成可视化图...")
            self.plot_area_coverage(area_id, coverage_data, results_dir)
        
        # 保存统计信息
        print("\n保存统计信息...")
        self.save_area_statistics(results_dir)
        
        # 创建交互式地图
        print("创建交互式地图...")
        self._create_folium_map(results_dir)

    def _create_folium_map(self, results_dir: str) -> None:
        """创建交互式地图"""
        # 计算地图中心
        center_lat = (self.areas['A1']['bounds']['min_lat'] + 
                     self.areas['A2']['bounds']['max_lat']) / 2
        center_lon = (self.areas['A1']['bounds']['min_lon'] + 
                     self.areas['A1']['bounds']['max_lon']) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles=None
        )

        # 添加底图
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
        ).add_to(m)

        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
        ).add_to(m)

        # 为每个区域添加图层
        for area_id, area_info in self.areas.items():
            # 读取数据
            with open(os.path.join(results_dir, 'json', f'area_{area_id}_coverage.json'), 'r') as f:
                data = json.load(f)
            
            # 创建图层组
            normal_group = folium.FeatureGroup(name=f"{area_info['name']} - 普通小区")
            hotspot_group = folium.FeatureGroup(name=f"{area_info['name']} - 热点小区")
            
            # 添加区域边界
            bounds = area_info['bounds']
            folium.Rectangle(
                bounds=[[bounds['min_lat'], bounds['min_lon']], 
                       [bounds['max_lat'], bounds['max_lon']]],
                color=area_info['color'],
                fill=True,
                weight=2,
                fill_opacity=0.1
            ).add_to(normal_group)
            
            # 添加H3六边形
            for cell in data['cells']:
                is_hotspot = cell in data['hotspot_cells']
                boundary = h3.cell_to_boundary(cell)
                
                folium.Polygon(
                    locations=boundary,
                    color='yellow' if is_hotspot else area_info['color'],
                    weight=1,
                    fill_opacity=0.3 if is_hotspot else 0.2,
                    popup=f"Area: {area_info['name']}<br>"
                          f"Type: {'热点小区' if is_hotspot else '普通小区'}<br>"
                          f"H3 Cell: {cell}"
                ).add_to(hotspot_group if is_hotspot else normal_group)
            
            normal_group.add_to(m)
            hotspot_group.add_to(m)
        
        # 添加图层控制
        folium.LayerControl(collapsed=False).add_to(m)
        
        # 保存地图
        output_file = os.path.join(results_dir, 'html', 'coverage_visualization.html')
        m.save(output_file)
        print(f"\n交互式地图已保存为 {output_file}")
        print(f"您可以在浏览器中打开此文件：file:///{os.path.abspath(output_file)}")

def main():
    generator = CoverageGridGenerator()
    generator.create_visualization()

if __name__ == '__main__':
    main() 