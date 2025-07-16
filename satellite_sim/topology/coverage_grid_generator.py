import numpy as np
import h3
import folium
from typing import List, Tuple, Dict
import json
import os
import matplotlib.pyplot as plt
from pymap3d import geodetic2ecef
from matplotlib.font_manager import FontProperties  # 添加字体支持

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

        # 设置中文字体
        self.font = FontProperties(family='SimHei')  # 使用黑体

    def _select_h3_resolution(self) -> int:
        """选择H3分辨率
        
        H3分辨率4的六边形特性：
        - 边长约为19.6km
        - 外接圆半径约为22.6km
        - 面积约为1170km²
        
        这与我们需要的覆盖半径(22.6km)完全匹配
        """
        return 4  # 直接使用分辨率4，因为其外接圆半径刚好为22.6km

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
        # 获取topology目录路径
        topology_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(topology_dir, 'results')
        
        # 创建主目录和子目录
        os.makedirs(results_dir, exist_ok=True)
        
        subdirs = ['json', 'plots', 'stats', 'html']
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        return results_dir

    def save_area_statistics(self, results_dir: str):
        """保存区域统计信息"""
        stats_file = os.path.join(results_dir, 'stats', 'area_statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:  # 指定UTF-8编码
            for area_id, area_info in self.areas.items():
                # 读取该区域的JSON数据
                json_path = os.path.join(results_dir, 'json', f'area_{area_id}_coverage.json')
                with open(json_path, 'r', encoding='utf-8') as json_file:  # 指定UTF-8编码
                    data = json.load(json_file)
                
                # 写入统计信息
                f.write(f"\nArea {area_info['name']} Statistics:\n")  # 使用英文
                f.write("-" * 40 + "\n")
                f.write(f"Coverage Radius: {self.R_c:.2f} km\n")
                f.write(f"H3 Resolution: {self.h3_resolution}\n")
                f.write(f"Total Cells: {len(data['cells'])}\n")
                
                # 计算小区面积
                sample_cell = data['cells'][0]['h3_index']
                cell_area = h3.cell_area(sample_cell, unit='km^2')
                f.write(f"Single Cell Area: {cell_area:.2f} km²\n")
                
                # 写入所有小区编码
                f.write("\nCell IDs:\n")
                for cell_info in data['cells']:
                    f.write(f"{cell_info['cell_id']}\n")
                
                f.write("\n" + "=" * 40 + "\n")

    def plot_area_coverage(self, area_id: str, data: Dict, results_dir: str):
        """生成静态可视化图"""
        plt.figure(figsize=(15, 10))
        
        # 绘制区域边界
        bounds = data['bounds']
        plt.plot([bounds['min_lon'], bounds['max_lon'], bounds['max_lon'], bounds['min_lon'], bounds['min_lon']],
                [bounds['min_lat'], bounds['min_lat'], bounds['max_lat'], bounds['max_lat'], bounds['min_lat']],
                'k--', linewidth=2, label='Area Boundary')  # 加粗区域边界
        
        # 绘制H3六边形和标注中心点
        for cell_info in data['cells']:
            boundary = h3.cell_to_boundary(cell_info['h3_index'])
            boundary_lats, boundary_lons = zip(*boundary)
            # 先填充小区
            plt.fill(boundary_lons, boundary_lats, 'lightblue', alpha=0.1)
            # 再绘制边界线
            plt.plot(boundary_lons + (boundary_lons[0],), 
                    boundary_lats + (boundary_lats[0],), 
                    'b-', linewidth=0.8, alpha=0.6)  # 添加清晰的蓝色边界线
            
            # 标注中心点
            center = cell_info['center_coordinates']['geodetic']
            plt.plot(center['longitude'], center['latitude'], 'r.', markersize=3)  # 略微增大中心点
            
            # 为部分小区添加ID标注
            if np.random.random() < 0.1:
                plt.text(center['longitude'], center['latitude'], 
                        cell_info['cell_id'], 
                        fontsize=8, ha='center', va='center')
        
        # 添加标题和标签
        plt.title(f"Coverage Analysis - {self.areas[area_id]['name']}\n"
                 f"(Total Cells: {len(data['cells'])})", fontproperties=self.font)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)  # 降低网格线的显示强度
        plt.legend()
        
        # 保存图片，增加DPI以提高清晰度
        plt.savefig(os.path.join(results_dir, 'plots', f'coverage_area_{area_id}.png'), 
                   dpi=400, bbox_inches='tight')
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
            
            # 生成小区ID和中心坐标
            cell_data = []
            for idx, cell in enumerate(sorted(cells)):
                cell_id = f"{area_id}_{idx:04d}"  # 生成小区ID
                lat, lon = h3.cell_to_latlng(cell)  # 获取中心坐标 [lat, lon]
                
                # 转换为ECEF坐标 (假设高度为0米)
                x, y, z = geodetic2ecef(lat, lon, 0)
                
                cell_data.append({
                    'cell_id': cell_id,
                    'h3_index': cell,
                    'center_coordinates': {
                        'geodetic': {  # 经纬度坐标
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': 0  # 假设在地表
                        },
                        'ecef': {  # ECEF坐标
                            'x': float(x),  # 转换为Python float类型
                            'y': float(y),
                            'z': float(z)
                        }
                    }
                })
            
            # 准备数据
            coverage_data = {
                'area_name': area_info['name'],
                'area_id': area_id,
                'bounds': area_info['bounds'],
                'resolution': self.h3_resolution,
                'coverage_radius_km': self.R_c,
                'cells': cell_data
            }
            
            # 保存JSON数据
            print("保存JSON数据...")
            json_file = os.path.join(results_dir, 'json', f'area_{area_id}_coverage.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(coverage_data, f, indent=2, ensure_ascii=False)
            
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
            with open(os.path.join(results_dir, 'json', f'area_{area_id}_coverage.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建图层组
            cell_group = folium.FeatureGroup(name=f"{area_info['name']} - 小区")
            center_group = folium.FeatureGroup(name=f"{area_info['name']} - 中心点")
            
            # 添加区域边界
            bounds = area_info['bounds']
            folium.Rectangle(
                bounds=[[bounds['min_lat'], bounds['min_lon']], 
                       [bounds['max_lat'], bounds['max_lon']]],
                color=area_info['color'],
                fill=True,
                weight=2,
                fill_opacity=0.1
            ).add_to(cell_group)
            
            # 添加六边形小区和中心点
            for cell in data['cells']:
                # 获取中心坐标
                center = cell['center_coordinates']['geodetic']
                center_coords = [center['latitude'], center['longitude']]
                
                # 获取六边形边界
                boundary = h3.cell_to_boundary(cell['h3_index'])
                
                # 创建悬停信息
                popup_html = f"""
                <div style='font-family: Arial, sans-serif;'>
                    <h4>{area_info['name']}</h4>
                    <b>小区ID:</b> {cell['cell_id']}<br>
                    <b>H3索引:</b> {cell['h3_index']}<br>
                    <b>中心位置:</b><br>
                    &nbsp;&nbsp;纬度: {center['latitude']:.6f}°<br>
                    &nbsp;&nbsp;经度: {center['longitude']:.6f}°<br>
                    <b>ECEF坐标:</b><br>
                    &nbsp;&nbsp;X: {cell['center_coordinates']['ecef']['x']:.2f} km<br>
                    &nbsp;&nbsp;Y: {cell['center_coordinates']['ecef']['y']:.2f} km<br>
                    &nbsp;&nbsp;Z: {cell['center_coordinates']['ecef']['z']:.2f} km
                </div>
                """
                
                # 添加六边形
                folium.Polygon(
                    locations=boundary,
                    color=area_info['color'],
                    weight=1,
                    fill_opacity=0.2,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(cell_group)
                
                # 添加中心点标记
                folium.CircleMarker(
                    location=center_coords,
                    radius=3,
                    color='red',
                    fill=True,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"小区ID: {cell['cell_id']}"
                ).add_to(center_group)
            
            cell_group.add_to(m)
            center_group.add_to(m)
        
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