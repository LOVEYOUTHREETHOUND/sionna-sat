import matplotlib.pyplot as plt
import numpy as np
import json
import h3
from typing import List, Tuple, Dict
import os
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

class CoverageVisualizer:
    def __init__(self):
        self.areas = {
            'A1': {
                'name': '低纬度典型区域',
                'color': 'red',
                'marker': 'o'
            },
            'A2': {
                'name': '高纬度典型区域',
                'color': 'blue',
                'marker': 's'
            }
        }
        
        # 获取topology目录路径
        topology_dir = os.path.dirname(os.path.abspath(__file__))
        # 创建输出目录
        self.output_dir = os.path.join(topology_dir, 'results', 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_coverage_data(self) -> Dict:
        """加载覆盖分析数据"""
        coverage_data = {}
        for area_id in self.areas.keys():
            with open(f'area_{area_id}_coverage.json', 'r') as f:
                coverage_data[area_id] = json.load(f)
        return coverage_data

    def plot_coverage_overview(self, coverage_data: Dict):
        """绘制总体覆盖概览图"""
        plt.figure(figsize=(15, 10))
        
        # 绘制每个区域
        for area_id, data in coverage_data.items():
            area_info = self.areas[area_id]
            bounds = data['bounds']
            
            # 绘制区域边界
            plt.plot([bounds['min_lon'], bounds['max_lon'], bounds['max_lon'], bounds['min_lon'], bounds['min_lon']],
                    [bounds['min_lat'], bounds['min_lat'], bounds['max_lat'], bounds['max_lat'], bounds['min_lat']],
                    color=area_info['color'], linestyle='--', label=f"{area_info['name']}边界")
            
            # 绘制SC中心点
            points = data['points']
            lats, lons = zip(*points)
            plt.scatter(lons, lats, c=area_info['color'], marker=area_info['marker'],
                       label=f"{area_info['name']}SC点 ({len(points)}个)")
            
            # 绘制热点区域
            hotspot_cells = [cell for cell, meta in data['cell_metadata'].items() 
                           if meta['type'] == 'hotspot']
            
            for cell in hotspot_cells:
                boundary = h3.cell_to_boundary(cell)
                boundary_lats, boundary_lons = zip(*boundary)
                plt.fill(boundary_lons, boundary_lats, color='yellow', alpha=0.3)
        
        plt.grid(True)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.title('卫星覆盖分析总览')
        plt.legend()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_dir, 'coverage_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_density_heatmap(self, coverage_data: Dict):
        """绘制用户密度热力图"""
        for area_id, data in coverage_data.items():
            plt.figure(figsize=(12, 8))
            
            # 创建网格
            bounds = data['bounds']
            cell_metadata = data['cell_metadata']
            
            # 收集所有小区的中心点和密度值
            centers = []
            densities = []
            for cell, meta in cell_metadata.items():
                center = h3.cell_to_latlng(cell)
                centers.append(center)
                densities.append(meta['ue_density'])
            
            if centers:
                lats, lons = zip(*centers)
                
                # 创建散点图，颜色表示密度
                scatter = plt.scatter(lons, lats, c=densities, cmap='YlOrRd',
                                   s=100, alpha=0.6)
                plt.colorbar(scatter, label='用户密度 (用户/km²)')
            
            # 绘制区域边界
            plt.plot([bounds['min_lon'], bounds['max_lon'], bounds['max_lon'], bounds['min_lon'], bounds['min_lon']],
                    [bounds['min_lat'], bounds['min_lat'], bounds['max_lat'], bounds['max_lat'], bounds['min_lat']],
                    color=self.areas[area_id]['color'], linestyle='--')
            
            plt.grid(True)
            plt.xlabel('经度')
            plt.ylabel('纬度')
            plt.title(f'{self.areas[area_id]["name"]}用户密度分布')
            
            # 保存图像
            plt.savefig(os.path.join(self.output_dir, f'density_heatmap_{area_id}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_statistics(self, coverage_data: Dict):
        """绘制统计信息图表"""
        # 准备数据
        area_names = []
        sc_counts = []
        cell_counts = []
        hotspot_counts = []
        
        for area_id, data in coverage_data.items():
            area_names.append(self.areas[area_id]['name'])
            sc_counts.append(len(data['points']))
            cell_counts.append(len(data['h3_cells']))
            hotspot_counts.append(sum(1 for meta in data['cell_metadata'].values() 
                                   if meta['type'] == 'hotspot'))
        
        # 创建柱状图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(area_names))
        width = 0.25
        
        plt.bar(x - width, sc_counts, width, label='SC数量', color='skyblue')
        plt.bar(x, cell_counts, width, label='小区数量', color='lightgreen')
        plt.bar(x + width, hotspot_counts, width, label='热点数量', color='orange')
        
        plt.xlabel('区域')
        plt.ylabel('数量')
        plt.title('覆盖分析统计')
        plt.xticks(x, area_names)
        plt.legend()
        
        # 添加数值标签
        for i in x:
            plt.text(i - width, sc_counts[i], str(sc_counts[i]), ha='center', va='bottom')
            plt.text(i, cell_counts[i], str(cell_counts[i]), ha='center', va='bottom')
            plt.text(i + width, hotspot_counts[i], str(hotspot_counts[i]), ha='center', va='bottom')
        
        # 保存图像
        plt.savefig(os.path.join(self.output_dir, 'coverage_statistics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualizations(self):
        """创建所有可视化图表"""
        print("正在生成可视化结果...")
        
        # 加载数据
        coverage_data = self.load_coverage_data()
        
        # 生成各类图表
        self.plot_coverage_overview(coverage_data)
        self.plot_density_heatmap(coverage_data)
        self.plot_statistics(coverage_data)
        
        print(f"\n可视化结果已保存到 {self.output_dir} 目录")
        print("生成的图像文件：")
        print("1. coverage_overview.png - 总体覆盖概览图")
        print("2. density_heatmap_A1.png - A1区域用户密度热力图")
        print("3. density_heatmap_A2.png - A2区域用户密度热力图")
        print("4. coverage_statistics.png - 覆盖分析统计图")

def main():
    visualizer = CoverageVisualizer()
    visualizer.create_visualizations()

if __name__ == '__main__':
    main() 