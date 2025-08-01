import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob

class SimulationResultPlotter:
    def __init__(self, results_dir='simulation_results'):
        """仿真结果绘图器
        Args:
            results_dir: 仿真结果目录
        """
        self.results_dir = results_dir
        self.output_dir = None
        
    def find_latest_results(self):
        """找到最新的仿真结果目录"""
        # 只查找目录，不包含文件
        timestamp_dirs = [
            d for d in glob.glob(os.path.join(self.results_dir, '*_*'))
            if os.path.isdir(d)
        ]
        if not timestamp_dirs:
            raise FileNotFoundError(f"在 {self.results_dir} 中未找到仿真结果目录")
        
        # 按目录名排序，取最新的
        latest_dir = max(timestamp_dirs, key=os.path.basename)
        return latest_dir
    
    def load_data(self, results_dir=None):
        """加载仿真数据
        Args:
            results_dir: 结果目录，如果为None则使用最新的
        """
        if results_dir is None:
            results_dir = self.find_latest_results()
        
        self.output_dir = os.path.join(results_dir, 'plots')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载SINR数据
        sinr_file = os.path.join(results_dir, 'sinr_user*_slot*_table.csv')
        sinr_files = glob.glob(sinr_file)
        if not sinr_files:
            raise FileNotFoundError(f"在 {results_dir} 中未找到SINR数据文件")
        
        self.sinr_data = pd.read_csv(sinr_files[0])
        print(f"加载SINR数据: {sinr_files[0]}")
        print(f"数据形状: {self.sinr_data.shape}")
        
        # 加载详细数据
        detail_file = os.path.join(results_dir, 'sinr_detail_user*_slot*_table.csv')
        detail_files = glob.glob(detail_file)
        if detail_files:
            self.detail_data = pd.read_csv(detail_files[0])
            print(f"加载详细数据: {detail_files[0]}")
            print(f"详细数据形状: {self.detail_data.shape}")
        else:
            self.detail_data = None
            print("未找到详细数据文件")
        
        # 提取基本信息
        self.num_users = len(self.sinr_data)
        self.num_slots = len(self.sinr_data.columns) - 1  # 减去'user'列
        self.slot_columns = [col for col in self.sinr_data.columns if col.startswith('slot_')]
        
        print(f"用户数量: {self.num_users}")
        print(f"时隙数量: {self.num_slots}")
        
    def plot_sinr_cdf(self, save_plot=True):
        """绘制SINR的CDF图"""
        plt.figure(figsize=(12, 8))
        
        # 收集所有SINR数据
        all_sinr = []
        for col in self.slot_columns:
            all_sinr.extend(self.sinr_data[col].values)
        
        # 过滤有效值
        valid_sinr = [x for x in all_sinr if x > -1000]  # 过滤掉-inf值
        
        if len(valid_sinr) == 0:
            print("警告: 没有有效的SINR数据")
            return
        
        # 计算统计信息
        mean_sinr = np.mean(valid_sinr)
        median_sinr = np.median(valid_sinr)
        std_sinr = np.std(valid_sinr)
        
        # 计算CDF
        sorted_sinr = np.sort(valid_sinr)
        cdf = np.arange(1, len(sorted_sinr) + 1) / len(sorted_sinr)
        
        # 绘制CDF
        plt.plot(sorted_sinr, cdf, 'b-', linewidth=2, label=f'SINR CDF (n={len(valid_sinr)})')
        
        # 添加统计线
        plt.axvline(mean_sinr, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_sinr:.2f} dB')
        plt.axvline(median_sinr, color='green', linestyle='--', alpha=0.7, 
                   label=f'Median: {median_sinr:.2f} dB')
        
        # 添加分位数线
        percentiles = [10, 25, 75, 90]
        colors = ['orange', 'purple', 'purple', 'orange']
        for p, color in zip(percentiles, colors):
            value = np.percentile(valid_sinr, p)
            plt.axvline(value, color=color, linestyle=':', alpha=0.5, 
                       label=f'{p}th percentile: {value:.2f} dB')
        
        # 设置图表属性
        plt.xlabel('SINR (dB)', fontsize=12)
        plt.ylabel('CDF', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 添加统计信息文本框
        stats_text = f'Statistics:\nMean: {mean_sinr:.2f} dB\nMedian: {median_sinr:.2f} dB\nStd: {std_sinr:.2f} dB\nMin: {min(valid_sinr):.2f} dB\nMax: {max(valid_sinr):.2f} dB'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(self.output_dir, 'sinr_cdf_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"SINR CDF图已保存: {filename}")
        
        plt.show()
        
    def plot_sinr_heatmap(self, save_plot=True):
        """绘制SINR热力图（按用户和时隙）"""
        plt.figure(figsize=(15, 10))
        
        # 准备热力图数据
        heatmap_data = self.sinr_data[self.slot_columns].values
        
        # 创建热力图
        sns.heatmap(heatmap_data, 
                   cmap='RdYlBu_r',  # 红色-黄色-蓝色，红色表示低SINR
                   center=0,  # 以0为中心
                   cbar_kws={'label': 'SINR (dB)'},
                   xticklabels=[f'Slot {i}' for i in range(self.num_slots)],
                   yticklabels=[f'User {i}' for i in range(self.num_users)])
        
        plt.title('SINR Heatmap by User and Timeslot', fontsize=14, fontweight='bold')
        plt.xlabel('Timeslot', fontsize=12)
        plt.ylabel('User ID', fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(self.output_dir, 'sinr_heatmap_by_user_slot.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"SINR热力图已保存: {filename}")
        
        plt.show()
        
    def plot_sinr_time_series(self, save_plot=True):
        """绘制SINR时间序列图"""
        plt.figure(figsize=(15, 8))
        
        # 计算每个时隙的统计信息
        slot_stats = []
        for col in self.slot_columns:
            values = self.sinr_data[col].values
            valid_values = values[values > -1000]
            if len(valid_values) > 0:
                slot_stats.append({
                    'mean': np.mean(valid_values),
                    'median': np.median(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values)
                })
            else:
                slot_stats.append({
                    'mean': np.nan, 'median': np.nan, 'std': np.nan,
                    'min': np.nan, 'max': np.nan
                })
        
        # 转换为DataFrame
        stats_df = pd.DataFrame(slot_stats)
        slots = range(self.num_slots)
        
        # 绘制时间序列
        plt.plot(slots, stats_df['mean'], 'b-o', linewidth=2, markersize=6, label='Mean SINR')
        plt.plot(slots, stats_df['median'], 'g-s', linewidth=2, markersize=6, label='Median SINR')
        
        # 添加误差带
        plt.fill_between(slots, 
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        alpha=0.3, color='blue', label='±1 Std Dev')
        
        # 添加最大最小值
        plt.plot(slots, stats_df['max'], 'r--', alpha=0.7, label='Max SINR')
        plt.plot(slots, stats_df['min'], 'r--', alpha=0.7, label='Min SINR')
        
        plt.xlabel('Timeslot', fontsize=12)
        plt.ylabel('SINR (dB)', fontsize=12)
        plt.title('SINR Time Series Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(self.output_dir, 'sinr_time_series.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"SINR时间序列图已保存: {filename}")
        
        plt.show()
        
    def plot_detailed_analysis(self, save_plot=True):
        """绘制详细分析图（如果有详细数据）"""
        if self.detail_data is None:
            print("没有详细数据，跳过详细分析图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed SINR Analysis', fontsize=16, fontweight='bold')
        
        # 1. 接收功率分布
        valid_rx_power = self.detail_data['rx_power_db'][self.detail_data['rx_power_db'] > -1000]
        axes[0, 0].hist(valid_rx_power, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Received Power (dBW)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Received Power Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 干扰功率分布
        valid_intra = self.detail_data['intra_interf_db'][self.detail_data['intra_interf_db'] > -1000]
        valid_inter = self.detail_data['inter_interf_db'][self.detail_data['inter_interf_db'] > -1000]
        
        axes[0, 1].hist(valid_intra, bins=50, alpha=0.7, color='red', label='Intra-cell', edgecolor='black')
        axes[0, 1].hist(valid_inter, bins=50, alpha=0.7, color='orange', label='Inter-cell', edgecolor='black')
        axes[0, 1].set_xlabel('Interference Power (dBW)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Interference Power Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. SINR vs 接收功率散点图
        valid_data = self.detail_data[self.detail_data['sinr_db'] > -1000]
        axes[1, 0].scatter(valid_data['rx_power_db'], valid_data['sinr_db'], alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Received Power (dBW)')
        axes[1, 0].set_ylabel('SINR (dB)')
        axes[1, 0].set_title('SINR vs Received Power')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 总干扰 vs SINR散点图
        total_interf = valid_data['intra_interf_db'] + valid_data['inter_interf_db']
        axes[1, 1].scatter(total_interf, valid_data['sinr_db'], alpha=0.6, s=20, color='red')
        axes[1, 1].set_xlabel('Total Interference Power (dBW)')
        axes[1, 1].set_ylabel('SINR (dB)')
        axes[1, 1].set_title('SINR vs Total Interference')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(self.output_dir, 'detailed_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"详细分析图已保存: {filename}")
        
        plt.show()
        
    def generate_all_plots(self, results_dir=None):
        """生成所有图表"""
        print("开始加载仿真数据...")
        self.load_data(results_dir)
        
        print("\n生成SINR CDF图...")
        self.plot_sinr_cdf()
        
        print("\n生成SINR热力图...")
        self.plot_sinr_heatmap()
        
        print("\n生成SINR时间序列图...")
        self.plot_sinr_time_series()
        
        print("\n生成详细分析图...")
        self.plot_detailed_analysis()
        
        print("\n生成地面拓扑热力图...")
        self.plot_ground_topology_heatmap()
        
        print(f"\n所有图表已保存到: {self.output_dir}")
        
        # 生成汇总报告
        self.generate_summary_report()
        
    def generate_summary_report(self):
        """生成汇总报告"""
        report_file = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("卫星通信系统SINR仿真分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"仿真参数:\n")
            f.write(f"- 用户数量: {self.num_users}\n")
            f.write(f"- 时隙数量: {self.num_slots}\n")
            f.write(f"- 总数据点数: {self.num_users * self.num_slots}\n\n")
            
            # 计算整体统计信息
            all_sinr = []
            for col in self.slot_columns:
                all_sinr.extend(self.sinr_data[col].values)
            valid_sinr = [x for x in all_sinr if x > -1000]
            
            f.write(f"SINR统计信息:\n")
            f.write(f"- 有效数据点数: {len(valid_sinr)}\n")
            f.write(f"- 平均SINR: {np.mean(valid_sinr):.2f} dB\n")
            f.write(f"- 中位数SINR: {np.median(valid_sinr):.2f} dB\n")
            f.write(f"- 标准差: {np.std(valid_sinr):.2f} dB\n")
            f.write(f"- 最小值: {min(valid_sinr):.2f} dB\n")
            f.write(f"- 最大值: {max(valid_sinr):.2f} dB\n")
            f.write(f"- 10%分位数: {np.percentile(valid_sinr, 10):.2f} dB\n")
            f.write(f"- 90%分位数: {np.percentile(valid_sinr, 90):.2f} dB\n\n")
            
            if self.detail_data is not None:
                f.write(f"详细分析:\n")
                valid_rx = self.detail_data['rx_power_db'][self.detail_data['rx_power_db'] > -1000]
                f.write(f"- 平均接收功率: {np.mean(valid_rx):.2f} dBW\n")
                f.write(f"- 平均星内干扰: {np.mean(valid_rx):.2f} dBW\n")
                f.write(f"- 平均星间干扰: {np.mean(valid_rx):.2f} dBW\n")
            
            f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"分析报告已保存: {report_file}")

    def calculate_sc_average_sinr(self):
        """计算每个小区的平均SINR
        Returns:
            sc_avg_sinr: 字典，键为小区索引，值为平均SINR
        """
        if self.sinr_data is None:
            print("SINR数据未加载，无法计算小区平均SINR。")
            return {}

        # 加载拓扑数据中的用户-小区映射关系
        import json
        import os
        
        # 拓扑数据文件路径
        topology_dir = os.path.join('data', 'topology_result', 'json')
        area_files = {
            'A1': os.path.join(topology_dir, 'area_A1_coverage.json'),
            'A2': os.path.join(topology_dir, 'area_A2_coverage.json')
        }
        
        # 构建用户-小区映射字典
        user_sc_mapping = {}
        
        for area_id, file_path in area_files.items():
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        area_data = json.load(f)
                    
                    # 从拓扑数据中提取用户-小区映射
                    for cell in area_data.get('cells', []):
                        cell_id = cell.get('cell_id', '')
                        for user in cell.get('users', []):
                            user_id = user.get('user_id', '')
                            # 提取用户索引（从USER_XXX格式中提取数字）
                            if user_id.startswith('USER_'):
                                try:
                                    user_idx = int(user_id.split('_')[1])
                                    user_sc_mapping[user_idx] = cell_id
                                except (ValueError, IndexError):
                                    continue
                    
                    print(f"从 {area_id} 区域加载了 {len([u for u in user_sc_mapping.keys() if u < 1000])} 个用户映射")
                else:
                    print(f"警告: 拓扑数据文件不存在: {file_path}")
            except Exception as e:
                print(f"加载拓扑数据时出错 {file_path}: {e}")
        
        if not user_sc_mapping:
            print("无法加载用户-小区映射关系，使用简化方法")
            # 回退到简化方法
            users_per_sc = 10
            user_sc_mapping = {user_idx: f"SC_{user_idx // users_per_sc}" 
                              for user_idx in range(self.num_users)}
        
        # 计算每个用户在所有时隙的平均SINR
        mean_sinr_per_user = self.sinr_data.mean(axis=0)

        # 按小区分组计算平均SINR
        sc_sinr_sum = {}
        sc_user_count = {}

        for user_idx, mean_sinr in mean_sinr_per_user.items():
            if user_idx in user_sc_mapping:
                sc_id = user_sc_mapping[user_idx]
                if sc_id not in sc_sinr_sum:
                    sc_sinr_sum[sc_id] = 0
                    sc_user_count[sc_id] = 0
                sc_sinr_sum[sc_id] += mean_sinr
                sc_user_count[sc_id] += 1

        # 计算每个小区的平均SINR
        sc_avg_sinr = {}
        for sc_id in sc_sinr_sum:
            sc_avg_sinr[sc_id] = sc_sinr_sum[sc_id] / sc_user_count[sc_id]

        print(f"计算了 {len(sc_avg_sinr)} 个小区的平均SINR")
        return sc_avg_sinr

    def plot_ground_topology_heatmap(self, save_plot=True):
        """根据地面拓扑生成热力图
        使用H3六边形网格和小区平均SINR数据
        """
        # 计算小区平均SINR
        sc_avg_sinr = self.calculate_sc_average_sinr()
        
        if not sc_avg_sinr:
            print("无法获取小区平均SINR数据，跳过地面拓扑热力图生成。")
            return

        # 导入H3库
        try:
            import h3
        except ImportError:
            print("需要安装h3库: pip install h3")
            return

        # 定义目标区域（与topology/coverage_grid_generator.py中的定义一致）
        areas = {
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

        # 为每个区域生成热力图
        for area_id, area_info in areas.items():
            self._plot_area_heatmap(area_id, area_info, sc_avg_sinr, save_plot)

    def _plot_area_heatmap(self, area_id, area_info, sc_avg_sinr, save_plot=True):
        """为单个区域生成热力图"""
        import json
        import os
        
        # 加载拓扑数据
        topology_file = os.path.join('data', 'topology_result', 'json', f'area_{area_id}_coverage.json')
        
        if not os.path.exists(topology_file):
            print(f"拓扑数据文件不存在: {topology_file}")
            return
        
        try:
            with open(topology_file, 'r', encoding='utf-8') as f:
                area_data = json.load(f)
        except Exception as e:
            print(f"加载拓扑数据时出错: {e}")
            return
        
        # 创建热力图数据
        heatmap_data = []
        
        for cell in area_data.get('cells', []):
            cell_id = cell.get('cell_id', '')
            center_coords = cell.get('center_coordinates', {}).get('geodetic', {})
            
            if center_coords and cell_id in sc_avg_sinr:
                lat = center_coords.get('latitude')
                lon = center_coords.get('longitude')
                sinr_value = sc_avg_sinr[cell_id]
                
                if lat is not None and lon is not None:
                    heatmap_data.append({
                        'lat': lat,
                        'lon': lon,
                        'sinr': sinr_value,
                        'cell_id': cell_id
                    })
        
        # 创建热力图
        plt.figure(figsize=(12, 8))
        
        if heatmap_data:
            # 提取坐标和SINR值
            lats = [d['lat'] for d in heatmap_data]
            lons = [d['lon'] for d in heatmap_data]
            sinrs = [d['sinr'] for d in heatmap_data]
            
            # 创建散点图，颜色表示SINR值
            scatter = plt.scatter(lons, lats, c=sinrs, cmap='Reds', s=100, alpha=0.8)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('Average SINR (dB)', rotation=270, labelpad=15)
            
            # 设置标题和标签
            area_name = area_data.get('area_name', f'Area {area_id}')
            plt.title(f'Ground Topology SINR Heatmap - {area_id} ({area_name})')
            plt.xlabel('Longitude (°)')
            plt.ylabel('Latitude (°)')
            
            # 设置坐标轴范围
            bounds = area_info['bounds']
            plt.xlim(bounds['min_lon'], bounds['max_lon'])
            plt.ylim(bounds['min_lat'], bounds['max_lat'])
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 添加小区标签（可选，如果小区数量不多）
            if len(heatmap_data) <= 20:  # 只在小数量时添加标签
                for data in heatmap_data:
                    plt.annotate(data['cell_id'], 
                                (data['lon'], data['lat']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7)
            
            if save_plot:
                plot_path = os.path.join(self.output_dir, f'ground_topology_heatmap_{area_id}.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                print(f"地面拓扑热力图 {area_id} 已保存到：{plot_path}")
                print(f"显示了 {len(heatmap_data)} 个小区的位置和SINR值")
            
            plt.close()
        else:
            print(f"区域 {area_id} 没有有效的SINR数据")
            plt.close()

def main():
    """主函数"""
    plotter = SimulationResultPlotter()
    
    try:
        plotter.generate_all_plots()
        print("\n所有图表生成完成！")
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 