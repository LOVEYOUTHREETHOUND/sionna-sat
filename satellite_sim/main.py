"""主程序"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

from config.system_params import (
    SATELLITE_PARAMS,
    CELL_PARAMS,
    CHANNEL_PARAMS,
    SIM_PARAMS
)

from models.satellite import Satellite
from models.cell import CellGrid
from models.beam import BeamGroup
from models.channel import ChannelModel

from utils.coordinates import geodetic_to_ecef, ecef_to_geodetic
from utils.geometry import calculate_path_loss

# 定义目标区域
TARGET_AREAS = {
    'A1': {
        'name': '低纬度区域',
        'bounds': {
            'min_lon': 90,
            'max_lon': 110,
            'min_lat': -10,
            'max_lat': 10
        },
        'color': 'red'
    },
    'A2': {
        'name': '高纬度区域',
        'bounds': {
            'min_lon': 90,
            'max_lon': 110,
            'min_lat': 25,
            'max_lat': 45
        },
        'color': 'blue'
    }
}

def plot_topology(cell_grid: CellGrid,
                 satellite: Satellite,
                 beam_group: BeamGroup,
                 time: datetime):
    """绘制系统拓扑图
    
    Args:
        cell_grid: 小区网格对象
        satellite: 卫星对象
        beam_group: 波束组对象
        time: 当前时间
    """
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地面小区
    cell_centers = cell_grid.get_all_cell_centers()
    
    # 按区域分组绘制小区
    for area_id in TARGET_AREAS:
        area_cells = cell_grid.get_cells_by_area(area_id)
        if not area_cells:
            continue
            
        area_centers = [cell.get_center_position() for cell in area_cells]
        cell_x = [pos[0]/1000 for pos in area_centers]  # 转换为km
        cell_y = [pos[1]/1000 for pos in area_centers]
        cell_z = [pos[2]/1000 for pos in area_centers]
        
        ax.scatter(cell_x, cell_y, cell_z, 
                  c=TARGET_AREAS[area_id]['color'], 
                  marker='o', 
                  label=f"Cells ({TARGET_AREAS[area_id]['name']})")
    
    # 绘制用户
    user_positions = cell_grid.get_all_user_positions()
    user_x = [pos[0]/1000 for pos in user_positions]
    user_y = [pos[1]/1000 for pos in user_positions]
    user_z = [pos[2]/1000 for pos in user_positions]
    ax.scatter(user_x, user_y, user_z, c='green', marker='.', label='Users')
    
    # 绘制卫星
    sat_pos = satellite.position
    ax.scatter([sat_pos[0]/1000], [sat_pos[1]/1000], [sat_pos[2]/1000],
               c='red', marker='^', s=100, label='Satellite')
    
    # 绘制波束覆盖
    coverage_map = beam_group.get_coverage_map()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(coverage_map)))
    
    for beam_id, covered_cells in coverage_map.items():
        if covered_cells:
            # 获取被覆盖小区的坐标
            covered_x = [cell_centers[i][0]/1000 for i in covered_cells]
            covered_y = [cell_centers[i][1]/1000 for i in covered_cells]
            covered_z = [cell_centers[i][2]/1000 for i in covered_cells]
            
            # 绘制波束覆盖区域
            ax.scatter(covered_x, covered_y, covered_z,
                      c=[colors[beam_id]], marker='o', s=50,
                      label=f'Beam {beam_id}')
            
            # 绘制从卫星到覆盖区域的连线
            for x, y, z in zip(covered_x, covered_y, covered_z):
                ax.plot([sat_pos[0]/1000, x],
                       [sat_pos[1]/1000, y],
                       [sat_pos[2]/1000, z],
                       c=colors[beam_id], alpha=0.2)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    
    # 添加标题
    ax.set_title(f'System Topology at {time}')
    
    # 添加图例
    ax.legend()
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    plt.show()

def plot_coverage_stats(stats_history: List[Dict]):
    """绘制覆盖统计信息
    
    Args:
        stats_history: 覆盖统计历史记录
    """
    # 提取时间序列数据
    times = [stat['time'] for stat in stats_history]
    coverage_rates = [stat['coverage_rate'] for stat in stats_history]
    num_covered_cells = [stat['covered_cells'] for stat in stats_history]
    num_covered_users = [stat['covered_users'] for stat in stats_history]
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 绘制覆盖率
    ax1.plot(times, coverage_rates)
    ax1.set_ylabel('Coverage Rate (%)')
    ax1.set_title('Coverage Statistics')
    ax1.grid(True)
    
    # 绘制覆盖小区数
    ax2.plot(times, num_covered_cells)
    ax2.set_ylabel('Number of Covered Cells')
    ax2.grid(True)
    
    # 绘制覆盖用户数
    ax3.plot(times, num_covered_users)
    ax3.set_ylabel('Number of Covered Users')
    ax3.set_xlabel('Time')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_channel_stats(channel_history: List[Dict]):
    """绘制信道统计信息
    
    Args:
        channel_history: 信道统计历史记录
    """
    # 提取时间序列数据
    times = [stat['time'] for stat in channel_history]
    path_losses = [stat['path_loss'] for stat in channel_history]
    snrs = [stat['snr'] for stat in channel_history]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制路径损耗
    ax1.plot(times, path_losses)
    ax1.set_ylabel('Path Loss (dB)')
    ax1.set_title('Channel Statistics')
    ax1.grid(True)
    
    # 绘制SNR
    ax2.plot(times, snrs)
    ax2.set_ylabel('SNR (dB)')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 1. 创建地面小区网格
    print("Creating cell grid...")
    cell_grid = CellGrid(
        areas=TARGET_AREAS,
        cell_radius=CELL_PARAMS['radius'],
        users_per_cell=CELL_PARAMS['num_users']
    )
    
    print(f"总小区数: {cell_grid.get_num_cells()}")
    for area_id in TARGET_AREAS:
        area_cells = cell_grid.get_cells_by_area(area_id)
        print(f"{TARGET_AREAS[area_id]['name']}小区数: {len(area_cells)}")
    print(f"总用户数: {cell_grid.get_total_users()}")
    
    # 2. 创建卫星对象
    print("Creating satellite...")
    satellite = Satellite(
        sat_id="SAT001",
        stk_data_file="satellite_orbit.csv",  # STK导出的轨道数据
        num_beams=SATELLITE_PARAMS['num_beams'],
        tx_power=SATELLITE_PARAMS['tx_power'],
        antenna_gain=SATELLITE_PARAMS['antenna_gain'],
        frequency=SATELLITE_PARAMS['frequency']
    )
    
    # 3. 创建波束组
    print("Creating beam group...")
    beam_group = BeamGroup(
        num_beams=SATELLITE_PARAMS['num_beams'],
        max_gain=SATELLITE_PARAMS['antenna_gain'],
        beam_width=1.0  # 1度波束宽度
    )
    
    # 4. 创建信道模型
    print("Creating channel model...")
    channel_model = ChannelModel(
        frequency=SATELLITE_PARAMS['frequency'],
        bandwidth=CHANNEL_PARAMS['bandwidth'],
        noise_figure=CHANNEL_PARAMS['noise_figure'],
        temperature=CHANNEL_PARAMS['temperature'],
        rain_rate=CHANNEL_PARAMS['rain_rate']
    )
    
    # 5. 初始化统计记录
    stats_history = []
    channel_history = []
    
    # 6. 仿真主循环
    print("Starting simulation...")
    start_time = datetime.now()
    for slot in range(SIM_PARAMS['num_slots']):
        current_time = start_time + timedelta(seconds=slot*SIM_PARAMS['slot_duration'])
        
        # 更新卫星位置
        satellite.update_position(current_time)
        
        # 更新波束覆盖
        cells_info = cell_grid.get_all_cells_info()
        beam_group.update_coverage(satellite.position, cells_info)
        
        # 获取用户位置
        user_positions = cell_grid.get_all_user_positions()
        
        # 计算信道响应
        H = channel_model.calculate_channel_response(
            satellite_position=satellite.position,
            user_positions=user_positions,
            satellite_gain=SATELLITE_PARAMS['antenna_gain'],
            user_gain=10,  # 用户天线增益
            beam_width=beam_group.beam_width,
            tx_power=SATELLITE_PARAMS['tx_power']
        )
        
        # 添加多普勒效应
        H = channel_model.add_doppler_effect(
            H=H,
            satellite_velocity=satellite.velocity,
            user_positions=user_positions,
            satellite_position=satellite.position
        )
        
        # 计算SNR
        snr = channel_model.calculate_snr(H, SATELLITE_PARAMS['tx_power'])
        
        # 记录统计信息
        coverage_stats = {
            'time': current_time,
            'coverage_rate': len(set().union(*beam_group.get_coverage_map().values())) / cell_grid.get_num_cells() * 100,
            'covered_cells': sum(len(cells) for cells in beam_group.get_coverage_map().values()),
            'covered_users': sum(len(users) for users in beam_group.get_user_assignment().values())
        }
        stats_history.append(coverage_stats)
        
        channel_stats = {
            'time': current_time,
            'snr': np.mean(snr),
            'path_loss': np.mean([
                calculate_path_loss(
                    np.linalg.norm(np.array(satellite.position) - np.array(pos)),
                    SATELLITE_PARAMS['frequency'],
                    CHANNEL_PARAMS['rain_rate']
                )
                for pos in user_positions
            ])
        }
        channel_history.append(channel_stats)
        
        # 每100个时隙打印一次进度
        if slot % 100 == 0:
            print(f"Simulated {slot}/{SIM_PARAMS['num_slots']} slots")
            
        # 每500个时隙绘制一次系统拓扑图
        if slot % 500 == 0:
            plot_topology(cell_grid, satellite, beam_group, current_time)
    
    # 7. 绘制统计结果
    print("Plotting results...")
    plot_coverage_stats(stats_history)
    plot_channel_stats(channel_history)
    
    print("Simulation completed.")

if __name__ == "__main__":
    main() 