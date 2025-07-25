import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SatelliteAnalyzer:
    def __init__(self, sat_file_path):
        """初始化卫星分析器
        Args:
            sat_file_path: 卫星位置数据文件路径
        """
        self.sat_file_path = sat_file_path
        self.num_sats_per_plane = 30  # 每个轨道平面30颗卫星
        self.num_planes = 60          # 60个轨道平面
        self.total_sats = 1800        # 总共1800颗卫星
        self.positions = self._load_satellite_positions()
        
    def _load_satellite_positions(self):
        """加载所有卫星的位置数据
        Returns:
            字典，键为卫星索引，值为DataFrame包含该卫星的位置数据
        """
        all_positions = {}
        current_sat = -1
        current_data = []
        
        with open(self.sat_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('Time'):
                if current_data:
                    df = pd.DataFrame(current_data, columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
                    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float) * 1000  # km to m
                    all_positions[current_sat] = df
                current_sat += 1
                current_data = []
                continue
                
            if line.startswith('--'):
                continue
                
            try:
                parts = line.split()
                if len(parts) >= 7:
                    time_str = ' '.join(parts[:4])
                    x, y, z = map(float, parts[4:7])
                    vx, vy, vz = map(float, parts[7:10])
                    current_data.append([time_str, x, y, z, vx, vy, vz])
            except:
                continue
                
        # 保存最后一颗卫星的数据
        if current_data:
            df = pd.DataFrame(current_data, columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float) * 1000  # km to m
            all_positions[current_sat] = df
            
        return all_positions
    
    def get_orbital_plane(self, sat_idx):
        """获取卫星所在的轨道平面编号
        Args:
            sat_idx: 卫星索引
        Returns:
            轨道平面编号
        """
        return sat_idx // self.num_sats_per_plane
    
    def get_nearest_satellites(self, target_sat_idx, time_idx=0):
        """获取目标卫星在指定时刻最近的卫星
        Args:
            target_sat_idx: 目标卫星索引
            time_idx: 时间索引（默认为第一个时刻）
        Returns:
            tuple: (同轨道最近4颗卫星索引, 不同轨道最近6颗卫星索引)
        """
        target_plane = self.get_orbital_plane(target_sat_idx)
        target_pos = np.array([
            self.positions[target_sat_idx].iloc[time_idx]['x'],
            self.positions[target_sat_idx].iloc[time_idx]['y'],
            self.positions[target_sat_idx].iloc[time_idx]['z']
        ])
        
        # 计算所有卫星到目标卫星的距离
        distances = {}
        for sat_idx in range(self.total_sats):
            if sat_idx == target_sat_idx:
                continue
                
            sat_pos = np.array([
                self.positions[sat_idx].iloc[time_idx]['x'],
                self.positions[sat_idx].iloc[time_idx]['y'],
                self.positions[sat_idx].iloc[time_idx]['z']
            ])
            dist = np.linalg.norm(sat_pos - target_pos)
            distances[sat_idx] = dist
        
        # 分别获取同轨道和不同轨道的最近卫星
        same_plane = []
        diff_plane = []
        
        # 按距离排序
        sorted_sats = sorted(distances.items(), key=lambda x: x[1])
        
        # 选择最近的卫星
        for sat_idx, dist in sorted_sats:
            plane = self.get_orbital_plane(sat_idx)
            if plane == target_plane and len(same_plane) < 4:
                same_plane.append(sat_idx)
            elif plane != target_plane and len(diff_plane) < 6:
                diff_plane.append(sat_idx)
                
            if len(same_plane) == 4 and len(diff_plane) == 6:
                break
                
        return same_plane, diff_plane
    
    def get_positions_for_duration(self, satellite_indices, duration_minutes=10):
        """获取指定卫星在给定时间段内的位置数据
        Args:
            satellite_indices: 卫星索引列表
            duration_minutes: 持续时间（分钟）
        Returns:
            DataFrame: 包含所有指定卫星在时间段内的位置数据
        """
        # 假设数据采样间隔为1秒
        num_samples = duration_minutes * 60
        
        all_data = []
        for sat_idx in satellite_indices:
            sat_data = self.positions[sat_idx].iloc[:num_samples]
            sat_data['satellite'] = f'SAT-{sat_idx}'
            sat_data['orbital_plane'] = self.get_orbital_plane(sat_idx)
            all_data.append(sat_data)
            
        return pd.concat(all_data, ignore_index=True)

    def plot_satellites_3d(self, target_sat_idx, same_plane_sats, diff_plane_sats, time_idx=0):
        """Plot satellite positions in ECEF coordinates
        Args:
            target_sat_idx: Target satellite index
            same_plane_sats: List of satellites in the same orbital plane
            diff_plane_sats: List of satellites in different orbital planes
            time_idx: Time index
        """
        # Create 3D figure
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Earth
        r = 6371000  # Earth radius (m)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # Get all satellite positions
        all_sats = [target_sat_idx] + same_plane_sats + diff_plane_sats
        
        # Plot satellite trajectories (first 100 time points)
        num_points = 100
        for sat_idx in all_sats:
            pos_data = self.positions[sat_idx].iloc[:num_points]
            x = pos_data['x'].values
            y = pos_data['y'].values
            z = pos_data['z'].values
            
            if sat_idx == target_sat_idx:
                # Target satellite trajectory (red)
                ax.plot(x, y, z, 'r-', alpha=0.3, linewidth=1)
            elif sat_idx in same_plane_sats:
                # Same-plane satellite trajectories (blue)
                ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=1)
            else:
                # Different-plane satellite trajectories (green)
                ax.plot(x, y, z, 'g-', alpha=0.3, linewidth=1)
        
        # Plot current satellite positions
        # Target satellite (red star)
        target_pos = self.positions[target_sat_idx].iloc[time_idx]
        ax.scatter(target_pos['x'], target_pos['y'], target_pos['z'], 
                  c='red', marker='*', s=200, label='Target Satellite')
        
        # Same-plane satellites (blue dots)
        for sat_idx in same_plane_sats:
            sat_pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'], 
                      c='blue', marker='o', s=100)
        
        # Different-plane satellites (green triangles)
        for sat_idx in diff_plane_sats:
            sat_pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'], 
                      c='green', marker='^', s=100)
        
        # Set plot properties
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Satellite Distribution (ECEF Coordinates)')
        
        # Add legend
        ax.scatter([], [], c='red', marker='*', s=200, label='Target Satellite')
        ax.scatter([], [], c='blue', marker='o', s=100, label='Same-Plane Satellites')
        ax.scatter([], [], c='green', marker='^', s=100, label='Different-Plane Satellites')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Save figure
        plt.savefig('satellite_distribution.png', dpi=300, bbox_inches='tight')
        print("\nSatellite distribution plot saved as: satellite_distribution.png")
        
        # Show figure
        plt.show()

    def plot_complete_trajectories(self, target_sat_idx, same_plane_sats, diff_plane_sats):
        """Plot complete trajectories for all satellites
        Args:
            target_sat_idx: Target satellite index
            same_plane_sats: List of satellites in the same orbital plane
            diff_plane_sats: List of satellites in different orbital planes
        """
        # Create 3D figure
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Earth
        r = 6371000  # Earth radius (m)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        earth = ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # Plot complete trajectories for all satellites
        all_sats = [target_sat_idx] + same_plane_sats + diff_plane_sats
        
        for sat_idx in all_sats:
            pos_data = self.positions[sat_idx]
            x = pos_data['x'].values
            y = pos_data['y'].values
            z = pos_data['z'].values
            
            if sat_idx == target_sat_idx:
                # Target satellite trajectory (red)
                ax.plot(x, y, z, 'r-', linewidth=2, label='Target Satellite')
                # Add start and end markers
                ax.scatter(x[0], y[0], z[0], c='red', marker='o', s=100, label='Start Position')
                ax.scatter(x[-1], y[-1], z[-1], c='red', marker='s', s=100, label='End Position')
            
            elif sat_idx in same_plane_sats:
                # Same-plane satellite trajectories (blue)
                ax.plot(x, y, z, 'b-', linewidth=1, alpha=0.7)
                ax.scatter(x[0], y[0], z[0], c='blue', marker='o', s=50)
                ax.scatter(x[-1], y[-1], z[-1], c='blue', marker='s', s=50)
            
            else:
                # Different-plane satellite trajectories (green)
                ax.plot(x, y, z, 'g-', linewidth=1, alpha=0.7)
                ax.scatter(x[0], y[0], z[0], c='green', marker='o', s=50)
                ax.scatter(x[-1], y[-1], z[-1], c='green', marker='s', s=50)
        
        # Set plot properties
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Complete Satellite Trajectories (ECEF Coordinates)')
        
        # Add legend
        ax.plot([], [], 'r-', linewidth=2, label='Target Satellite')
        ax.plot([], [], 'b-', linewidth=1, label='Same-Plane Satellites')
        ax.plot([], [], 'g-', linewidth=1, label='Different-Plane Satellites')
        ax.scatter([], [], c='k', marker='o', s=50, label='Start Position')
        ax.scatter([], [], c='k', marker='s', s=50, label='End Position')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Save figure
        plt.savefig('satellite_complete_trajectories.png', dpi=300, bbox_inches='tight')
        print("\nComplete trajectories plot saved as: satellite_complete_trajectories.png")
        
        # Show figure
        plt.show()

def main():
    # Configure file path
    sat_file = 'data\satellite_result\Satellite1_Fixed_Position_Velocity1.txt'
    
    # Create analyzer instance
    analyzer = SatelliteAnalyzer(sat_file)
    
    # Select target satellite
    target_sat = 1
    
    # Get nearest satellites
    same_plane, diff_plane = analyzer.get_nearest_satellites(target_sat)
    
    print(f"\nTarget Satellite: SAT-{target_sat} (Orbital Plane {analyzer.get_orbital_plane(target_sat)})")
    print("\nNearest 4 satellites in same orbital plane:")
    for sat in same_plane:
        print(f"SAT-{sat} (Orbital Plane {analyzer.get_orbital_plane(sat)})")
    
    print("\nNearest 6 satellites in different orbital planes:")
    for sat in diff_plane:
        print(f"SAT-{sat} (Orbital Plane {analyzer.get_orbital_plane(sat)})")
    
    # Get position data for all satellites over 10 minutes
    all_sats = [target_sat] + same_plane + diff_plane
    positions_df = analyzer.get_positions_for_duration(all_sats)
    
    # Save results to CSV file
    output_file = 'satellite_positions_10min.csv'
    positions_df.to_csv(output_file, index=False)
    print(f"\nPosition data saved to: {output_file}")
    
    # Show data preview
    print("\nData Preview:")
    print(positions_df.head())
    
    # Plot satellite distribution (current positions)
    print("\nPlotting current positions...")
    analyzer.plot_satellites_3d(target_sat, same_plane, diff_plane)
    
    # Plot complete trajectories
    print("\nPlotting complete trajectories...")
    analyzer.plot_complete_trajectories(target_sat, same_plane, diff_plane)

if __name__ == '__main__':
    main() 