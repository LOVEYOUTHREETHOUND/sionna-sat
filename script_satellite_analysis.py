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
    
    def get_nearest_satellites(self, target_sat_idx, time_idx=0, n_same=4, n_diff=20):
        """获取目标卫星在指定时刻最近的卫星
        Args:
            target_sat_idx: 目标卫星索引
            time_idx: 时间索引（默认为第一个时刻）
            n_same: 同轨道最近卫星数
            n_diff: 不同轨道最近卫星数
        Returns:
            tuple: (同轨道最近n_same颗卫星索引, 不同轨道最近n_diff颗卫星索引)
        """
        target_plane = self.get_orbital_plane(target_sat_idx)
        target_pos = np.array([
            self.positions[target_sat_idx].iloc[time_idx]['x'],
            self.positions[target_sat_idx].iloc[time_idx]['y'],
            self.positions[target_sat_idx].iloc[time_idx]['z']
        ])
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
        same_plane = []
        diff_plane = []
        sorted_sats = sorted(distances.items(), key=lambda x: x[1])
        for sat_idx, dist in sorted_sats:
            plane = self.get_orbital_plane(sat_idx)
            if plane == target_plane and len(same_plane) < n_same:
                same_plane.append(sat_idx)
            elif plane != target_plane and len(diff_plane) < n_diff:
                diff_plane.append(sat_idx)
            if len(same_plane) == n_same and len(diff_plane) == n_diff:
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

    def split_diff_plane_satellites(self, target_plane, diff_plane_sats):
        """将不同轨道的卫星分为左右两侧（环形编号）"""
        left_sats = []
        right_sats = []
        for sat_idx in diff_plane_sats:
            plane = self.get_orbital_plane(sat_idx)
            # 计算环形距离
            delta = (plane - target_plane) % self.num_planes
            if 0 < delta <= self.num_planes // 2:
                right_sats.append(sat_idx)
            else:
                left_sats.append(sat_idx)
        return left_sats, right_sats

    def get_plane_normal(self, plane_id, time_idx=0):
        """计算指定轨道面在当前时刻的法向量（用该面上三颗卫星的位置叉乘获得）"""
        base_idx = plane_id * self.num_sats_per_plane
        idx1, idx2, idx3 = base_idx, base_idx+1, base_idx+2
        p1 = np.array([self.positions[idx1].iloc[time_idx][c] for c in ['x','y','z']])
        p2 = np.array([self.positions[idx2].iloc[time_idx][c] for c in ['x','y','z']])
        p3 = np.array([self.positions[idx3].iloc[time_idx][c] for c in ['x','y','z']])
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        return normal

    def split_diff_plane_by_plane_normal(self, target_sat_idx, diff_plane_sats, time_idx=0):
        """
        用目标卫星所在轨道面法向量，将不同轨道的卫星分为左右两侧。
        左右的定义：点积>0为右侧，<0为左侧。
        """
        target_plane = self.get_orbital_plane(target_sat_idx)
        normal = self.get_plane_normal(target_plane, time_idx)
        target_pos = np.array([self.positions[target_sat_idx].iloc[time_idx][c] for c in ['x','y','z']])
        left_sats, right_sats = [], []
        for sat_idx in diff_plane_sats:
            sat_pos = np.array([self.positions[sat_idx].iloc[time_idx][c] for c in ['x','y','z']])
            vec = sat_pos - target_pos
            sign = np.dot(vec, normal)
            if sign > 0:
                right_sats.append(sat_idx)
            else:
                left_sats.append(sat_idx)
        return left_sats, right_sats

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
        
        # Split diff_plane_sats into left and right by plane normal
        left_sats, right_sats = self.split_diff_plane_by_plane_normal(target_sat_idx, diff_plane_sats, time_idx)
        
        # Plot satellite trajectories (first 100 time points)
        num_points = 100
        for sat_idx in all_sats:
            pos_data = self.positions[sat_idx].iloc[:num_points]
            x = pos_data['x'].values
            y = pos_data['y'].values
            z = pos_data['z'].values
            if sat_idx == target_sat_idx:
                ax.plot(x, y, z, 'r-', alpha=0.3, linewidth=1)
            elif sat_idx in same_plane_sats:
                ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=1)
            elif sat_idx in left_sats:
                ax.plot(x, y, z, color='orange', alpha=0.3, linewidth=1)
            elif sat_idx in right_sats:
                ax.plot(x, y, z, color='purple', alpha=0.3, linewidth=1)
        
        # Plot current satellite positions
        target_pos = self.positions[target_sat_idx].iloc[time_idx]
        ax.scatter(target_pos['x'], target_pos['y'], target_pos['z'], 
                  c='red', marker='*', s=200, label='Target Satellite')
        for sat_idx in same_plane_sats:
            sat_pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'], 
                      c='blue', marker='o', s=100)
        for sat_idx in left_sats:
            sat_pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'], 
                      color='orange', marker='^', s=100)
        for sat_idx in right_sats:
            sat_pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'], 
                      color='purple', marker='^', s=100)
        
        # Set plot properties
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Satellite Distribution (ECEF Coordinates)')
        
        # Add legend
        ax.scatter([], [], c='red', marker='*', s=200, label='Target Satellite')
        ax.scatter([], [], c='blue', marker='o', s=100, label='Same-Plane Satellites')
        ax.scatter([], [], color='orange', marker='^', s=100, label='Diff-Plane Left')
        ax.scatter([], [], color='purple', marker='^', s=100, label='Diff-Plane Right')
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

    def plot_multi_targets_3d(self, target_sat_indices, same_plane_dict, diff_plane_dict, left_right_dict, time_idx=0):
        """多目标卫星的3D分布图，每个目标卫星的左右用不同颜色区分"""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        r = 6371000
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        # 颜色分配
        color_map = [
            ('red', 'orange'),
            ('green', 'lime'),
            ('purple', 'magenta')
        ]
        marker_map = ['*', 'o', 's']
        all_plotted = set()
        for i, target_sat in enumerate(target_sat_indices):
            same_plane_sats = same_plane_dict[target_sat]
            left_sats, right_sats = left_right_dict[target_sat]
            diff_plane_sats = diff_plane_dict[target_sat]
            # 轨迹
            all_sats = [target_sat] + same_plane_sats + diff_plane_sats
            for sat_idx in all_sats:
                pos_data = self.positions[sat_idx].iloc[:100]
                x = pos_data['x'].values
                y = pos_data['y'].values
                z = pos_data['z'].values
                if sat_idx == target_sat:
                    ax.plot(x, y, z, color=color_map[i][0], alpha=0.3, linewidth=1)
                elif sat_idx in same_plane_sats:
                    ax.plot(x, y, z, color=color_map[i][1], alpha=0.3, linewidth=1)
                elif sat_idx in left_sats:
                    ax.plot(x, y, z, color=color_map[i][0], alpha=0.3, linewidth=1)
                elif sat_idx in right_sats:
                    ax.plot(x, y, z, color=color_map[i][1], alpha=0.3, linewidth=1)
            # 当前时刻位置
            target_pos = self.positions[target_sat].iloc[time_idx]
            ax.scatter(target_pos['x'], target_pos['y'], target_pos['z'],
                       c=color_map[i][0], marker=marker_map[i], s=200, label=f'Target {i+1}')
            for sat_idx in same_plane_sats:
                sat_pos = self.positions[sat_idx].iloc[time_idx]
                ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'],
                           c=color_map[i][1], marker=marker_map[i], s=100, label=f'Target {i+1} Same-Plane' if (i, 'same') not in all_plotted else None)
                all_plotted.add((i, 'same'))
            for sat_idx in left_sats:
                sat_pos = self.positions[sat_idx].iloc[time_idx]
                ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'],
                           color=color_map[i][0], marker='^', s=100, label=f'Target {i+1} Left' if (i, 'left') not in all_plotted else None)
                all_plotted.add((i, 'left'))
            for sat_idx in right_sats:
                sat_pos = self.positions[sat_idx].iloc[time_idx]
                ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'],
                           color=color_map[i][1], marker='^', s=100, label=f'Target {i+1} Right' if (i, 'right') not in all_plotted else None)
                all_plotted.add((i, 'right'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Multi-Target Satellite Distribution (ECEF Coordinates)')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.savefig('satellite_multi_target_distribution.png', dpi=300, bbox_inches='tight')
        print("\nMulti-target satellite distribution plot saved as: satellite_multi_target_distribution.png")
        plt.show()

    def plot_targets_only(self, target_sat_indices, time_idx=0):
        """只显示三个目标卫星的位置和轨迹"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        r = 6371000
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        color_map = ['red', 'green', 'purple']
        marker_map = ['*', 'o', 's']
        for i, sat_idx in enumerate(target_sat_indices):
            pos_data = self.positions[sat_idx].iloc[:100]
            x = pos_data['x'].values
            y = pos_data['y'].values
            z = pos_data['z'].values
            ax.plot(x, y, z, color=color_map[i], alpha=0.5, linewidth=2)
            sat_pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'],
                       c=color_map[i], marker=marker_map[i], s=200, label=f'Target {i+1}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Three Target Satellites (ECEF Coordinates)')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.savefig('three_target_satellites.png', dpi=300, bbox_inches='tight')
        print("\nThree target satellites plot saved as: three_target_satellites.png")
        plt.show()

    def plot_single_target_neighbors(self, target_sat, same_plane, diff_plane, left_sats, right_sats, time_idx=0, idx=1):
        """显示单个目标卫星及其最近20个不同轨道卫星（左右分色）和同轨道4个卫星"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        r = 6371000
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        color_map = ('red', 'orange')
        # 轨迹
        all_sats = [target_sat] + same_plane + diff_plane
        for sat_idx in all_sats:
            pos_data = self.positions[sat_idx].iloc[:100]
            x = pos_data['x'].values
            y = pos_data['y'].values
            z = pos_data['z'].values
            if sat_idx == target_sat:
                ax.plot(x, y, z, color=color_map[0], alpha=0.5, linewidth=2)
            elif sat_idx in same_plane:
                ax.plot(x, y, z, color=color_map[1], alpha=0.3, linewidth=1)
            elif sat_idx in left_sats:
                ax.plot(x, y, z, color='orange', alpha=0.3, linewidth=1)
            elif sat_idx in right_sats:
                ax.plot(x, y, z, color='purple', alpha=0.3, linewidth=1)
        # 当前时刻位置
        sat_pos = self.positions[target_sat].iloc[time_idx]
        ax.scatter(sat_pos['x'], sat_pos['y'], sat_pos['z'],
                   c=color_map[0], marker='*', s=200, label='Target')
        for sat_idx in same_plane:
            pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(pos['x'], pos['y'], pos['z'],
                       c=color_map[1], marker='o', s=100, label='Same-Plane' if sat_idx == same_plane[0] else None)
        for sat_idx in left_sats:
            pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(pos['x'], pos['y'], pos['z'],
                       color='orange', marker='^', s=100, label='Diff-Plane Left' if sat_idx == left_sats[0] else None)
        for sat_idx in right_sats:
            pos = self.positions[sat_idx].iloc[time_idx]
            ax.scatter(pos['x'], pos['y'], pos['z'],
                       color='purple', marker='^', s=100, label='Diff-Plane Right' if sat_idx == right_sats[0] else None)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Target Satellite {target_sat} and 20 Nearest Neighbors')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.savefig(f'target_{idx}_neighbors.png', dpi=300, bbox_inches='tight')
        print(f"\nTarget satellite {target_sat} neighbors plot saved as: target_{idx}_neighbors.png")
        plt.show()

    def find_three_closest_from_nearest(self, center_sat_idx=1, time_idx=0, n_nearest=20):
        # 1. 找sat-1最近的20个卫星
        center_pos = np.array([
            self.positions[center_sat_idx].iloc[time_idx]['x'],
            self.positions[center_sat_idx].iloc[time_idx]['y'],
            self.positions[center_sat_idx].iloc[time_idx]['z']
        ])
        distances = {}
        for sat_idx in range(self.total_sats):
            if sat_idx == center_sat_idx:
                continue
            sat_pos = np.array([
                self.positions[sat_idx].iloc[time_idx]['x'],
                self.positions[sat_idx].iloc[time_idx]['y'],
                self.positions[sat_idx].iloc[time_idx]['z']
            ])
            dist = np.linalg.norm(sat_pos - center_pos)
            distances[sat_idx] = dist
        nearest_sats = sorted(distances.items(), key=lambda x: x[1])[:n_nearest]
        candidate_indices = [center_sat_idx] + [idx for idx, _ in nearest_sats]
        # 2. 在这21颗卫星中穷举三元组
        import itertools
        min_max_dist = float('inf')
        best_triplet = None
        positions = np.array([
            [self.positions[idx].iloc[time_idx]['x'],
            self.positions[idx].iloc[time_idx]['y'],
            self.positions[idx].iloc[time_idx]['z']]
            for idx in candidate_indices
        ])
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(positions))
        n = len(candidate_indices)
        for i, j, k in itertools.combinations(range(n), 3):
            dists = [dist_matrix[i, j], dist_matrix[i, k], dist_matrix[j, k]]
            max_dist = max(dists)
            if max_dist < min_max_dist:
                min_max_dist = max_dist
                best_triplet = (candidate_indices[i], candidate_indices[j], candidate_indices[k])
        return list(best_triplet)


def main():
    # Configure file path
    sat_file = 'data\satellite_result\Satellite1_Fixed_Position_Velocity1.txt'
    analyzer = SatelliteAnalyzer(sat_file)
    # 自动选择三颗彼此最近的卫星
    print("Selecting 3 closest satellites as targets...")
    target_sats = analyzer.find_three_closest_from_nearest(time_idx=0)
    print(f"Selected target satellites: {target_sats}")
    same_plane_dict = {}
    diff_plane_dict = {}
    left_right_dict = {}
    for target_sat in target_sats:
        same_plane, diff_plane = analyzer.get_nearest_satellites(target_sat, n_same=4, n_diff=20)
        same_plane_dict[target_sat] = same_plane
        diff_plane_dict[target_sat] = diff_plane
        left_sats, right_sats = analyzer.split_diff_plane_by_plane_normal(target_sat, diff_plane, 0)
        left_right_dict[target_sat] = (left_sats, right_sats)
        print(f"\nTarget Satellite: SAT-{target_sat} (Orbital Plane {analyzer.get_orbital_plane(target_sat)})")
        print("  Nearest 4 satellites in same orbital plane:")
        for sat in same_plane:
            print(f"    SAT-{sat} (Orbital Plane {analyzer.get_orbital_plane(sat)})")
        print("  Nearest 20 satellites in different orbital planes (Left):")
        for sat in left_sats:
            print(f"    SAT-{sat} (Orbital Plane {analyzer.get_orbital_plane(sat)})")
        print("  Nearest 20 satellites in different orbital planes (Right):")
        for sat in right_sats:
            print(f"    SAT-{sat} (Orbital Plane {analyzer.get_orbital_plane(sat)})")
    # 合并所有目标卫星相关卫星索引，获取10分钟内位置数据
    all_sats = set(target_sats)
    for v in same_plane_dict.values():
        all_sats.update(v)
    for v in diff_plane_dict.values():
        all_sats.update(v)
    positions_df = analyzer.get_positions_for_duration(list(all_sats))
    output_file = 'satellite_positions_10min.csv'
    positions_df.to_csv(output_file, index=False)
    print(f"\nPosition data saved to: {output_file}")
    print("\nData Preview:")
    print(positions_df.head())
    print("\nPlotting three target satellites only...")
    analyzer.plot_targets_only(target_sats)
    for i, target_sat in enumerate(target_sats):
        print(f"\nPlotting neighbors for target satellite {target_sat}...")
        analyzer.plot_single_target_neighbors(
            target_sat,
            same_plane_dict[target_sat],
            diff_plane_dict[target_sat],
            left_right_dict[target_sat][0],
            left_right_dict[target_sat][1],
            idx=i+1
        )
    print("\nPlotting multi-target satellite distribution...")
    analyzer.plot_multi_targets_3d(target_sats, same_plane_dict, diff_plane_dict, left_right_dict)

if __name__ == '__main__':
    main() 