import numpy as np
import json
import os
import pandas as pd
from datetime import datetime

class StaticData:
    def __init__(self, system_param_path, satellite_result_path, topology_json_dir, 
                 num_sats=1800, num_timesteps=None, 
                 sat_range=None, sc_range=None):  # 添加范围参数
        """初始化StaticData
        Args:
            system_param_path: 系统参数文件路径
            satellite_result_path: 卫星位置数据文件路径
            topology_json_dir: 拓扑JSON文件目录
            num_sats: 卫星总数，默认1800
            num_timesteps: 时隙数，默认None表示加载所有时隙
            sat_range: 要仿真的卫星范围，格式(start, end)，默认None表示所有卫星
            sc_range: 要仿真的SC范围，格式(start, end)，默认None表示所有SC
        """
        # 1. 加载系统参数
        self.system_params = self._load_system_params(system_param_path)

        # 2. 加载所有卫星在每个时隙的ECEF位置
        self.sat_positions = self._load_satellite_positions(
            satellite_result_path, 
            num_sats, 
            num_timesteps,
            sat_range
        )

        # 3. 加载两个目标区域各自的SC中心
        self.sc_centers = self._load_sc_centers(topology_json_dir)
        if sc_range is not None:
            start, end = sc_range
            self.sc_centers = self.sc_centers[start:end]

        # 4. 加载所有用户的ECEF位置信息和用户-SC映射关系
        self.user_positions, self.user_sc_mapping = self._load_user_positions(topology_json_dir)
        if sc_range is not None:
            # 筛选在选定SC范围内的用户
            start, end = sc_range
            valid_users = np.where((self.user_sc_mapping >= start) & (self.user_sc_mapping < end))[0]
            self.user_positions = self.user_positions[valid_users]
            self.user_sc_mapping = self.user_sc_mapping[valid_users]
            # 重新映射SC索引
            self.user_sc_mapping -= start

        # 5. 预计算系统参数
        # 计算EIRP
        eirp_density = float(self.system_params['eirp_density'])  # dBW/MHz
        bandwidth = float(self.system_params['bandwidth'])  # Hz
        self.system_params['EIRP'] = eirp_density + 10 * np.log10(bandwidth/1e6)  # dBW

        # 计算波长
        carrier_freq = float(self.system_params['carrier_freq'])  # Hz
        speed_of_light = float(self.system_params['speed_of_light'])  # m/s
        self.system_params['wavelength'] = speed_of_light / carrier_freq  # m

        # 计算ka参数 (用于增益衰减计算)
        self.system_params['ka'] = 2 * np.pi / self.system_params['wavelength'] * self.system_params['antenna_diameter'] / 2

    def _load_system_params(self, path):
        """加载系统参数
        Args:
            path: 系统参数JSON文件路径
        Returns:
            包含所有系统参数的字典
        """
        with open(path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # 合并所有参数到一个字典
        all_params = {}
        all_params.update(params['basic_params'])
        all_params.update(params['physical_constants'])
        
        # 确保参数单位正确
        speed_of_light = float(all_params['speed_of_light'])  # 应该是3e8 m/s
        carrier_freq = float(all_params['carrier_freq'])      # 应该是Hz
        
        # 计算波长 (米)
        wavelength = speed_of_light / carrier_freq
        all_params['wavelength'] = wavelength
        print(f"\n波长计算:")
        print(f"  光速: {speed_of_light} m/s")
        print(f"  载波频率: {carrier_freq/1e9} GHz")
        print(f"  波长: {wavelength*100:.2f} cm")
        
        # 设置天线直径（不再使用波束数计算）
        # 对于Ka波段卫星通信，天线直径通常在1-2米范围
        antenna_diameter = 1.5  # 设置为1.5米
        all_params['antenna_diameter'] = antenna_diameter
        
        # 计算ka参数
        ka = (2 * np.pi / wavelength) * (antenna_diameter / 2)
        all_params['ka'] = ka
        print(f"\n天线参数:")
        print(f"  天线直径: {antenna_diameter:.2f} m")
        print(f"  ka参数: {ka:.2f}")
        
        # 设置接收天线增益（用户终端）
        all_params['receiver_gain'] = 25.0  # 设置为25dB，这是典型的用户终端天线增益
        
        # 验证参数合理性
        if ka > 200:
            print(f"\n警告: ka参数 ({ka:.2f}) 可能过大，这会导致过大的增益衰减")
        if antenna_diameter > 3:
            print(f"\n警告: 天线直径 ({antenna_diameter:.2f}m) 可能过大")
        
        return all_params

    def _load_satellite_positions(self, sat_path, num_sats=1800, num_timesteps=None, sat_range=None):
        """加载卫星位置数据
        Args:
            sat_path: 卫星位置数据文件路径
            num_sats: 卫星总数
            num_timesteps: 要加载的时隙数，None表示全部加载
            sat_range: 要加载的卫星范围(start, end)，None表示加载所有卫星
        Returns:
            numpy array of shape [num_sats, num_timesteps, 3] containing ECEF positions
        """
        # 确定要加载的卫星范围
        start_sat = 0 if sat_range is None else sat_range[0]
        end_sat = num_sats if sat_range is None else sat_range[1]
        
        # 读取整个文件
        with open(sat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 存储所有卫星的位置数据
        all_sat_pos = []
        current_sat_data = []
        
        # 标记是否在读取数据行
        reading_data = False
        
        # 遍历文件的每一行
        for line in lines:
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
                
            # 检查是否是表头行
            if line.startswith('Time'):
                reading_data = False
                # 如果已经收集了上一个卫星的数据，保存它
                if current_sat_data:
                    all_sat_pos.append(np.array(current_sat_data))
                    current_sat_data = []
                continue
                
            # 跳过分隔线
            if line.startswith('--'):
                reading_data = True
                continue
            
            # 处理数据行
            if reading_data:
                try:
                    # 检查行是否包含日期时间格式（至少包含年份）
                    if not any(year in line for year in ['2025', '2024', '2026']):
                        continue
                        
                    # 分割行，保留非空字符串
                    parts = [p for p in line.split() if p.strip()]
                    
                    # 确保有足够的数据列（时间戳占用4个部分：日期、月份、年份、时间）
                    if len(parts) >= 7:  # 4(时间戳) + 3(xyz坐标)
                        # 提取坐标（跳过时间戳部分，即前4个元素）
                        x = float(parts[4]) * 1000  # km to m
                        y = float(parts[5]) * 1000  # km to m
                        z = float(parts[6]) * 1000  # km to m
                        
                        current_sat_data.append([x, y, z])
                        
                        # 如果达到了指定的时隙数，保存当前卫星数据并开始新的卫星
                        if num_timesteps is not None and len(current_sat_data) >= num_timesteps:
                            all_sat_pos.append(np.array(current_sat_data))
                            current_sat_data = []
                            reading_data = False
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line: {line}")
                    continue
        
        # 添加最后一个卫星的数据
        if current_sat_data:
            all_sat_pos.append(np.array(current_sat_data))
        
        # 检查是否获取到了数据
        if not all_sat_pos:
            raise ValueError("No valid satellite position data found in file")
        
        # 确保所有卫星都有相同数量的时间点
        min_timesteps = min(len(pos) for pos in all_sat_pos)
        all_sat_pos = [pos[:min_timesteps] for pos in all_sat_pos]
        
        # 只返回指定范围内的卫星数据
        all_sat_pos = all_sat_pos[start_sat:end_sat]
        
        return np.stack(all_sat_pos, axis=0)

    def _load_sc_centers(self, json_dir):
        sc_centers = []
        for area in ['A1', 'A2']:
            path = os.path.join(json_dir, f'area_{area}_coverage.json')
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cell in data['cells']:
                    ecef = cell['center_coordinates']['ecef']
                    sc_centers.append([ecef['x'], ecef['y'], ecef['z']])
        return np.array(sc_centers)

    def _load_user_positions(self, json_dir):
        user_pos = []
        user_sc_mapping = []  # 存储每个用户对应的SC索引
        sc_idx = 0
        for area in ['A1', 'A2']:
            path = os.path.join(json_dir, f'area_{area}_coverage.json')
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cell in data['cells']:
                    for user in cell['users']:
                        ecef = user['coordinates']['ecef']
                        user_pos.append([ecef['x'], ecef['y'], ecef['z']])
                        user_sc_mapping.append(sc_idx)
                    sc_idx += 1
        return np.array(user_pos), np.array(user_sc_mapping)


class SimulateSnapshot:
    def __init__(self, static_data: StaticData, snapshot_idx: int):
        self.static_data = static_data
        self.snapshot_idx = snapshot_idx
        self.num_sats = static_data.sat_positions.shape[0]
        self.num_sc = static_data.sc_centers.shape[0]
        self.num_users = static_data.user_positions.shape[0]
        
        # 1. 计算每个小区中心距离最近的卫星
        sat_pos = static_data.sat_positions[:, snapshot_idx, :]  # [num_sats, 3]
        sc_pos = static_data.sc_centers  # [num_sc, 3]
        # 计算距离矩阵 [num_sc, num_sats]
        self.dists = np.linalg.norm(sc_pos[:, None, :] - sat_pos[None, :, :], axis=2)
        # 关联矩阵: [num_sc, num_sats]，每行只有一个1，其余为0
        self.association = np.zeros((self.num_sc, self.num_sats), dtype=int)
        closest_sat_idx = np.argmin(self.dists, axis=1)  # [num_sc]
        self.association[np.arange(self.num_sc), closest_sat_idx] = 1

        # 2. 计算每个用户的干扰卫星
        self.interference_association = np.zeros((self.num_users, self.num_sats), dtype=int)
        for user_idx in range(self.num_users):
            # 获取用户所在SC的索引和服务卫星
            user_sc_idx = self.static_data.user_sc_mapping[user_idx]
            serving_sat_idx = closest_sat_idx[user_sc_idx]
            
            # 获取服务卫星的星下点
            serving_sat_subsatellite = self._get_subsatellite_point(sat_pos[serving_sat_idx])
            
            # 定义干扰区域（目标区域外一圈卫星）
            for sat_idx in range(self.num_sats):
                if sat_idx == serving_sat_idx:
                    # 服务卫星：标记为2（用于后续区分星内干扰）
                    self.interference_association[user_idx, sat_idx] = 2
                    continue
                    
                # 获取候选干扰卫星的星下点
                sat_subsatellite = self._get_subsatellite_point(sat_pos[sat_idx])
                
                # 计算与服务卫星星下点的距离（地表距离）
                dist = self._compute_surface_distance(serving_sat_subsatellite, sat_subsatellite)
                
                # 如果在干扰区域内（假设一圈的距离阈值为2000km）
                if dist <= 2000e3:  # 2000km
                    # 检查卫星是否对用户可见
                    if self._is_satellite_visible(sat_pos[sat_idx], self.static_data.user_positions[user_idx]):
                        self.interference_association[user_idx, sat_idx] = 1

        # 预计算噪声功率
        self.precompute_noise()

    def _get_subsatellite_point(self, sat_pos):
        """计算卫星的星下点坐标（ECEF）
        Args:
            sat_pos: 卫星ECEF坐标 [3,]
        Returns:
            星下点ECEF坐标 [3,]
        """
        earth_radius = 6371e3  # 地球半径（米）
        return earth_radius * sat_pos / np.linalg.norm(sat_pos)

    def _compute_surface_distance(self, point1, point2):
        """计算地球表面上两点间的大圆距离
        Args:
            point1: ECEF坐标 [3,]
            point2: ECEF坐标 [3,]
        Returns:
            大圆距离（米）
        """
        earth_radius = 6371e3  # 地球半径（米）
        # 计算两点的单位向量
        v1 = point1 / np.linalg.norm(point1)
        v2 = point2 / np.linalg.norm(point2)
        # 计算夹角
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # 返回大圆距离
        return earth_radius * angle

    def _is_satellite_visible(self, sat_pos, user_pos):
        """判断卫星是否对用户可见
        Args:
            sat_pos: 卫星ECEF坐标 [3,]
            user_pos: 用户ECEF坐标 [3,]
        Returns:
            bool: 是否可见
        """
        # 计算卫星仰角
        earth_radius = 6371e3  # 地球半径（米）
        user_height = np.linalg.norm(user_pos) - earth_radius
        sat_range = np.linalg.norm(sat_pos - user_pos)
        cos_elevation = (np.linalg.norm(sat_pos) ** 2 + sat_range ** 2 - 
                        (earth_radius + user_height) ** 2) / (2 * np.linalg.norm(sat_pos) * sat_range)
        elevation = np.arccos(np.clip(cos_elevation, -1.0, 1.0))
        
        # 假设最小可见仰角为10度
        min_elevation = np.deg2rad(10)
        return elevation >= min_elevation

    def precompute_noise(self):
        # 1. 计算等效噪声温度 Teq
        # G/T = G_R - 10*log10(Teq)  =>  Teq = 10**((G_R - G/T)/10)
        g_over_t = float(self.static_data.system_params['g_over_t'])  # dB/K
        receiver_gain = float(self.static_data.system_params['receiver_gain'])  # dB
        boltzmann_constant = float(self.static_data.system_params['boltzmann_constant'])  # J/K
        bandwidth = float(self.static_data.system_params['bandwidth'])  # Hz
        # G/T = G_R - 10*log10(Teq)
        # => Teq = 10**((G_R - G/T)/10)
        Teq = 10 ** ((receiver_gain - g_over_t) / 10)
        # 2. 计算噪声功率 N0 = k * Teq * B
        self.N0 = boltzmann_constant * Teq * bandwidth  # 单位：瓦
        print('该用户噪声计算值为：', self.N0,'W')
        return self.N0

    def run_timeslots(self, num_timeslots):
        timeslot_sim = SimulateTimeslot()
        for t in range(num_timeslots):
            # TODO: 传递必要参数，进行SINR计算
            timeslot_sim.run()



class SimulateTimeslot:
    def __init__(self, static_data: StaticData, association: np.ndarray, interference_association: np.ndarray, system_params: dict, timeslot_idx: int, snapshot_sat_positions: np.ndarray):
        self.static_data = static_data
        self.association = association  # [num_sc, num_sats]
        self.interference_association = interference_association  # [num_users, num_sats]
        self.system_params = system_params
        self.timeslot_idx = timeslot_idx
        self.num_sats = association.shape[1]
        self.num_sc = association.shape[0]
        self.num_users = static_data.user_positions.shape[0]
        self.beams_per_sat = int(system_params.get('beams_per_sat', 1))
        self.beam_schedule = self._round_robin_beam_schedule()
        # 预计算噪声功率
        self.N0 = self._precompute_noise()
        # 存储snapshot时刻的卫星位置 [num_sats, 3]
        self.snapshot_sat_positions = snapshot_sat_positions

    def _round_robin_beam_schedule(self):
        schedule = -np.ones((self.num_sats, self.beams_per_sat), dtype=int)
        for sat in range(self.num_sats):
            sc_indices = np.where(self.association[:, sat] == 1)[0]
            num_sc = len(sc_indices)
            offset = self.timeslot_idx * self.beams_per_sat % num_sc if num_sc > 0 else 0
            for b in range(self.beams_per_sat):
                if num_sc > 0:
                    idx = (offset + b) % num_sc
                    schedule[sat, b] = sc_indices[idx]
        return schedule

    def _precompute_noise(self):
        """预计算噪声功率"""
        # 1. 计算等效噪声温度 Teq
        # G/T = G_R - 10*log10(Teq)  =>  Teq = 10**((G_R - G/T)/10)
        g_over_t = float(self.system_params['g_over_t'])  # dB/K
        receiver_gain = float(self.system_params['receiver_gain'])  # dB
        boltzmann_constant = float(self.system_params['boltzmann_constant'])  # J/K
        bandwidth = float(self.system_params['bandwidth'])  # Hz
        
        # 计算等效噪声温度
        Teq = 10 ** ((receiver_gain - g_over_t) / 10)
        
        # 2. 计算噪声功率 N0 = k * Teq * B
        N0 = boltzmann_constant * Teq * bandwidth  # 单位：瓦
        print('系统噪声功率计算值为：', N0, 'W')
        return N0

    def compute_received_power(self, sat_pos, user_pos, sc_center_pos):
        """计算接收功率
        Args:
            sat_pos: 卫星ECEF坐标 [3,]
            user_pos: 用户ECEF坐标 [3,]
            sc_center_pos: SC中心ECEF坐标 [3,]
        Returns:
            接收功率 (dBW)
        """
        print("\n接收功率计算过程:")
        print("1. 初始参数:")
        print(f"  卫星位置 (ECEF): {sat_pos}")
        print(f"  用户位置 (ECEF): {user_pos}")
        print(f"  小区中心 (ECEF): {sc_center_pos}")
        
        # 1. 获取预计算的EIRP
        EIRP = self.static_data.system_params['EIRP']  # dBW
        print("\n2. EIRP计算:")
        print(f"  EIRP密度: {self.static_data.system_params['eirp_density']} dBW/MHz")
        print(f"  带宽: {self.static_data.system_params['bandwidth']/1e6} MHz")
        print(f"  最终EIRP: {EIRP} dBW")
        
        # 2. 计算离轴角
        D_SC = sc_center_pos - sat_pos  # [3,]
        D_UE = user_pos - sat_pos  # [3,]
        cos_alpha = np.dot(D_SC, D_UE) / (np.linalg.norm(D_SC) * np.linalg.norm(D_UE))
        alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
        print("\n3. 离轴角计算:")
        print(f"  卫星到小区中心向量: {D_SC}")
        print(f"  卫星到用户向量: {D_UE}")
        print(f"  向量夹角余弦: {cos_alpha}")
        print(f"  离轴角: {np.rad2deg(alpha):.2f} 度")
        
        # 3. 计算增益衰减
        ka = self.static_data.system_params['ka']
        if alpha == 0:
            delta_G = 0  # dB
        else:
            x = ka * np.sin(alpha)
            from scipy.special import j1
            delta_G = 10 * np.log10(4 * (j1(x)/x)**2)  # dB
        print("\n4. 增益衰减计算:")
        print(f"  ka参数: {ka}")
        if alpha != 0:
            print(f"  x = ka * sin(alpha): {x}")
            print(f"  贝塞尔函数值 j1(x)/x: {j1(x)/x}")
        print(f"  增益衰减: {delta_G} dB")
        
        # 4. 接收天线增益
        G_R = float(self.static_data.system_params['receiver_gain'])  # dB
        print("\n5. 接收天线增益:")
        print(f"  G_R: {G_R} dB")
        
        # 5. 计算路径损耗
        d = np.linalg.norm(D_UE)  # 距离(m)
        f0 = float(self.static_data.system_params['carrier_freq']) / 1e9  # GHz
        L_s = float(self.static_data.system_params['scintillation_loss'])  # dB
        # 阴影衰落（对数正态，标准差2dB）
        F_s = np.random.normal(0, 2.0)  # dB
        print(f"\n6. 路径损耗计算:")
        print(f"  传播距离: {d/1000:.2f} km")
        print(f"  载波频率: {f0} GHz")
        print(f"  闪烁损耗: {L_s} dB")
        print(f"  阴影衰落: {F_s:.2f} dB (sigma=2dB)")
        # TODO: 还未添加大气吸收损耗
        PL = 32.45 + 20 * np.log10(f0 * d) + F_s + L_s  # dB
        print(f"  总路径损耗: {PL:.2f} dB")
        
        # 6. 计算总接收功率
        P_rx = EIRP + delta_G + G_R - PL  # dBW
        print("\n7. 最终接收功率计算:")
        print(f"  P_rx = EIRP + delta_G + G_R - PL")
        print(f"  P_rx = {EIRP} + ({delta_G}) + {G_R} - {PL}")
        print(f"  P_rx = {P_rx} dBW")
        print(f"  P_rx = {10**(P_rx/10):.2e} W")
        return P_rx

    def compute_interference(self, user_idx):
        """计算指定用户的总干扰功率
        Args:
            user_idx: 用户索引
        Returns:
            干扰功率 (W, 线性单位)
        """
        # 获取用户位置
        user_pos = self.static_data.user_positions[user_idx]
        # 使用snapshot时刻的卫星位置
        sat_pos = self.snapshot_sat_positions
        # 获取用户的干扰卫星
        interfering_sats = np.where(self.interference_association[user_idx] == 1)[0]
        
        total_interference = 0
        for sat_idx in interfering_sats:
            # 获取该卫星当前激活的波束指向的SC中心
            active_sc_indices = self.beam_schedule[sat_idx]
            active_sc_indices = active_sc_indices[active_sc_indices >= 0]  # 移除-1
            
            for sc_idx in active_sc_indices:
                sc_center_pos = self.static_data.sc_centers[sc_idx]
                # 计算干扰功率（dBW）
                interference_power_db = self.compute_received_power(
                    sat_pos[sat_idx],
                    user_pos,
                    sc_center_pos
                )
                # 转换为线性单位（W）并累加
                total_interference += 10 ** (interference_power_db / 10)
        
        return total_interference

    def compute_interference_vectorized(self):
        """向量化计算所有用户的干扰功率
        Returns:
            所有用户的干扰功率数组 (W, 线性单位) [num_users,]
        """
        # 1. 获取当前时隙的卫星位置 [num_sats, 3]
        sat_positions = self.snapshot_sat_positions
        
        # 2. 创建干扰功率数组 [num_users]
        total_interference = np.zeros(self.num_users)
        
        # 3. 创建活跃波束矩阵 [num_sats, num_sc]
        active_beams = np.zeros((self.num_sats, self.static_data.sc_centers.shape[0]), dtype=bool)
        for sat_idx in range(self.num_sats):
            active_beams[sat_idx, self.beam_schedule[sat_idx][self.beam_schedule[sat_idx] >= 0]] = True
        
        # 4. 批量计算每个用户的干扰
        for user_idx in range(self.num_users):
            user_pos = self.static_data.user_positions[user_idx]  # [3,]
            
            # 获取服务卫星（标记为2）和干扰卫星（标记为1）
            serving_sat = np.where(self.interference_association[user_idx] == 2)[0][0]
            interfering_sats = np.where(self.interference_association[user_idx] == 1)[0]
            
            # 4.1 计算星内干扰（服务卫星的其他波束）
            intra_interference = 0
            active_sc_indices = np.where(active_beams[serving_sat])[0]
            if len(active_sc_indices) > 0:
                # 排除用户自己的SC
                user_sc = self.static_data.user_sc_mapping[user_idx]
                active_sc_indices = active_sc_indices[active_sc_indices != user_sc]
                
                if len(active_sc_indices) > 0:
                    interference_powers_db = np.zeros(len(active_sc_indices))
                    for i, sc_idx in enumerate(active_sc_indices):
                        interference_powers_db[i] = self.compute_received_power(
                            sat_positions[serving_sat],
                            user_pos,
                            self.static_data.sc_centers[sc_idx]
                        )
                    intra_interference = np.sum(10 ** (interference_powers_db / 10))
            
            # 4.2 计算星间干扰（其他卫星的所有活跃波束）
            inter_interference = 0
            for sat_idx in interfering_sats:
                active_sc_indices = np.where(active_beams[sat_idx])[0]
                if len(active_sc_indices) > 0:
                    interference_powers_db = np.zeros(len(active_sc_indices))
                    for i, sc_idx in enumerate(active_sc_indices):
                        interference_powers_db[i] = self.compute_received_power(
                            sat_positions[sat_idx],
                            user_pos,
                            self.static_data.sc_centers[sc_idx]
                        )
                    inter_interference += np.sum(10 ** (interference_powers_db / 10))
            
            # 4.3 总干扰为星内干扰和星间干扰之和
            total_interference[user_idx] = intra_interference + inter_interference
        
        return total_interference

    def compute_noise(self):
        """返回预计算的噪声功率值"""
        return self.N0

    def compute_sinr(self):
        """计算当前时隙所有用户的SINR
        Returns:
            所有用户的SINR值数组 (dB) [num_users,]
        """
        # 1. 创建波束调度查找矩阵 [num_sats, num_sc]
        beam_schedule_matrix = np.zeros((self.num_sats, self.static_data.sc_centers.shape[0]), dtype=bool)
        for sat_idx in range(self.num_sats):
            beam_schedule_matrix[sat_idx, self.beam_schedule[sat_idx][self.beam_schedule[sat_idx] >= 0]] = True
        
        # 2. 获取当前时隙的卫星位置
        sat_positions = self.snapshot_sat_positions  # [num_sats, 3]
        
        # 3. 获取噪声功率
        noise = self.compute_noise()  # 标量
        
        # 4. 为每个用户计算SINR
        sinr_all = np.full(self.num_users, float('-inf'))
        
        print("\n开始计算每个用户的SINR:")
        for user_idx in range(self.num_users):
            # 获取用户的服务小区和服务卫星
            user_sc = self.static_data.user_sc_mapping[user_idx]
            serving_sat = np.where(self.interference_association[user_idx] == 2)[0][0]
            
            print(f"\n用户 {user_idx}:")
            print(f"  所属小区: SC-{user_sc}")
            print(f"  服务卫星: SAT-{serving_sat}")
            
            # 检查用户是否被调度
            if not beam_schedule_matrix[serving_sat, user_sc]:
                print("  状态: 未被调度")
                continue
            
            print("  状态: 已被调度")
            
            # 计算接收功率
            rx_power_db = self.compute_received_power(
                sat_positions[serving_sat],
                self.static_data.user_positions[user_idx],
                self.static_data.sc_centers[user_sc]
            )
            rx_power = 10 ** (rx_power_db / 10)
            print(f"  接收功率: {rx_power_db:.2f} dBW ({rx_power:.2e} W)")
            
            # 计算星内干扰
            intra_interference = 0
            active_sc_indices = np.where(beam_schedule_matrix[serving_sat])[0]
            active_sc_indices = active_sc_indices[active_sc_indices != user_sc]
            
            print("  星内干扰:")
            if len(active_sc_indices) > 0:
                for sc_idx in active_sc_indices:
                    interference_power_db = self.compute_received_power(
                        sat_positions[serving_sat],
                        self.static_data.user_positions[user_idx],
                        self.static_data.sc_centers[sc_idx]
                    )
                    interference_power = 10 ** (interference_power_db / 10)
                    intra_interference += interference_power
                    print(f"    - 来自SC-{sc_idx}: {interference_power_db:.2f} dBW ({interference_power:.2e} W)")
            else:
                print("    无星内干扰")
            
            print(f"    总星内干扰: {10*np.log10(intra_interference):.2f} dBW ({intra_interference:.2e} W)")
            
            # 计算星间干扰
            inter_interference = 0
            interfering_sats = np.where(self.interference_association[user_idx] == 1)[0]
            
            print("  星间干扰:")
            if len(interfering_sats) > 0:
                for sat_idx in interfering_sats:
                    active_sc_indices = np.where(beam_schedule_matrix[sat_idx])[0]
                    if len(active_sc_indices) > 0:
                        sat_total_interference = 0
                        print(f"    来自SAT-{sat_idx}:")
                        for sc_idx in active_sc_indices:
                            interference_power_db = self.compute_received_power(
                                sat_positions[sat_idx],
                                self.static_data.user_positions[user_idx],
                                self.static_data.sc_centers[sc_idx]
                            )
                            interference_power = 10 ** (interference_power_db / 10)
                            sat_total_interference += interference_power
                            print(f"      - SC-{sc_idx}: {interference_power_db:.2f} dBW ({interference_power:.2e} W)")
                        inter_interference += sat_total_interference
                        print(f"      小计: {10*np.log10(sat_total_interference):.2f} dBW ({sat_total_interference:.2e} W)")
            else:
                print("    无星间干扰")
            
            print(f"    总星间干扰: {10*np.log10(inter_interference):.2f} dBW ({inter_interference:.2e} W)")
            
            # 计算总干扰
            total_interference = intra_interference + inter_interference
            print(f"  总干扰功率: {10*np.log10(total_interference):.2f} dBW ({total_interference:.2e} W)")
            print(f"  噪声功率: {10*np.log10(noise):.2f} dBW ({noise:.2e} W)")
            
            # 计算SINR
            # denominator = total_interference + noise
            denominator = noise
            if denominator > 0:
                sinr = rx_power / denominator
                print(f"sinr: {sinr}")
                sinr_db = 10 * np.log10(sinr)
                sinr_all[user_idx] = sinr_db
                print(f"  SINR: {sinr_db:.2f} dB")
            else:
                print("  SINR: 无效 (分母为0)")
        
        return sinr_all


class SimulationResultWriter:
    def __init__(self, num_users, num_timeslots, result_dir_root='simulation_results'):
        # 生成仿真开始时间戳
        self.start_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.num_users = num_users
        self.num_timeslots = num_timeslots
        # 结果目录
        self.result_dir = os.path.join(result_dir_root, self.start_time_str)
        os.makedirs(self.result_dir, exist_ok=True)
        self.sinr_table = None
        self.detail_records = []

    def record_sinr(self, sinr_results):
        """记录每个时隙的SINR结果（shape: [num_users,]）"""
        if self.sinr_table is None:
            self.sinr_table = []
        # 兼容list和array
        self.sinr_table.append(list(sinr_results))

    def record_detail(self, slot_idx, user_idx, rx_power_db, intra_db, inter_db, noise_db, sinr_db):
        """记录每个用户每个时隙的详细中间过程"""
        self.detail_records.append({
            'slot': slot_idx,
            'user': user_idx,
            'rx_power_db': rx_power_db,
            'intra_interf_db': intra_db,
            'inter_interf_db': inter_db,
            'noise_db': noise_db,
            'sinr_db': sinr_db
        })

    def save(self):
        # 保存SINR表格
        sinr_arr = np.array(self.sinr_table)  # shape: [num_timeslots, num_users]
        sinr_df = pd.DataFrame(sinr_arr.T)  # 行：用户，列：时隙
        sinr_df.index.name = 'user'
        sinr_df.columns = [f'slot_{i}' for i in range(self.num_timeslots)]
        sinr_file = os.path.join(self.result_dir, f'sinr_user{self.num_users}_slot{self.num_timeslots}_table.csv')
        sinr_df.to_csv(sinr_file)
        print(f"SINR表格已保存: {sinr_file}")
        # 保存详细过程表格
        detail_df = pd.DataFrame(self.detail_records)
        detail_file = os.path.join(self.result_dir, f'sinr_detail_user{self.num_users}_slot{self.num_timeslots}_table.csv')
        detail_df.to_csv(detail_file, index=False)
        print(f"详细过程表格已保存: {detail_file}")


def main():
    """运行基本仿真实验"""
    # 1. 配置仿真参数
    system_param_path = 'config/system_params.json'  # 更新为.json后缀
    satellite_result_path = 'data/satellite_result/Satellite1_Fixed_Position_Velocity.txt'
    topology_json_dir = 'data/topology_result/json'
    
    # 创建结果保存目录
    import os
    results_dir = 'simulation_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建结果保存目录: {results_dir}")

    # 仿真范围配置
    num_sats = 1  # 前5个卫星
    num_scs = 1  # A1区域前100个小区
    simulation_time = 10  # 10秒
    snapshot_interval = 10  # 10秒一个snapshot
    timeslot_interval = 1  # 1秒一个timeslot
    
    # 计算snapshot和timeslot数量
    num_snapshots = simulation_time // snapshot_interval
    slots_per_snapshot = snapshot_interval // timeslot_interval
    
    print(f"开始仿真实验：")
    print(f"- 卫星数量: {num_sats}")
    print(f"- 小区数量: {num_scs}")
    print(f"- 仿真时长: {simulation_time}秒")
    print(f"- Snapshot间隔: {snapshot_interval}秒")
    print(f"- Timeslot间隔: {timeslot_interval}秒")
    
    # 2. 初始化StaticData
    static_data = StaticData(
        system_param_path=system_param_path,
        satellite_result_path=satellite_result_path,
        topology_json_dir=topology_json_dir,
        num_sats=1800,  # 总卫星数
        num_timesteps=simulation_time,  # 加载10秒的位置数据
        sat_range=(0, num_sats),  # 只使用前5个卫星
        sc_range=(0, num_scs)  # 只使用前100个小区
    )
    
    # 3. 记录结果
    all_sinr_results = []  # 存储所有时隙的SINR结果
    detail_writer = None
    # 4. 开始仿真
    for snapshot_idx in range(num_snapshots):
        print(f"\n处理Snapshot {snapshot_idx + 1}/{num_snapshots}")
        snapshot_sim = SimulateSnapshot(static_data, snapshot_idx)
        for slot_idx in range(slots_per_snapshot):
            global_slot_idx = snapshot_idx * slots_per_snapshot + slot_idx
            print(f"  处理Timeslot {slot_idx + 1}/{slots_per_snapshot} (全局时隙 {global_slot_idx + 1})")
            timeslot_sim = SimulateTimeslot(
                static_data=static_data,
                association=snapshot_sim.association,
                interference_association=snapshot_sim.interference_association,
                system_params=static_data.system_params,
                timeslot_idx=global_slot_idx,
                snapshot_sat_positions=static_data.sat_positions[:, snapshot_idx, :]
            )
            # 计算SINR并收集详细过程
            sinr_results = []
            if detail_writer is None:
                detail_writer = SimulationResultWriter(
                    num_users=static_data.user_positions.shape[0],
                    num_timeslots=num_snapshots * slots_per_snapshot
                )
            beam_schedule_matrix = np.zeros((timeslot_sim.num_sats, timeslot_sim.static_data.sc_centers.shape[0]), dtype=bool)
            for sat_idx in range(timeslot_sim.num_sats):
                beam_schedule_matrix[sat_idx, timeslot_sim.beam_schedule[sat_idx][timeslot_sim.beam_schedule[sat_idx] >= 0]] = True
            sat_positions = timeslot_sim.snapshot_sat_positions
            noise = timeslot_sim.compute_noise()
            for user_idx in range(timeslot_sim.num_users):
                user_sc = timeslot_sim.static_data.user_sc_mapping[user_idx]
                serving_sat = np.where(timeslot_sim.interference_association[user_idx] == 2)[0][0]
                if not beam_schedule_matrix[serving_sat, user_sc]:
                    sinr_results.append(float('-inf'))
                    detail_writer.record_detail(global_slot_idx, user_idx, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'))
                    continue
                rx_power_db = timeslot_sim.compute_received_power(
                    sat_positions[serving_sat],
                    timeslot_sim.static_data.user_positions[user_idx],
                    timeslot_sim.static_data.sc_centers[user_sc]
                )
                rx_power = 10 ** (rx_power_db / 10)
                # 星内干扰
                intra_interference = 0
                active_sc_indices = np.where(beam_schedule_matrix[serving_sat])[0]
                active_sc_indices = active_sc_indices[active_sc_indices != user_sc]
                if len(active_sc_indices) > 0:
                    for sc_idx in active_sc_indices:
                        interference_power_db = timeslot_sim.compute_received_power(
                            sat_positions[serving_sat],
                            timeslot_sim.static_data.user_positions[user_idx],
                            timeslot_sim.static_data.sc_centers[sc_idx]
                        )
                        interference_power = 10 ** (interference_power_db / 10)
                        intra_interference += interference_power
                intra_db = 10 * np.log10(intra_interference) if intra_interference > 0 else float('-inf')
                # 星间干扰
                inter_interference = 0
                interfering_sats = np.where(timeslot_sim.interference_association[user_idx] == 1)[0]
                if len(interfering_sats) > 0:
                    for sat_idx2 in interfering_sats:
                        active_sc_indices2 = np.where(beam_schedule_matrix[sat_idx2])[0]
                        if len(active_sc_indices2) > 0:
                            for sc_idx2 in active_sc_indices2:
                                interference_power_db = timeslot_sim.compute_received_power(
                                    sat_positions[sat_idx2],
                                    timeslot_sim.static_data.user_positions[user_idx],
                                    timeslot_sim.static_data.sc_centers[sc_idx2]
                                )
                                interference_power = 10 ** (interference_power_db / 10)
                                inter_interference += interference_power
                inter_db = 10 * np.log10(inter_interference) if inter_interference > 0 else float('-inf')
                noise_db = 10 * np.log10(noise) if noise > 0 else float('-inf')
                total_interference = intra_interference + inter_interference
                denominator = total_interference + noise
                if denominator > 0:
                    sinr = rx_power / denominator
                    sinr_db = 10 * np.log10(sinr)
                else:
                    sinr_db = float('-inf')
                sinr_results.append(sinr_db)
                detail_writer.record_detail(global_slot_idx, user_idx, rx_power_db, intra_db, inter_db, noise_db, sinr_db)
            all_sinr_results.append(sinr_results)
            detail_writer.record_sinr(sinr_results)
            print(f"    平均SINR: {np.mean([v for v in sinr_results if v != float('-inf')]):.2f} dB")
            print(f"    最大SINR: {np.max([v for v in sinr_results if v != float('-inf')]):.2f} dB")
            print(f"    最小SINR: {np.min([v for v in sinr_results if v != float('-inf')]):.2f} dB")
    # 5. 处理并保存结果
    all_sinr_results = np.array(all_sinr_results)  # [num_timeslots, num_users]
    mean_sinr = np.mean(all_sinr_results)
    max_sinr = np.max(all_sinr_results)
    min_sinr = np.min(all_sinr_results)
    print("\n仿真完成！")
    print(f"整体统计信息：")
    print(f"- 平均SINR: {mean_sinr:.2f} dB")
    print(f"- 最大SINR: {max_sinr:.2f} dB")
    print(f"- 最小SINR: {min_sinr:.2f} dB")
    # 保存表格
    if detail_writer is not None:
        detail_writer.save()
    # 仍可保存npy以兼容旧流程
    np.save('simulation_results/sinr_results.npy', all_sinr_results)


if __name__ == '__main__':
    main()