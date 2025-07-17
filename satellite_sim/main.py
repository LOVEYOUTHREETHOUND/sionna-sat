import numpy as np
import json
import os

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
        params = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().split('#')[0].strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    params[key] = value
        return params

    def _load_satellite_positions(self, sat_path, num_sats=1800, num_timesteps=None, sat_range=None):
        """加载卫星位置数据
        Args:
            sat_path: 卫星位置数据文件路径
            num_sats: 卫星总数
            num_timesteps: 要加载的时隙数，None表示全部加载
            sat_range: 要加载的卫星范围(start, end)，None表示加载所有卫星
        """
        # 确定要加载的卫星范围
        start_sat = 0 if sat_range is None else sat_range[0]
        end_sat = num_sats if sat_range is None else sat_range[1]
        num_sats_to_load = end_sat - start_sat

        all_sat_pos = []
        with open(sat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        idx = 0
        for sat_idx in range(num_sats):
            # 跳过不需要的卫星数据
            if sat_idx < start_sat:
                # 计算每个卫星的数据行数（2行表头 + 时隙数据行）
                if idx < len(lines):
                    while idx < len(lines) and not (lines[idx].startswith('-') or lines[idx].startswith('Time')):
                        idx += 1
                    idx += 2  # 跳过下一个卫星的表头
                continue
            
            if sat_idx >= end_sat:
                break

            # 跳过2行表头
            idx += 2
            pos = []
            count = 0
            while idx < len(lines):
                line = lines[idx].strip()
                if not line or line.startswith('-') or line.startswith('Time'):
                    break  # 到下一个块的表头
                parts = line.split()
                if len(parts) < 4:
                    idx += 1
                    continue
                x, y, z = map(float, parts[1:4])
                pos.append([x, y, z])
                idx += 1
                count += 1
                if num_timesteps is not None and count >= num_timesteps:
                    break
            all_sat_pos.append(np.array(pos))

        return np.stack(all_sat_pos, axis=0) if all_sat_pos else np.zeros((0,0,0))

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

        # 2. 计算每个用户的干扰卫星（除服务卫星外最近的18个）
        self.interference_association = np.zeros((self.num_users, self.num_sats), dtype=int)
        for user_idx in range(self.num_users):
            # 获取用户所在SC的索引（从static_data中获取）
            user_sc_idx = self.static_data.user_sc_mapping[user_idx]
            # 获取服务卫星索引
            serving_sat_idx = closest_sat_idx[user_sc_idx]
            # 获取到所有卫星的距离
            user_to_sats = np.linalg.norm(
                static_data.user_positions[user_idx][None, :] - sat_pos,
                axis=1
            )
            # 将服务卫星的距离设为无穷大，这样就不会被选为干扰卫星
            user_to_sats[serving_sat_idx] = np.inf
            # 选择最近的18个卫星作为干扰卫星
            interfering_sats = np.argpartition(user_to_sats, 18)[:18]
            self.interference_association[user_idx, interfering_sats] = 1

        # 预计算噪声功率
        self.precompute_noise()

    def precompute_noise(self):
        # 1. 计算等效噪声温度 Teq
        # G/T = G_R - 10*log10(Teq)  =>  Teq = 10**((G_R - G/T)/10)
        g_over_t = float(self.system_params['g_over_t'])  # dB/K
        receiver_gain = float(self.system_params['receiver_gain'])  # dB
        boltzmann_constant = float(self.system_params['boltzmann_constant'])  # J/K
        bandwidth = float(self.system_params['bandwidth'])  # Hz
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
    def __init__(self, static_data: StaticData, association: np.ndarray, system_params: dict, timeslot_idx: int):
        self.static_data = static_data
        self.association = association  # [num_sc, num_sats]
        self.system_params = system_params
        self.timeslot_idx = timeslot_idx
        self.num_sats = association.shape[1]
        self.num_sc = association.shape[0]
        self.beams_per_sat = int(system_params.get('beams_per_sat', 1))
        self.beam_schedule = self._round_robin_beam_schedule()

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

    def compute_received_power(self, sat_pos, user_pos, sc_center_pos):
        """计算接收功率
        Args:
            sat_pos: 卫星ECEF坐标 [3,]
            user_pos: 用户ECEF坐标 [3,]
            sc_center_pos: SC中心ECEF坐标 [3,]
        Returns:
            接收功率 (dBW)
        """
        # 1. 获取预计算的EIRP
        EIRP = self.static_data.system_params['EIRP']  # dBW
        
        # 2. 计算离轴角
        # 计算卫星到SC中心和用户的向量
        D_SC = sc_center_pos - sat_pos  # [3,]
        D_UE = user_pos - sat_pos  # [3,]
        # 计算离轴角(弧度)
        cos_alpha = np.dot(D_SC, D_UE) / (np.linalg.norm(D_SC) * np.linalg.norm(D_UE))
        alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))  # 防止数值误差导致domain error
        
        # 3. 计算增益衰减
        ka = self.static_data.system_params['ka']
        if alpha == 0:
            delta_G = 0  # dB
        else:
            x = ka * np.sin(alpha)
            # 使用scipy.special.j1计算第一类贝塞尔函数
            from scipy.special import j1
            delta_G = 10 * np.log10(4 * (j1(x)/x)**2)  # dB
        
        # 4. 接收天线增益
        G_R = float(self.static_data.system_params['receiver_gain'])  # dB
        
        # 5. 计算路径损耗
        d = np.linalg.norm(D_UE)  # 距离(m)
        f0 = float(self.static_data.system_params['carrier_freq']) / 1e9  # GHz
        L_s = float(self.static_data.system_params['scintillation_loss'])  # dB
        PL = 32.45 + 20 * np.log10(f0 * d) + L_s  # dB
        
        # 6. 计算总接收功率
        P_rx = EIRP + delta_G + G_R - PL  # dBW
        
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
        # 获取当前时隙的卫星位置
        sat_pos = self.static_data.sat_positions[:, self.timeslot_idx, :]
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
        sat_positions = self.static_data.sat_positions[:, self.timeslot_idx]
        
        # 2. 创建干扰功率数组 [num_users]
        total_interference = np.zeros(self.num_users)
        
        # 3. 创建活跃波束矩阵 [num_sats, num_sc]
        active_beams = np.zeros((self.num_sats, self.static_data.sc_centers.shape[0]), dtype=bool)
        for sat_idx in range(self.num_sats):
            active_beams[sat_idx, self.beam_schedule[sat_idx][self.beam_schedule[sat_idx] >= 0]] = True
        
        # 4. 批量计算每个用户的干扰
        for user_idx in range(self.num_users):
            user_pos = self.static_data.user_positions[user_idx]  # [3,]
            # 获取干扰卫星
            interfering_sats = np.where(self.interference_association[user_idx] == 1)[0]
            
            # 对每个干扰卫星的活跃波束计算干扰
            for sat_idx in interfering_sats:
                # 获取该卫星当前活跃的SC
                active_sc_indices = np.where(active_beams[sat_idx])[0]
                if len(active_sc_indices) > 0:
                    # 批量计算该卫星所有活跃波束到用户的接收功率
                    interference_powers_db = np.zeros(len(active_sc_indices))
                    for i, sc_idx in enumerate(active_sc_indices):
                        interference_powers_db[i] = self.compute_received_power(
                            sat_positions[sat_idx],
                            user_pos,
                            self.static_data.sc_centers[sc_idx]
                        )
                    # 转换为线性单位并累加
                    total_interference[user_idx] += np.sum(10 ** (interference_powers_db / 10))
        
        return total_interference

    def compute_noise(self):
        # 直接返回预计算的噪声值
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
        
        # 2. 获取所有用户的服务卫星
        user_sc_indices = self.static_data.user_sc_mapping  # [num_users]
        serving_sat_indices = np.where(self.association[user_sc_indices] == 1)[1]  # [num_users]
        
        # 3. 批量计算接收功率
        rx_powers = np.zeros(self.num_users)  # [num_users]
        # 获取当前时隙的卫星位置
        sat_positions = self.static_data.sat_positions[:, self.timeslot_idx]  # [num_sats, 3]
        
        # 只为被调度的用户计算接收功率
        scheduled_mask = beam_schedule_matrix[serving_sat_indices, user_sc_indices]  # [num_users]
        scheduled_users = np.where(scheduled_mask)[0]
        
        if len(scheduled_users) > 0:
            # 计算这些用户的接收功率
            for user_idx in scheduled_users:
                rx_power_db = self.compute_received_power(
                    sat_positions[serving_sat_indices[user_idx]],
                    self.static_data.user_positions[user_idx],
                    self.static_data.sc_centers[user_sc_indices[user_idx]]
                )
                rx_powers[user_idx] = 10 ** (rx_power_db / 10)  # 转换为线性单位
        
        # 4. 批量计算干扰
        interference = self.compute_interference_vectorized()  # [num_users]
        
        # 5. 获取噪声功率
        noise = self.compute_noise()  # 标量
        
        # 6. 计算SINR
        denominator = interference + noise
        valid_mask = denominator > 0
        sinr_all = np.full(self.num_users, float('-inf'))
        
        if np.any(valid_mask):
            sinr_all[valid_mask] = 10 * np.log10(rx_powers[valid_mask] / denominator[valid_mask])
        
        return sinr_all


def main():
    """运行基本仿真实验"""
    # 1. 配置仿真参数
    system_param_path = 'satellite_sim/config/system_params.txt'
    satellite_result_path = 'data/satellite_positions.txt'
    topology_json_dir = 'data/topology_result/json'
    
    # 仿真范围配置
    num_sats = 5  # 前5个卫星
    num_scs = 100  # A1区域前100个小区
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
    
    # 4. 开始仿真
    for snapshot_idx in range(num_snapshots):
        print(f"\n处理Snapshot {snapshot_idx + 1}/{num_snapshots}")
        
        # 初始化snapshot仿真器
        snapshot_sim = SimulateSnapshot(static_data, snapshot_idx)
        
        # 对每个timeslot进行仿真
        for slot_idx in range(slots_per_snapshot):
            # 计算当前时隙的全局索引
            global_slot_idx = snapshot_idx * slots_per_snapshot + slot_idx
            print(f"  处理Timeslot {slot_idx + 1}/{slots_per_snapshot} (全局时隙 {global_slot_idx + 1})")
            
            # 初始化timeslot仿真器
            timeslot_sim = SimulateTimeslot(
                static_data=static_data,
                association=snapshot_sim.association,
                system_params=static_data.system_params,
                timeslot_idx=global_slot_idx
            )
            
            # 计算SINR
            sinr_results = timeslot_sim.compute_sinr()
            all_sinr_results.append(sinr_results)
            
            # 输出当前时隙的统计信息
            print(f"    平均SINR: {np.mean(sinr_results):.2f} dB")
            print(f"    最大SINR: {np.max(sinr_results):.2f} dB")
            print(f"    最小SINR: {np.max(sinr_results):.2f} dB")
    
    # 5. 处理并保存结果
    all_sinr_results = np.array(all_sinr_results)  # [num_timeslots, num_users]
    
    # 计算整体统计信息
    mean_sinr = np.mean(all_sinr_results)
    max_sinr = np.max(all_sinr_results)
    min_sinr = np.min(all_sinr_results)
    
    print("\n仿真完成！")
    print(f"整体统计信息：")
    print(f"- 平均SINR: {mean_sinr:.2f} dB")
    print(f"- 最大SINR: {max_sinr:.2f} dB")
    print(f"- 最小SINR: {min_sinr:.2f} dB")
    
    # 可以根据需要保存结果到文件
    np.save('simulation_results/sinr_results.npy', all_sinr_results)


if __name__ == '__main__':
    main()