import numpy as np
import pandas as pd
import json
import os

def load_system_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    all_params = {}
    all_params.update(params['basic_params'])
    all_params.update(params['physical_constants'])
    all_params['wavelength'] = float(all_params['speed_of_light']) / float(all_params['carrier_freq'])
    all_params['antenna_diameter'] = 1.5
    all_params['ka'] = 2 * np.pi / all_params['wavelength'] * all_params['antenna_diameter'] / 2
    all_params['EIRP'] = float(all_params['eirp_density']) + 10 * np.log10(float(all_params['bandwidth'])/1e6)
    return all_params

def load_satellite_positions(csv_path, time_idx):
    df = pd.read_csv(csv_path)
    sat_names = df['satellite'].unique()
    sat_positions = []
    for sat in sat_names:
        sat_df = df[df['satellite'] == sat].iloc[time_idx]
        sat_positions.append([sat_df['x'], sat_df['y'], sat_df['z']])
    return np.array(sat_positions), sat_names

def load_target_cell_users(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sc_center = np.array([data['target_cell_ecef']['x'], data['target_cell_ecef']['y'], data['target_cell_ecef']['z']])
    users = data['users']
    user_positions = np.array([[u['ecef']['x'], u['ecef']['y'], u['ecef']['z']] for u in users])
    user_ids = [u['user_id'] for u in users]
    return sc_center, user_positions, user_ids

def compute_received_power(sat_pos, user_pos, sc_center_pos, params):
    EIRP = params['EIRP']
    D_SC = sc_center_pos - sat_pos
    D_UE = user_pos - sat_pos
    cos_alpha = np.dot(D_SC, D_UE) / (np.linalg.norm(D_SC) * np.linalg.norm(D_UE))
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    ka = params['ka']
    if alpha == 0:
        delta_G = 0
    else:
        x = ka * np.sin(alpha)
        from scipy.special import j1
        delta_G = 10 * np.log10(4 * (j1(x)/x)**2) if x != 0 else 0
    G_R = float(params['receiver_gain'])
    d = np.linalg.norm(D_UE)
    f0 = float(params['carrier_freq']) / 1e9
    L_s = float(params['scintillation_loss'])
    F_s = np.random.normal(0, 2.0)
    PL = 32.45 + 20 * np.log10(f0 * d) + F_s + L_s
    P_rx = EIRP + delta_G + G_R - PL
    return P_rx

def compute_noise(params):
    g_over_t = float(params['g_over_t'])
    receiver_gain = float(params['receiver_gain'])
    boltzmann_constant = float(params['boltzmann_constant'])
    bandwidth = float(params['bandwidth'])
    Teq = 10 ** ((receiver_gain - g_over_t) / 10)
    N0 = boltzmann_constant * Teq * bandwidth
    return N0

def is_satellite_visible(sat_pos, user_pos, min_elev_deg=10):
    earth_radius = 6371e3
    user_height = np.linalg.norm(user_pos) - earth_radius
    sat_range = np.linalg.norm(sat_pos - user_pos)
    cos_elevation = (np.linalg.norm(sat_pos) ** 2 + sat_range ** 2 - (earth_radius + user_height) ** 2) / (2 * np.linalg.norm(sat_pos) * sat_range)
    elevation = np.arccos(np.clip(cos_elevation, -1.0, 1.0))
    min_elevation = np.deg2rad(min_elev_deg)
    return elevation >= min_elevation

def round_robin_bandwidth_allocation(slot_idx, user_idx, num_slots=600, num_freq_grids=150, num_users=10, bandwidth=30e6):
    freq_grid_bandwidth = bandwidth / num_freq_grids
    assigned_freq_grids = [f for f in range(num_freq_grids) if (slot_idx * num_freq_grids + f) % num_users == user_idx]
    user_bandwidth = len(assigned_freq_grids) * freq_grid_bandwidth
    return user_bandwidth

def main():
    # 配置
    system_param_path = 'config/system_params.json'
    sat_csv_path = 'results/excel/satellite_positions_10min.csv'
    target_cell_json = 'results/json/target_cell_users.json'
    output_xlsx = f'results/excel/user_snr_timeslot{time_idx}.xlsx'
    time_idx = 500  # 可修改为任意时隙编号

    # 1. 加载参数和数据
    params = load_system_params(system_param_path)
    sc_center, user_positions, user_ids = load_target_cell_users(target_cell_json)
    sat_positions, sat_names = load_satellite_positions(sat_csv_path, time_idx)
    bandwidth = float(params['bandwidth'])
    num_slots = 600
    num_freq_grids = 150
    num_users = len(user_ids)

    alpha_oh = 0.2  # 开销比例
    # 2. 计算每个用户的SNR及详细中间量
    snr_list = []
    noise = compute_noise(params)
    for i, user_pos in enumerate(user_positions):
        dists = np.linalg.norm(sat_positions - user_pos, axis=1)
        serving_idx = np.argmin(dists)
        serving_sat_pos = sat_positions[serving_idx]
        serving_sat_name = sat_names[serving_idx]
        # 计算接收功率及中间量
        EIRP = params['EIRP']
        D_SC = sc_center - serving_sat_pos
        D_UE = user_pos - serving_sat_pos
        cos_alpha = np.dot(D_SC, D_UE) / (np.linalg.norm(D_SC) * np.linalg.norm(D_UE))
        alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
        alpha_deg = np.degrees(alpha)
        ka = params['ka']
        if alpha == 0:
            delta_G = 0
        else:
            x = ka * np.sin(alpha)
            from scipy.special import j1
            delta_G = 10 * np.log10(4 * (j1(x)/x)**2) if x != 0 else 0
        G_R = float(params['receiver_gain'])
        d = np.linalg.norm(D_UE)
        f0 = float(params['carrier_freq']) / 1e9
        L_s = float(params['scintillation_loss'])
        F_s = np.random.normal(0, 2.0)
        PL = 32.45 + 20 * np.log10(f0 * d) + F_s + L_s
        rx_power_db = EIRP + delta_G + G_R - PL
        rx_power = 10 ** (rx_power_db / 10)
        denominator = noise
        snr = rx_power / denominator if denominator > 0 else 0
        snr_db = 10 * np.log10(snr) if snr > 0 else -np.inf
        # 频谱效率
        spectral_efficiency = 0.15 * np.log2(1 + snr) if snr > 0 else 0
        # Round-robin带宽分配
        user_bandwidth = round_robin_bandwidth_allocation(time_idx, i, num_slots=num_slots, num_freq_grids=num_freq_grids, num_users=num_users, bandwidth=bandwidth)
        # 有效带宽修正
        beff = user_bandwidth * (1 - alpha_oh)
        # 实际吞吐量（bps）
        throughput = spectral_efficiency * beff
        snr_list.append({
            'user_id': user_ids[i],
            'serving_satellite': serving_sat_name,
            'noise': noise,
            'path_loss': PL,
            'off_axis_angle_deg': alpha_deg,
            'beam_gain_db': delta_G,
            'rx_power_dbw': rx_power_db,
            'snr_db': snr_db,
            'spectral_efficiency': spectral_efficiency,
            'allocated_bandwidth_hz': user_bandwidth,
            'effective_bandwidth_hz': beff,
            'throughput_bps': throughput
        })

    # 3. 输出为xlsx
    df = pd.DataFrame(snr_list)
    df.to_excel(output_xlsx, index=False)
    print(f'SNR结果已保存到: {output_xlsx}')

if __name__ == '__main__':
    main()
