"""卫星系统级仿真器

结合Sionna系统级仿真框架和卫星信道模型
"""

import sionna
import sionna.sys
import sionna.phy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

from config.system_params import (
    SATELLITE_PARAMS,
    CELL_PARAMS,
    CHANNEL_PARAMS,
    SIM_PARAMS
)

# 定义目标区域 (保留用于参考)
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

class ChannelMatrix(sionna.phy.Block):
    """MIMO信道矩阵生成和管理类
    
    该类负责生成和更新MIMO系统的信道矩阵
    """
    def __init__(self,
                 resource_grid,
                 batch_size,
                 num_rx,
                 num_tx,
                 num_rx_ant,
                 num_tx_ant,
                 coherence_time,
                 precision=None):
        super().__init__(precision=precision)
        self.resource_grid = resource_grid
        self.coherence_time = coherence_time
        self.batch_size = batch_size
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        
        # 初始化衰落相关系数
        self.rho_fading = sionna.phy.config.tf_rng.uniform(
            [batch_size, num_rx, num_tx], 
            minval=.95, 
            maxval=.99, 
            dtype=self.rdtype
        )
        # 初始化衰落系数为1
        self.fading = tf.ones([batch_size, num_rx, num_tx], dtype=self.rdtype)

    def call(self, channel_model):
        """生成OFDM信道响应"""
        ofdm_channel = sionna.phy.channel.GenerateOFDMChannel(channel_model, self.resource_grid)
        h_freq = ofdm_channel(self.batch_size)
        return h_freq

    def update(self, channel_model, h_freq, slot):
        """根据相干时间更新信道响应"""
        h_freq_new = self.call(channel_model)
        change = tf.cast(tf.math.mod(slot, self.coherence_time) == 0, self.cdtype)
        h_freq = change * h_freq_new + (tf.cast(1, self.cdtype) - change) * h_freq
        return h_freq

    def apply_fading(self, h_freq):
        """应用时变衰落效应"""
        # 更新衰落系数,使用AR-1过程
        self.fading = tf.cast(1, self.rdtype) - self.rho_fading + \
            self.rho_fading * self.fading + \
            sionna.phy.config.tf_rng.uniform(
                self.fading.shape, 
                minval=-.1, 
                maxval=.1, 
                dtype=self.rdtype
            )
        # 确保衰落系数非负
        self.fading = tf.maximum(self.fading, tf.cast(0, self.rdtype))
        
        # 扩展维度
        fading_expand = sionna.phy.utils.insert_dims(self.fading, 1, axis=2)
        fading_expand = sionna.phy.utils.insert_dims(fading_expand, 3, axis=4)
        
        # 应用衰落到信道响应
        h_freq_fading = tf.cast(tf.math.sqrt(fading_expand), self.cdtype) * h_freq
        return h_freq_fading


def get_stream_management(direction,
                          num_rx,
                          num_tx,
                          num_streams_per_ut,
                          num_ut_per_sector):
    """创建流管理对象"""
    if direction == "downlink":
        num_streams_per_tx = num_streams_per_ut * num_ut_per_sector
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in np.arange(i2 * num_ut_per_sector, (i2 + 1) * num_ut_per_sector)])    
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1
    else:
        num_streams_per_tx = num_streams_per_ut
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in np.arange(i1 * num_ut_per_sector, (i1 + 1) * num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1
    
    stream_management = sionna.phy.mimo.StreamManagement(
        rx_tx_association, num_streams_per_tx
    )
    return stream_management


def get_sinr(tx_power,
             stream_management,
             no,
             direction,
             h_freq_fading,
             num_bs,
             num_ut_per_sector,
             num_streams_per_ut,
             resource_grid):
    """计算SINR"""
    # 重塑发射功率维度
    s = tx_power.shape
    tx_power = tf.reshape(tx_power, [s[0], s[1] * s[2]] + s[3:])
    
    # 选择预编码方案
    if direction == "downlink":
        precoded_channel = sionna.phy.ofdm.RZFPrecodedChannel(
            resource_grid=resource_grid,
            stream_management=stream_management
        )
        h_eff = precoded_channel(h_freq_fading, tx_power=tx_power, alpha=no)
    else:
        precoded_channel = sionna.phy.ofdm.EyePrecodedChannel(
            resource_grid=resource_grid,
            stream_management=stream_management
        )
        h_eff = precoded_channel(h_freq_fading, tx_power=tx_power)
    
    # 计算SINR
    lmmse_posteq_sinr = sionna.phy.ofdm.LMMSEPostEqualizationSINR(
        resource_grid=resource_grid,
        stream_management=stream_management
    )
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
    
    # 重塑SINR维度
    sinr = tf.reshape(
        sinr, sinr.shape[:-2] + [num_bs * num_ut_per_sector, num_streams_per_ut]
    )
    sinr = tf.reshape(
        sinr, sinr.shape[:-2] + [num_bs, num_ut_per_sector, num_streams_per_ut]
    )
    sinr = tf.transpose(sinr, [0, 3, 1, 2, 4, 5])
    
    return sinr


def estimate_achievable_rate(sinr_eff_db_last,
                             num_ofdm_sym,
                             num_subcarriers):
    """估计可达速率"""
    rate_achievable_est = sionna.phy.utils.log2(
        tf.cast(1, sinr_eff_db_last.dtype) + 
        sionna.phy.utils.db_to_lin(sinr_eff_db_last)
    )
    rate_achievable_est = sionna.phy.utils.insert_dims(
        rate_achievable_est, 2, axis=-2
    )
    rate_achievable_est = tf.tile(
        rate_achievable_est, 
        [1, 1, num_ofdm_sym, num_subcarriers, 1]
    )
    return rate_achievable_est


def init_result_history(batch_size,
                        num_slots,
                        num_bs,
                        num_ut_per_sector):
    """初始化仿真结果历史记录"""
    hist = {}
    for key in ['pathloss_serving_cell',
                'tx_power', 'olla_offset',
                'sinr_eff', 'pf_metric',
                'num_decoded_bits', 'mcs_index',
                'harq', 'num_allocated_re']:
        hist[key] = tf.TensorArray(
            size=num_slots,
            element_shape=[batch_size, num_bs, num_ut_per_sector],
            dtype=tf.float32
        )
    return hist


def record_results(hist,
                   slot,
                   sim_failed=False,
                   pathloss_serving_cell=None,
                   num_allocated_re=None,
                   tx_power_per_ut=None,
                   num_decoded_bits=None,
                   mcs_index=None,
                   harq_feedback=None,
                   olla_offset=None,
                   sinr_eff=None,
                   pf_metric=None,
                   shape=None):
    """记录仿真结果"""
    if not sim_failed:
        for key, value in zip(['pathloss_serving_cell', 'olla_offset', 'sinr_eff',
                               'num_allocated_re', 'tx_power', 'num_decoded_bits',
                               'mcs_index', 'harq'],
                               [pathloss_serving_cell, olla_offset, sinr_eff,
                                num_allocated_re, tx_power_per_ut, num_decoded_bits,
                                mcs_index, harq_feedback]):
            hist[key] = hist[key].write(slot, tf.cast(value, tf.float32))
        hist['pf_metric'] = hist['pf_metric'].write(
            slot, tf.reduce_mean(pf_metric, axis=[-2, -3])
        )
    else:
        nan_tensor = tf.cast(tf.fill(shape, float('nan')), dtype=tf.float32)
        for key in hist:
            hist[key] = hist[key].write(slot, nan_tensor)
    return hist


def clean_hist(hist, batch=0):
    """清理历史记录"""
    for key in hist:
        try:
            hist[key] = hist[key].numpy()[:, batch, :, :]
        except:
            pass

    hist['mcs_index'] = np.where(
        hist['harq'] == -1, np.nan, hist['mcs_index']
    )
    hist['sinr_eff'] = np.where(
        hist['harq'] == -1, np.nan, hist['sinr_eff']
    )
    hist['tx_power'] = np.where(
        hist['harq'] == -1, np.nan, hist['tx_power']
    )
    hist['num_allocated_re'] = np.where(
        hist['harq'] == -1, 0, hist['num_allocated_re']
    )
    hist['harq'] = np.where(
        hist['harq'] == -1, np.nan, hist['harq']
    )
    return hist 


class SatelliteSystemSimulator(sionna.phy.Block):
    """卫星系统级仿真器"""
    def __init__(self,
                 batch_size,
                 resource_grid,
                 direction,
                 satellite_params,
                 cell_params,
                 channel_params,
                 target_areas,
                 coherence_time=100,
                 precision=None):
        super().__init__(precision=precision)
        
        self.batch_size = batch_size
        self.resource_grid = resource_grid
        self.direction = direction
        self.satellite_params = satellite_params
        self.cell_params = cell_params
        self.channel_params = channel_params
        self.target_areas = target_areas
        self.coherence_time = coherence_time
        
        # 设置系统参数 (使用标准sionna组件)
        # 创建标准拓扑
        num_rings = 1  # 单个环形小区布局
        self.num_bs = sionna.sys.get_num_hex_in_grid(num_rings) * 3
        self.num_ut_per_sector = cell_params['num_users'] // self.num_bs
        self.num_ut = self.num_bs * self.num_ut_per_sector
        
        # 创建天线阵列
        bs_array = sionna.phy.channel.tr38901.PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=3,
            polarization='dual',
            polarization_type='VH',
            antenna_pattern='38.901',
            carrier_frequency=satellite_params['frequency']
        )
        
        ut_array = sionna.phy.channel.tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=satellite_params['frequency']
        )
        
        self.num_bs_ant = bs_array.num_ant
        if bs_array.polarization == 'dual':
            self.num_bs_ant *= 2
        self.num_ut_ant = ut_array.num_ant
        
        # 设置发射端和接收端参数
        if direction == 'downlink':
            self.num_tx = self.num_bs  # 发射端是卫星波束
            self.num_rx = self.num_ut  # 接收端是用户
            self.num_tx_ant = self.num_bs_ant  # 发射天线数
            self.num_rx_ant = self.num_ut_ant  # 用户接收天线数
            self.num_tx_per_sector = 1
        else:
            self.num_tx = self.num_ut  # 发射端是用户
            self.num_rx = self.num_bs  # 接收端是卫星波束
            self.num_tx_ant = self.num_ut_ant  # 用户发射天线数
            self.num_rx_ant = self.num_bs_ant  # 接收天线数
            self.num_tx_per_sector = self.num_ut_per_sector
        
        # 设置流管理
        self.num_streams_per_ut = resource_grid.num_streams_per_tx
        self.stream_management = get_stream_management(
            direction,
            self.num_rx,
            self.num_tx,
            self.num_streams_per_ut,
            self.num_ut_per_sector
        )
        
        # 设置噪声功率谱密度
        k = sionna.phy.constants.BOLTZMANN_CONSTANT  # 玻尔兹曼常数
        self.no = tf.cast(k * channel_params['temperature'] * 
                     resource_grid.subcarrier_spacing, self.rdtype)
        
        # 设置时隙持续时间
        self.slot_duration = resource_grid.ofdm_symbol_duration * \
            resource_grid.num_ofdm_symbols
            
        # 创建标准地面信道模型(用作基础)
        common_params = {
            'carrier_frequency': satellite_params['frequency'],
            'ut_array': ut_array,
            'bs_array': bs_array,
            'direction': self.direction,
            'enable_pathloss': True,
            'enable_shadow_fading': True,
            'precision': self.precision
        }
        
        # 使用UMi模型(可根据需要替换)
        self.channel_model = sionna.phy.channel.tr38901.UMi(o2i_model='low', **common_params)
        
        # 创建标准拓扑
        self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, \
            self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc, self.grid = \
            sionna.sys.gen_hexgrid_topology(
                batch_size=self.batch_size,
                num_rings=num_rings,
                num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=10,
                max_bs_ut_dist=80,
                scenario='umi',
                los=True,
                return_grid=True,
                precision=self.precision
            )
            
        self.channel_model.set_topology(
            self.ut_loc, self.bs_loc, self.ut_orientations,
            self.bs_orientations, self.ut_velocities,
            self.in_state, self.los, self.bs_virtual_loc
        )
        
        # 初始化信道矩阵生成器
        self.channel_matrix = ChannelMatrix(
            resource_grid=resource_grid,
            batch_size=batch_size,
            num_rx=self.num_rx,
            num_tx=self.num_tx,
            num_rx_ant=self.num_rx_ant,
            num_tx_ant=self.num_tx_ant,
            coherence_time=coherence_time,
            precision=precision
        )
        
        # 初始化物理层抽象
        self.phy_abs = sionna.sys.PHYAbstraction(precision=precision)
        
        # 初始化外环链路自适应
        self.olla = sionna.sys.OuterLoopLinkAdaptation(
            self.phy_abs,
            self.num_ut_per_sector,
            batch_size=[batch_size, self.num_bs]
        )
        
        # 初始化调度器
        self.scheduler = sionna.sys.PFSchedulerSUMIMO(
            self.num_ut_per_sector,
            resource_grid.fft_size,
            resource_grid.num_ofdm_symbols,
            batch_size=[batch_size, self.num_bs],
            num_streams_per_ut=self.num_streams_per_ut,
            precision=precision
        )
        
    def _reset(self, bler_target, olla_delta_up):
        """重置仿真状态"""
        self.olla.reset()
        self.olla.bler_target = bler_target
        self.olla.olla_delta_up = olla_delta_up

        last_harq_feedback = - tf.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=tf.int32
        )

        sinr_eff_feedback = tf.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=self.rdtype
        )

        num_decoded_bits = tf.zeros(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            tf.int32
        )
        return last_harq_feedback, sinr_eff_feedback, num_decoded_bits
    
    def _group_by_sector(self, tensor):
        """按扇区分组"""
        tensor = tf.reshape(tensor, [self.batch_size,
                                     self.num_bs,
                                     self.num_ut_per_sector,
                                     self.resource_grid.num_ofdm_symbols])

        return tf.transpose(tensor, [0, 1, 3, 2])
    
    @tf.function
    def call(self,
             num_slots,
             bler_target,
             olla_delta_up,
             mcs_table_index=1,
             fairness_dl=0,
             guaranteed_power_ratio_dl=0.5
    ):
        """运行仿真"""
        # 初始化历史记录
        hist = init_result_history(self.batch_size, num_slots, self.num_bs, self.num_ut_per_sector)
        
        # 重置仿真状态
        last_harq_feedback, sinr_eff_feedback, num_decoded_bits = \
            self._reset(bler_target, olla_delta_up)
        
        # 生成初始信道矩阵
        h_freq = self.channel_matrix(self.channel_model)
        
        # 仿真开始时间
        start_time = datetime.now()
        
        def simulate_slot(slot,
                          hist,
                          harq_feedback,
                          sinr_eff_feedback,
                          num_decoded_bits,
                          h_freq):
            """单个时隙仿真 - 这里实现卫星信道特性的自定义计算"""
            try:
                # 更新信道矩阵
                h_freq = self.channel_matrix.update(self.channel_model, h_freq, slot)
                
                # 计算当前时间
                current_time = start_time + timedelta(seconds=slot*self.slot_duration)
                
                # 自定义卫星场景的路径损耗计算
                
                # 获取用户位置
                user_positions = self.ut_loc  # [batch_size, num_ut, 3]
                
                # 获取卫星位置 (使用基站位置作为替代)
                satellite_position = self.bs_loc[:, 0, :]  # [batch_size, 3]
                
                # 计算用户与卫星之间的距离
                relative_positions = user_positions - tf.expand_dims(satellite_position, axis=1)
                distances = tf.norm(relative_positions, axis=2)  # [batch_size, num_ut]
                
                # 计算仰角 (elevation angle)
                user_unit_vectors = user_positions / tf.norm(user_positions, axis=2, keepdims=True)
                relative_unit_vectors = relative_positions / tf.expand_dims(distances, axis=2)
                cos_angles = tf.reduce_sum(user_unit_vectors * relative_unit_vectors, axis=2)
                elevation_angles = tf.acos(tf.clip_by_value(cos_angles, -1.0, 1.0)) - tf.constant(np.pi/2, dtype=self.rdtype)
                elevation_degrees = elevation_angles * 180.0 / np.pi
                
                # 1. 自由空间路径损耗 (Free Space Path Loss)
                wavelength = tf.constant(3e8 / self.satellite_params['frequency'], dtype=self.rdtype)
                fspl_db = 20.0 * tf.math.log(4.0 * np.pi * distances / wavelength) / tf.math.log(tf.constant(10.0, dtype=self.rdtype))
                
                # 2. 大气损耗 (Atmospheric Loss)
                atmospheric_loss_db = tf.where(
                    elevation_degrees > 0,
                    15.0 / (elevation_degrees + 10.0),  # 低仰角损耗较大
                    tf.ones_like(elevation_degrees) * 15.0  # 负仰角设为最大损耗
                )
                
                # 3. 雨衰 (Rain Attenuation)
                rain_rate = tf.constant(self.channel_params.get('rain_rate', 0.0), dtype=self.rdtype)
                frequency_ghz = self.satellite_params['frequency'] / 1e9
                rain_loss_db = tf.where(
                    elevation_degrees > 0,
                    rain_rate * frequency_ghz * (0.5 / tf.sin(elevation_angles + 1e-10)),
                    tf.zeros_like(elevation_degrees)
                )
                
                # 4. 云雾损耗 (Clouds and Fog Loss)
                clouds_loss_db = tf.where(
                    elevation_degrees > 0,
                    0.1 * frequency_ghz / tf.sin(elevation_angles + 1e-10),
                    tf.zeros_like(elevation_degrees)
                )
                
                # 5. 闪烁损耗 (Scintillation Loss)
                scintillation_loss_db = tf.where(
                    elevation_degrees > 0,
                    0.07 * frequency_ghz * tf.sqrt(1.0 / tf.sin(elevation_angles + 1e-10)),
                    tf.zeros_like(elevation_degrees)
                )
                
                # 6. 阴影衰落 (Shadow Fading)
                shadow_fading_std = tf.constant(8.0, dtype=self.rdtype)  # 标准差8dB
                shadow_fading_db = tf.random.normal(
                    tf.shape(distances),
                    mean=0.0,
                    stddev=shadow_fading_std,
                    dtype=self.rdtype
                )
                
                # 总路径损耗
                total_path_loss_db = fspl_db + atmospheric_loss_db + rain_loss_db + \
                                    clouds_loss_db + scintillation_loss_db + shadow_fading_db
                
                # 转换为线性尺度
                total_path_loss_linear = tf.pow(10.0, total_path_loss_db / 10.0)
                
                # 计算接收功率
                tx_power_w = tf.pow(10.0, self.satellite_params['tx_power'] / 10.0) / 1000.0  # dBW to W
                tx_antenna_gain_linear = tf.pow(10.0, self.satellite_params['antenna_gain'] / 10.0)
                rx_antenna_gain_linear = tf.pow(10.0, 0.0 / 10.0)  # 假设接收端天线增益为0dBi
                
                # 简化波束增益模型
                beam_width_rad = tf.constant(1.0, dtype=self.rdtype) * np.pi / 180.0  # 1度波束宽度
                
                # 使用用户到卫星的角度作为波束角度
                angles_to_satellite = tf.acos(tf.clip_by_value(cos_angles, -1.0, 1.0))
                
                # 简化波束增益模型
                beam_gain_reduction_db = tf.where(
                    angles_to_satellite <= beam_width_rad,
                    3.0 * tf.square(angles_to_satellite / beam_width_rad),  # 波束内损耗模型
                    tf.ones_like(angles_to_satellite) * 30.0  # 波束外损耗
                )
                
                # 应用波束增益衰减
                effective_tx_gain_linear = tx_antenna_gain_linear / tf.pow(10.0, beam_gain_reduction_db / 10.0)
                
                # 计算接收功率
                rx_power_w = tx_power_w * effective_tx_gain_linear * rx_antenna_gain_linear / total_path_loss_linear
                rx_power_dbm = 10.0 * tf.math.log(rx_power_w * 1000.0) / tf.math.log(tf.constant(10.0, dtype=self.rdtype))
                
                # 计算噪声功率
                k_boltzmann = tf.constant(1.38e-23, dtype=self.rdtype)  # 玻尔兹曼常数
                temperature_k = tf.constant(self.channel_params['temperature'], dtype=self.rdtype)  # 系统温度
                bandwidth_hz = tf.constant(self.channel_params['bandwidth'], dtype=self.rdtype)  # 带宽
                noise_figure_db = tf.constant(self.channel_params['noise_figure'], dtype=self.rdtype)  # 噪声系数
                
                noise_power_w = k_boltzmann * temperature_k * bandwidth_hz * tf.pow(10.0, noise_figure_db / 10.0)
                noise_power_dbm = 10.0 * tf.math.log(noise_power_w * 1000.0) / tf.math.log(tf.constant(10.0, dtype=self.rdtype))
                
                # 计算SNR
                snr_db = rx_power_dbm - noise_power_dbm
                
                # 将计算结果转换为符合Sionna期望的格式的信道矩阵
                channel_mag = tf.sqrt(tf.pow(10.0, snr_db / 10.0) * noise_power_w / tx_power_w)
                
                # 为每个用户创建一个随机相位
                random_phase = tf.random.uniform(
                    tf.shape(channel_mag),
                    minval=0,
                    maxval=2*np.pi,
                    dtype=self.rdtype
                )
                
                # 打印调试信息
                if slot == 0 or slot % 100 == 0:
                    print(f"\n时隙 {slot} 的SNR计算结果:")
                    print(f"  平均SNR: {tf.reduce_mean(snr_db):.2f} dB")
                    print(f"  最大SNR: {tf.reduce_max(snr_db):.2f} dB")
                    print(f"  最小SNR: {tf.reduce_min(snr_db):.2f} dB")
                    print(f"  路径损耗组成:")
                    print(f"    - 自由空间损耗: {tf.reduce_mean(fspl_db):.2f} dB")
                    print(f"    - 大气损耗: {tf.reduce_mean(atmospheric_loss_db):.2f} dB")
                    print(f"    - 雨衰: {tf.reduce_mean(rain_loss_db):.2f} dB")
                    print(f"    - 云雾损耗: {tf.reduce_mean(clouds_loss_db):.2f} dB")
                    print(f"    - 闪烁损耗: {tf.reduce_mean(scintillation_loss_db):.2f} dB")
                    print(f"    - 阴影衰落: {tf.reduce_mean(shadow_fading_db):.2f} dB")
                
                # 创建复数信道系数
                channel_coef = tf.complex(
                    channel_mag * tf.cos(random_phase),
                    channel_mag * tf.sin(random_phase)
                )
                
                # 应用到现有信道矩阵 (这需要针对您的具体场景进行适配)
                # 为简化，我们使用原有信道矩阵并应用额外的衰落效应
                h_freq_fading = self.channel_matrix.apply_fading(h_freq)
                
                # 估计可达速率
                rate_achievable_est = estimate_achievable_rate(
                    self.olla.sinr_eff_db_last,
                    self.resource_grid.num_ofdm_symbols,
                    self.resource_grid.fft_size
                )
                
                # 调度用户
                is_scheduled = self.scheduler(
                    num_decoded_bits,
                    rate_achievable_est
                )
                
                # 计算分配的子载波数
                num_allocated_sc = tf.minimum(tf.reduce_sum(
                    tf.cast(is_scheduled, tf.int32), axis=-1), 1
                )
                num_allocated_sc = tf.reduce_sum(
                    num_allocated_sc, axis=-2
                )
                
                # 计算分配的资源元素数
                num_allocated_re = \
                    tf.reduce_sum(tf.cast(is_scheduled, tf.int32),
                                  axis=[-1, -3, -4])
                
                # 计算路径损耗
                pathloss_all_pairs, pathloss_serving_cell = sionna.sys.get_pathloss(
                    h_freq_fading,
                    rx_tx_association=tf.convert_to_tensor(
                        self.stream_management.rx_tx_association
                    )
                )
                
                pathloss_serving_cell = self._group_by_sector(
                    pathloss_serving_cell
                )
                
                # 功率控制
                if self.direction == 'uplink':
                    # 上行开环功率控制
                    tx_power_per_ut = sionna.sys.open_loop_uplink_power_control(
                        pathloss_serving_cell,
                        num_allocated_sc,
                        alpha=1.0,  # 路径损耗补偿因子
                        p0_dbm=-80.0,  # 目标接收功率
                        ut_max_power_dbm=23.0  # 用户最大发射功率(dBm)
                    )
                else:
                    # 下行公平功率控制
                    one = tf.cast(1, pathloss_serving_cell.dtype)
                    rx_power_tot = tf.reduce_sum(
                        one / pathloss_all_pairs, axis=-2
                    )
                    rx_power_tot = self._group_by_sector(rx_power_tot)
                    interference_dl = rx_power_tot - one / pathloss_serving_cell
                    interference_dl *= sionna.phy.utils.dbm_to_watt(self.satellite_params['tx_power'] + 30)  # 转换为dBm

                    tx_power_per_ut, _ = sionna.sys.downlink_fair_power_control(
                        pathloss_serving_cell,
                        interference_dl + self.no,
                        num_allocated_sc,
                        bs_max_power_dbm=self.satellite_params['tx_power'] + 30,  # 转换为dBm
                        guaranteed_power_ratio=guaranteed_power_ratio_dl,
                        fairness=fairness_dl,
                        precision=self.precision
                    )
                
                # 在子载波间分配功率
                tx_power = sionna.sys.spread_across_subcarriers(
                    tx_power_per_ut,
                    is_scheduled,
                    num_tx=1,  # 每个波束一个发射天线
                    precision=self.precision
                )
                
                # 计算SINR
                sinr = get_sinr(
                    tx_power,
                    self.stream_management,
                    self.no,
                    self.direction,
                    h_freq_fading,
                    self.num_bs,
                    self.num_ut_per_sector,
                    self.num_streams_per_ut,
                    self.resource_grid
                )
                
                # 选择MCS
                mcs_index = self.olla(
                    num_allocated_re,
                    harq_feedback=harq_feedback,
                    sinr_eff=sinr_eff_feedback
                )
                
                # 计算PHY层性能
                num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
                    mcs_index,
                    sinr=sinr,
                    mcs_table_index=mcs_table_index,
                    mcs_category=int(self.direction == 'downlink')
                )
                
                # 更新SINR反馈
                sinr_eff_feedback = tf.where(
                    num_allocated_re > 0,
                    sinr_eff,
                    tf.cast(0., self.rdtype)
                )
                
                # 记录结果
                hist = record_results(
                    hist,
                    slot,
                    sim_failed=False,
                    pathloss_serving_cell=tf.reduce_sum(
                        pathloss_serving_cell, axis=-2
                    ),
                    num_allocated_re=num_allocated_re,
                    tx_power_per_ut=tf.reduce_sum(
                        tx_power_per_ut, axis=-2
                    ),
                    num_decoded_bits=num_decoded_bits,
                    mcs_index=mcs_index,
                    harq_feedback=harq_feedback,
                    olla_offset=self.olla.offset,
                    sinr_eff=sinr_eff,
                    pf_metric=self.scheduler.pf_metric
                )
                
                # 记录Sionna计算的SINR (用于比较)
                if slot == 0 or slot % 100 == 0:
                    print(f"  Sionna计算的SINR: {tf.reduce_mean(sinr):.2f} dB")
                
            except tf.errors.InvalidArgumentError as e:
                print(f"SINR computation did not succeed at slot {slot}.\n"
                      f"Error message: {e}. Skipping slot...")
                hist = record_results(
                    hist, 
                    slot, 
                    shape=[self.batch_size, self.num_bs, self.num_ut_per_sector], 
                    sim_failed=True
                )
            
            return [slot + 1, hist, harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq]
        
        # 使用while_loop进行时隙仿真
        _, hist, *_ = tf.while_loop(
            lambda i, *_: i < num_slots,
            simulate_slot,
            [0, hist, last_harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq]
        )
        
        # 组装结果
        for key in hist:
            hist[key] = hist[key].stack()
            
        return hist


def plot_coverage_stats(stats_history):
    """绘制覆盖统计图"""
    # 提取时间序列数据
    slots = np.arange(len(stats_history))
    coverage_rates = [stat['coverage_rate'] for stat in stats_history]
    covered_cells = [stat['covered_cells'] for stat in stats_history]
    covered_users = [stat['covered_users'] for stat in stats_history]
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 绘制覆盖率
    ax1.plot(slots, coverage_rates)
    ax1.set_ylabel('Coverage Rate (%)')
    ax1.set_title('Coverage Statistics')
    ax1.grid(True)
    
    # 绘制覆盖小区数
    ax2.plot(slots, covered_cells)
    ax2.set_ylabel('Number of Covered Cells')
    ax2.grid(True)
    
    # 绘制覆盖用户数
    ax3.plot(slots, covered_users)
    ax3.set_ylabel('Number of Covered Users')
    ax3.set_xlabel('Slot')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_channel_stats(channel_history):
    """绘制信道统计图"""
    # 提取时间序列数据
    slots = np.arange(len(channel_history))
    snrs = [stat['snr'] for stat in channel_history]
    path_losses = [stat['path_loss'] for stat in channel_history]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制SNR
    ax1.plot(slots, snrs)
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Channel Statistics')
    ax1.grid(True)
    
    # 绘制路径损耗
    ax2.plot(slots, path_losses)
    ax2.set_ylabel('Path Loss (dB)')
    ax2.set_xlabel('Slot')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_performance_metrics(hist):
    """绘制性能指标CDF图"""
    # 计算平均性能指标
    results_avg = {
        'TBLER': (1 - np.nanmean(hist['harq'], axis=0)).flatten(),
        'MCS': np.nanmean(hist['mcs_index'], axis=0).flatten(),
        '# decoded bits / slot': np.nanmean(hist['num_decoded_bits'], axis=0).flatten(),
        'Effective SINR [dB]': 10*np.log10(np.nanmean(hist['sinr_eff'], axis=0).flatten()),
        'OLLA offset': np.nanmean(hist['olla_offset'], axis=0).flatten(),
        'TX power [dBm]': 10*np.log10(np.nanmean(hist['tx_power'], axis=0).flatten()) + 30,
        'Pathloss [dB]': 10*np.log10(np.nanmean(hist['pathloss_serving_cell'], axis=0).flatten()),
        '# allocated REs / slot': np.nanmean(hist['num_allocated_re'], axis=0).flatten(),
        'PF metric': np.nanmean(hist['pf_metric'], axis=0).flatten()
    }
    metrics = list(results_avg.keys())
    
    # 绘制性能指标CDF图
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle('Per-user performance metrics', y=.99)
    
    for ii in range(3):
        for jj in range(3):
            ax = axs[ii, jj]
            if 3*ii + jj < len(metrics):
                metric = metrics[3*ii + jj]
                x, y = get_cdf(results_avg[metric])
                ax.plot(x, y)
                if metric == 'TBLER':
                    ax.plot([0.1]*2, [0, 1], '--k', label='target')
                    ax.legend()
                ax.set_xlabel(metric)
                ax.grid(True)
                ax.set_ylabel('CDF')
    
    fig.tight_layout()
    plt.show()


def get_cdf(values):
    """计算累积分布函数"""
    values = np.array(values).flatten()
    n = len(values)
    sorted_val = np.sort(values)
    cumulative_prob = np.arange(1, n + 1) / n
    return sorted_val, cumulative_prob


def main():
    """主函数"""
    # 创建资源网格配置
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=14,              # 每时隙14个OFDM符号
        fft_size=512,                     # FFT大小，即子载波数
        subcarrier_spacing=15e3,          # 子载波间隔15kHz
        num_tx=1,                         # 每用户1个发射天线
        num_streams_per_tx=1              # 每发射天线1个流
    )
    
    # 创建仿真器
    simulator = SatelliteSystemSimulator(
        batch_size=1,
        resource_grid=resource_grid,
        direction='downlink',             # 下行链路
        satellite_params=SATELLITE_PARAMS,
        cell_params=CELL_PARAMS,
        channel_params=CHANNEL_PARAMS,
        target_areas=TARGET_AREAS,
        coherence_time=100                # 信道相干时间100个时隙
    )
    
    # 运行仿真
    print("Starting simulation...")
    hist = simulator(
        num_slots=1000,                   # 仿真1000个时隙
        bler_target=0.1,                  # 目标块误率10%
        olla_delta_up=0.2                 # OLLA上调步长0.2dB
    )
    
    # 清理历史记录
    hist = clean_hist(hist)
    
    # 绘制性能指标
    print("Plotting performance metrics...")
    plot_performance_metrics(hist)
    
    print("Simulation completed.")


if __name__ == "__main__":
    main() 