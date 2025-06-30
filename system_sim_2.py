import sionna
import sionna.sys
import sionna.phy

import numpy as np
import matplotlib.pyplot as plt

from NTN.tr38811.scenarios.leo import LEO  # Import our new LEO scenario
from satellite import Satellite  # 导入卫星场景类

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

from topology import gen_custom_hexgrid_topology


class ChannelMatrix(sionna.phy.Block):
    """MIMO信道矩阵生成和管理类
    
    该类负责生成和更新MIMO系统的信道矩阵,包括时变衰落效应
    
    Args:
        resource_grid: OFDM资源网格配置对象
        batch_size: 批处理大小
        num_rx: 接收端数量
        num_tx: 发射端数量
        num_rx_ant: 接收天线数
        num_tx_ant: 发射天线数
        coherence_time: 信道相干时间(单位:时隙)
        precision: 数值精度
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
        
        # 打印初始化参数
        tf.print("\nDebug ChannelMatrix initialization:")
        tf.print("- batch_size:", batch_size)
        tf.print("- num_rx:", num_rx)
        tf.print("- num_tx:", num_tx)
        tf.print("- num_rx_ant:", num_rx_ant)
        tf.print("- num_tx_ant:", num_tx_ant)
        
        # 初始化衰落相关系数,范围0.95-0.99
        self.rho_fading = sionna.phy.config.tf_rng.uniform(
            [batch_size, num_rx, num_tx], 
            minval=.95, 
            maxval=.99, 
            dtype=self.rdtype
        )
        # 初始化衰落系数为1
        self.fading = tf.ones([batch_size, num_rx, num_tx], dtype=self.rdtype)

    def call(self, channel_model):
        """生成OFDM信道响应
        
        Args:
            channel_model: TR38.901信道模型对象
            
        Returns:
            h_freq: 频域信道响应
        """
        # 打印调试信息
        tf.print("\nDebug ChannelMatrix call method:")
        tf.print("- fading shape:", tf.shape(self.fading))
        
        # 生成信道响应
        h_freq, _ = channel_model(self.batch_size)  # 忽略时延返回值
        
        # 打印生成的信道响应形状
        tf.print("- h_freq shape:", tf.shape(h_freq))
        
        # 确保数据类型正确
        h_freq = tf.cast(h_freq, self.cdtype)
        
        return h_freq

    def update(self,
               channel_model,
               h_freq,
               slot):
        """根据相干时间更新信道响应
        
        每个相干时间周期重新生成信道响应
        
        Args:
            channel_model: TR38.901信道模型对象
            h_freq: 当前频域信道响应
            slot: 当前时隙索引
            
        Returns:
            h_freq: 更新后的频域信道响应
        """
        # 打印调试信息
        tf.print("\nDebug update method:")
        tf.print("- input h_freq shape:", tf.shape(h_freq))
        
        h_freq_new = self.call(channel_model)
        # 判断是否需要更新信道(基于相干时间)
        change = tf.cast(tf.math.mod(
            slot, self.coherence_time) == 0, self.cdtype)
        # 在相干时间边界更新信道响应
        h_freq = change * h_freq_new + \
            (tf.cast(1, self.cdtype) - change) * h_freq
            
        # 打印更新后的形状
        tf.print("- updated h_freq shape:", tf.shape(h_freq))
        
        return h_freq

    def apply_fading(self,
                     h_freq):
        """应用时变衰落效应
        
        使用自回归过程模拟时变衰落
        
        Args:
            h_freq: 频域信道响应
            
        Returns:
            h_freq_fading: 加入衰落效应后的信道响应
        """
        # 打印输入形状
        tf.print("\nDebug apply_fading input shapes:")
        tf.print("- h_freq shape:", tf.shape(h_freq))
        tf.print("- self.fading shape:", tf.shape(self.fading))
        
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
        
        # 获取h_freq的维度
        h_shape = tf.shape(h_freq)
        
        # 检查维度是否匹配预期
        tf.assert_equal(h_shape[1], self.num_rx, 
                       message="h_freq的接收端维度与预期不符")
        tf.assert_equal(h_shape[2], self.num_tx, 
                       message="h_freq的发射端维度与预期不符")
        
        # 逐步添加维度
        fading_expand = self.fading
        fading_expand = tf.expand_dims(fading_expand, axis=3)  # 接收天线维度
        fading_expand = tf.expand_dims(fading_expand, axis=4)  # 发射天线维度
        fading_expand = tf.expand_dims(fading_expand, axis=5)  # OFDM符号维度
        fading_expand = tf.expand_dims(fading_expand, axis=6)  # 子载波维度
        
        # 打印扩展后的形状
        tf.print("Expanded fading shape:", tf.shape(fading_expand))
        
        # 广播到目标形状
        fading_expand = tf.broadcast_to(
            fading_expand,
            [
                h_shape[0],  # batch_size
                h_shape[1],  # num_rx
                h_shape[2],  # num_tx
                h_shape[3],  # num_rx_ant
                h_shape[4],  # num_tx_ant
                h_shape[5],  # num_ofdm_symbols
                h_shape[6]   # num_subcarriers
            ]
        )
        
        # 打印最终形状
        tf.print("Final shapes:")
        tf.print("- fading_expand:", tf.shape(fading_expand))
        tf.print("- h_freq:", tf.shape(h_freq))
        
        # 应用衰落到信道响应
        h_freq_fading = tf.cast(tf.math.sqrt(fading_expand), self.cdtype) * h_freq
        return h_freq_fading


def get_stream_management(direction,
                          num_rx,
                          num_tx,
                          num_streams_per_ut,
                          num_ut_per_sector):
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


def get_stream_management_satellite(direction,
                          num_rx,
                          num_tx,
                          num_streams_per_ut,
                          num_ut_per_beam):
    """为卫星场景创建流管理
    
    Args:
        direction: 传输方向 ('uplink' 或 'downlink')
        num_rx: 接收端数量
        num_tx: 发射端数量
        num_streams_per_ut: 每个用户的流数
        num_ut_per_beam: 每个波束的用户数
        
    Returns:
        stream_management: StreamManagement对象
    """
    if direction == "downlink":
        num_streams_per_tx = num_streams_per_ut * num_ut_per_beam
        rx_tx_association = np.zeros([num_rx, num_tx])
        # 每个波束服务其对应的用户
        idx = np.array([[i1, i2] for i2 in range(num_tx) 
                       for i1 in np.arange(i2 * num_ut_per_beam, 
                                         (i2 + 1) * num_ut_per_beam)])
    else:
        num_streams_per_tx = num_streams_per_ut
        rx_tx_association = np.zeros([num_rx, num_tx])
        # 每个用户对应其所在波束
        idx = np.array([[i1, i2] for i1 in range(num_rx) 
                       for i2 in np.arange(i1 * num_ut_per_beam, 
                                         (i1 + 1) * num_ut_per_beam)])
    
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
    """计算SINR
    
    Args:
        tx_power: 发射功率
        stream_management: 流管理对象
        no: 噪声功率谱密度
        direction: 传输方向 ('uplink' 或 'downlink')
        h_freq_fading: 信道矩阵
        num_bs: 基站数量
        num_ut_per_sector: 每扇区用户数
        num_streams_per_ut: 每用户流数
        resource_grid: 资源网格配置
        
    Returns:
        sinr: 计算得到的SINR
    """
    # 1. 打印初始信息用于调试
    tf.print("\n=== SINR Calculation Debug Info ===")
    h_shape = tf.shape(h_freq_fading)
    tf.print("Initial h_freq_fading shape:", h_shape)
    
    try:
        # 2. 重塑发射功率维度
        s = tx_power.shape
        tx_power = tf.reshape(tx_power, [s[0], s[1] * s[2]] + s[3:])
        tf.print("Reshaped tx_power shape:", tf.shape(tx_power))
        
        # 3. 重塑信道矩阵为标准7维格式
        # Sionna期望的维度顺序: [batch_size, num_rx, num_tx, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
        h_freq_fading = tf.transpose(h_freq_fading, [0, 1, 2, 3, 5, 8, 7])
        
        # 4. 选择预编码方案
        if direction == "downlink":
            tf.print("Using RZF precoding for downlink")
            precoded_channel = sionna.phy.ofdm.RZFPrecodedChannel(
                resource_grid=resource_grid,
                stream_management=stream_management
            )
            h_eff = precoded_channel(h_freq_fading, tx_power=tx_power, alpha=no)
        else:
            tf.print("Using Eye precoding for uplink")
            precoded_channel = sionna.phy.ofdm.EyePrecodedChannel(
                resource_grid=resource_grid,
                stream_management=stream_management
            )
            h_eff = precoded_channel(h_freq_fading, tx_power=tx_power)
        
        # 5. 计算SINR
        tf.print("Computing LMMSE post-equalization SINR")
        lmmse_posteq_sinr = sionna.phy.ofdm.LMMSEPostEqualizationSINR(
            resource_grid=resource_grid,
            stream_management=stream_management
        )
        sinr = lmmse_posteq_sinr(h_eff, no=no)
        tf.print("SINR computation successful")
        
        # 6. 重塑SINR维度以匹配期望的输出格式
        sinr = tf.reshape(
            sinr, 
            sinr.shape[:-2] + [num_bs * num_ut_per_sector, num_streams_per_ut]
        )
        sinr = tf.reshape(
            sinr, 
            sinr.shape[:-2] + [num_bs, num_ut_per_sector, num_streams_per_ut]
        )
        sinr = tf.transpose(sinr, [0, 3, 1, 2, 4, 5])
        
        tf.print("Final SINR shape:", tf.shape(sinr))
        return sinr
        
    except Exception as e:
        tf.print("\nERROR in SINR calculation:")
        tf.print("Error message:", e)
        tf.print("Current shapes:")
        tf.print("- h_freq_fading:", tf.shape(h_freq_fading))
        tf.print("- tx_power:", tf.shape(tx_power))
        tf.print("- stream_management rx_tx_association:", 
                tf.shape(stream_management.rx_tx_association))
        raise


def estimate_achievable_rate(sinr_eff_db_last,
                             num_ofdm_sym,
                             num_subcarriers):
    """估计可达速率
    
    基于香农公式计算给定SINR下的理论可达速率
    
    Args:
        sinr_eff_db_last: 上一时隙的有效SINR(dB)
        num_ofdm_sym: OFDM符号数
        num_subcarriers: 子载波数
        
    Returns:
        rate_achievable_est: 估计的可达速率
    """
    # 使用香农公式计算可达速率
    rate_achievable_est = sionna.phy.utils.log2(
        tf.cast(1, sinr_eff_db_last.dtype) + 
        sionna.phy.utils.db_to_lin(sinr_eff_db_last)
    )
    # 扩展维度
    rate_achievable_est = sionna.phy.utils.insert_dims(
        rate_achievable_est, 2, axis=-2
    )
    # 复制到所有资源元素
    rate_achievable_est = tf.tile(
        rate_achievable_est, 
        [1, 1, num_ofdm_sym, num_subcarriers, 1]
    )
    return rate_achievable_est


def init_result_history(batch_size,
                        num_slots,
                        num_bs,
                        num_ut_per_sector):
    """初始化仿真结果历史记录
    
    创建TensorArray存储各项性能指标
    
    Args:
        batch_size: 批处理大小
        num_slots: 总时隙数
        num_bs: 基站数量
        num_ut_per_sector: 每扇区用户数
        
    Returns:
        hist: 包含各项指标的字典
    """
    hist = {}
    # 初始化各项性能指标的存储数组
    for key in ['pathloss_serving_cell',  # 服务小区路损
                'tx_power',                # 发射功率
                'olla_offset',             # OLLA偏移
                'sinr_eff',                # 有效SINR
                'pf_metric',               # 比例公平度量
                'num_decoded_bits',        # 解码比特数
                'mcs_index',               # MCS索引
                'harq',                    # HARQ反馈
                'num_allocated_re']:       # 分配的资源元素数
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



class SystemLevelSimulator(sionna.phy.Block):
    def __init__(self,
                 batch_size,
                 num_rings,
                 num_ut_per_sector,  # 对于卫星场景，这是每个波束的用户数
                 carrier_frequency,
                 resource_grid,
                 scenario,
                 direction,
                 ut_array,
                 bs_array,
                 bs_max_power_dbm,
                 ut_max_power_dbm,
                 coherence_time,
                 pf_beta=0.98,
                 max_bs_ut_dist=None,
                 min_bs_ut_dist=None,
                 temperature=294,
                 o2i_model='low',
                 average_street_width=20.0,
                 average_building_height=5.0,
                 custom_bs_positions=None,
                 precision=None):
        super().__init__(precision=precision)
        assert scenario in ['umi', 'uma', 'rma', 'satellite']
        assert direction in ['uplink', 'downlink']
        self.scenario = scenario
        self.batch_size = int(batch_size)
        self.resource_grid = resource_grid
        self.direction = direction
        self.bs_max_power_dbm = bs_max_power_dbm
        self.ut_max_power_dbm = ut_max_power_dbm
        self.coherence_time = tf.cast(coherence_time, tf.int32)
        
        # 计算基站（卫星波束）数量
        if scenario == 'satellite':
            self.num_bs = sionna.sys.get_num_hex_in_grid(num_rings)  # 每个六边形代表一个波束
            self.num_sectors_per_bs = 1  # 卫星场景不需要扇区
            self.num_ut_per_sector = int(num_ut_per_sector)  # 每个波束的用户数
            self.num_ut = self.num_bs * self.num_ut_per_sector  # 总用户数
        else:
            num_cells = sionna.sys.get_num_hex_in_grid(num_rings)
            self.num_bs = num_cells  # 基站数量
            self.num_sectors_per_bs = 3  # 每个基站3个扇区
            self.num_ut_per_sector = int(num_ut_per_sector)  # 每个扇区的用户数
            self.num_ut = self.num_bs * self.num_sectors_per_bs * self.num_ut_per_sector  # 总用户数
            
        # 设置天线数量
        if scenario == 'satellite':
            # 卫星场景
            self.num_ut_ant = ut_array.num_ant  # 用户天线数
            self.num_bs_ant = bs_array.num_ant  # 卫星天线数
            if bs_array.polarization == 'dual':
                self.num_bs_ant *= 2  # 双极化天线
        else:
            # 地面场景
            self.num_ut_ant = ut_array.num_ant
            self.num_bs_ant = bs_array.num_ant
            if bs_array.polarization == 'dual':
                self.num_bs_ant *= 2
            
        # 设置发射接收端数量
        if self.direction == 'uplink':
            self.num_tx = self.num_ut  # 发射端是用户
            self.num_rx = self.num_bs * (self.num_sectors_per_bs if scenario != 'satellite' else 1)  # 接收端是基站/卫星
            self.num_tx_ant = self.num_ut_ant  # 发射天线数是用户天线数
            self.num_rx_ant = self.num_bs_ant  # 接收天线数是基站/卫星天线数
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx = self.num_bs * (self.num_sectors_per_bs if scenario != 'satellite' else 1)  # 发射端是基站/卫星
            self.num_rx = self.num_ut  # 接收端是用户
            self.num_tx_ant = self.num_bs_ant  # 发射天线数是基站/卫星天线数
            self.num_rx_ant = self.num_ut_ant  # 接收天线数是用户天线数
            self.num_tx_per_sector = 1

        self.num_streams_per_ut = resource_grid.num_streams_per_tx

        # 根据场景选择不同的流管理方式
        if scenario == 'satellite':
            self.stream_management = get_stream_management_satellite(
                direction,
                self.num_rx,
                self.num_tx,
                self.num_streams_per_ut,
                self.num_ut_per_sector
            )
        else:
            self.stream_management = get_stream_management(
                direction,
                self.num_rx,
                self.num_tx,
                self.num_streams_per_ut,
                self.num_ut_per_sector
            )

        # 打印调试信息
        tf.print("\nDebug system parameters:")
        tf.print("- scenario:", self.scenario)
        tf.print("- direction:", self.direction)
        tf.print("- num_bs:", self.num_bs)
        tf.print("- num_sectors_per_bs:", self.num_sectors_per_bs)
        tf.print("- num_ut_per_sector:", self.num_ut_per_sector)
        tf.print("- total num_ut:", self.num_ut)
        tf.print("- num_tx:", self.num_tx)
        tf.print("- num_rx:", self.num_rx)
        tf.print("- num_tx_ant:", self.num_tx_ant)
        tf.print("- num_rx_ant:", self.num_rx_ant)
        tf.print("- num_streams_per_ut:", self.num_streams_per_ut)

        self.no = tf.cast(sionna.phy.constants.BOLTZMANN_CONSTANT * temperature * 
                          resource_grid.subcarrier_spacing, self.rdtype)
        
        self.slot_duration = resource_grid.ofdm_symbol_duration * \
            resource_grid.num_ofdm_symbols
        
        self._setup_channel_model(
            scenario, carrier_frequency, o2i_model, ut_array, bs_array,
            average_street_width, average_building_height
        )

        self._setup_topology(num_rings, min_bs_ut_dist, max_bs_ut_dist, custom_bs_positions)
        self.phy_abs = sionna.sys.PHYAbstraction(precision=self.precision)
        
        self.olla = sionna.sys.OuterLoopLinkAdaptation(
            self.phy_abs,
            self.num_ut_per_sector,
            batch_size=[self.batch_size, self.num_bs]
        )

        self.scheduler = sionna.sys.PFSchedulerSUMIMO(
            self.num_ut_per_sector,
            resource_grid.fft_size,
            resource_grid.num_ofdm_symbols,
            batch_size=[self.batch_size, self.num_bs],
            num_streams_per_ut=self.num_streams_per_ut,
            beta=pf_beta,
            precision=self.precision
        )

    def _setup_channel_model(self, scenario, carrier_frequency, o2i_model,
                             ut_array, bs_array, average_street_width,
                             average_building_height):
        common_params = {
            'carrier_frequency': carrier_frequency,
            'ut_array': ut_array,
            'bs_array': bs_array,
            'direction': self.direction,
            'enable_pathloss': True,
            'precision': self.precision
        }

        if scenario == 'umi':
            self.channel_model = sionna.phy.channel.tr38901.UMi(o2i_model=o2i_model, **common_params)
        elif scenario == 'uma':
            self.channel_model = sionna.phy.channel.tr38901.UMa(o2i_model=o2i_model, **common_params)
        elif scenario == 'rma':
            self.channel_model = sionna.phy.channel.tr38901.RMa(
                average_street_width=average_street_width,
                average_building_height=average_building_height,
                **common_params
            )
        elif scenario == 'satellite':
            # 使用新的LEO场景类
            self.channel_model = LEO(
                height=500000.,        # 500km
                elevation=90.,         # 90度
                beam_type="service",   # 服务波束
                enable_atmospheric_loss=True,
                enable_rain_loss=True,
                enable_clouds_fog_loss=True,
                enable_scintillation_loss=True,
                rain_rate=0.,
                resource_grid=self.resource_grid,  # 添加resource_grid参数
                **common_params
            )
            
            # 打印调试信息
            tf.print("\nDebug channel model setup:")
            tf.print("- num_rx:", self.num_rx)
            tf.print("- num_tx:", self.num_tx)
            tf.print("- num_rx_ant:", self.num_rx_ant)
            tf.print("- num_tx_ant:", self.num_tx_ant)

        # 初始化信道矩阵生成器
        self.channel_matrix = ChannelMatrix(
            resource_grid=self.resource_grid,
            batch_size=self.batch_size,
            num_rx=self.num_rx,        # 用户数
            num_tx=self.num_tx,        # 波束数
            num_rx_ant=self.num_rx_ant,  # 用户天线数
            num_tx_ant=self.num_tx_ant,  # 卫星天线数
            coherence_time=self.coherence_time,
            precision=self.precision
        )

        # 打印调试信息
        tf.print("\nDebug channel matrix setup:")
        tf.print("- resource_grid shape:", [
            self.resource_grid.num_ofdm_symbols,
            self.resource_grid.fft_size
        ])
        tf.print("- batch_size:", self.batch_size)
        tf.print("- num_rx:", self.num_rx)
        tf.print("- num_tx:", self.num_tx)
        tf.print("- num_rx_ant:", self.num_rx_ant)
        tf.print("- num_tx_ant:", self.num_tx_ant)
        tf.print("- coherence_time:", self.coherence_time)

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist, custom_bs_positions):
        self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, \
            self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc, self.grid = \
            gen_custom_hexgrid_topology(
                batch_size=self.batch_size,
                num_rings=num_rings,
                num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=min_bs_ut_dist,
                max_bs_ut_dist=max_bs_ut_dist,
                scenario=self.scenario,
                los=True,
                return_grid=True,
                custom_bs_positions=custom_bs_positions,
                precision=self.precision
            )
        self.channel_model.set_topology(
            ut_loc=self.ut_loc,
            bs_loc=self.bs_loc,
            ut_orientations=self.ut_orientations,
            bs_orientations=self.bs_orientations,
            ut_velocities=self.ut_velocities,
            los=self.los,
            bs_virtual_loc=self.bs_virtual_loc
        )

    def _reset(self,
               bler_target,
               olla_delta_up):
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
        """Group tensor by beam for satellite scenario or by sector for terrestrial scenario
        
        Args:
            tensor: Input tensor with shape [batch_size, num_rx, num_tx] or [batch_size, num_rx]
            
        Returns:
            Reshaped tensor with shape [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
        """
        # Print input tensor shape for debugging
        tensor_shape = tf.shape(tensor)
        tf.print("\nDebug _group_by_sector:")
        tf.print("- Input tensor shape:", tensor_shape)
        tf.print("- batch_size:", self.batch_size)
        tf.print("- num_bs:", self.num_bs)
        tf.print("- num_ut_per_sector:", self.num_ut_per_sector)
        tf.print("- scenario:", self.scenario)
        
        # 检查输入张量的维度
        tensor_rank = tf.rank(tensor)
        
        if tensor_rank == 1:
            # 如果输入是1维的，扩展维度
            tensor = tf.expand_dims(tensor, axis=0)  # Add batch dimension
            tensor = tf.expand_dims(tensor, axis=-1)  # Add tx dimension
        elif tensor_rank == 2:
            # 如果输入是2维的，添加tx维度
            tensor = tf.expand_dims(tensor, axis=-1)
        
        # 现在tensor应该是3维的 [batch_size, num_rx, num_tx]
        tf.print("- After dimension expansion:", tf.shape(tensor))
        
        if self.scenario == 'satellite':
            # 卫星场景：直接重塑为 [batch_size, num_bs, num_ut_per_sector]
            tensor = tf.reshape(tensor, [self.batch_size, self.num_bs, -1])
        else:
            # 地面场景：需要考虑扇区
            tensor = tf.reshape(tensor, [self.batch_size, -1, self.num_bs])
            tensor = tf.reduce_mean(tensor, axis=1)  # 取平均得到每个基站的值
            tensor = tf.expand_dims(tensor, axis=-1)  # 添加用户维度
            tensor = tf.tile(tensor, [1, 1, self.num_ut_per_sector])
        
        tf.print("- After first reshape:", tf.shape(tensor))
        
        # 确保最后一个维度是 num_ut_per_sector
        if tensor.shape[-1] != self.num_ut_per_sector:
            tensor = tf.reduce_mean(tensor, axis=-1, keepdims=True)
            tensor = tf.tile(tensor, [1, 1, self.num_ut_per_sector])
        
        # 添加 OFDM 符号维度
        tensor = tf.expand_dims(tensor, axis=2)  # [batch_size, num_bs, 1, num_ut_per_sector]
        tensor = tf.tile(tensor, [1, 1, self.resource_grid.num_ofdm_symbols, 1])
        
        tf.print("- Final shape:", tf.shape(tensor))
        return tensor

    @tf.function(jit_compile=True)
    def call(self,
             num_slots,
             alpha_ul,
             p0_dbm_ul,
             bler_target,
             olla_delta_up,
             mcs_table_index=1,
             fairness_dl=0,
             guaranteed_power_ratio_dl=0.5
    ):
        hist = init_result_history(self.batch_size, num_slots, self.num_bs, self.num_ut_per_sector)

        last_harq_feedback, sinr_eff_feedback, num_decoded_bits = \
            self._reset(bler_target, olla_delta_up)

        self.channel_matrix = ChannelMatrix(self.resource_grid,
                                            self.batch_size,
                                            self.num_rx,
                                            self.num_tx,
                                            self.num_rx_ant,
                                            self.num_tx_ant,
                                            self.coherence_time,
                                            precision=self.precision)

        h_freq = self.channel_matrix(self.channel_model)

        def simulate_slot(slot,
                          hist,
                          harq_feedback,
                          sinr_eff_feedback,
                          num_decoded_bits,
                          h_freq):
            try:
                h_freq = self.channel_matrix.update(self.channel_model,
                                                    h_freq,
                                                    slot)
                h_freq_fading = self.channel_matrix.apply_fading(h_freq)
                rate_achievable_est = estimate_achievable_rate(
                    self.olla.sinr_eff_db_last,
                    self.resource_grid.num_ofdm_symbols,
                    self.resource_grid.fft_size
                )
                is_scheduled = self.scheduler(
                    num_decoded_bits,
                    rate_achievable_est
                )

                num_allocated_sc = tf.minimum(tf.reduce_sum(
                    tf.cast(is_scheduled, tf.int32), axis=-1), 1
                )
                num_allocated_sc = tf.reduce_sum(
                    num_allocated_sc, axis=-2
                )
                num_allocated_re = \
                    tf.reduce_sum(tf.cast(is_scheduled, tf.int32),
                                  axis=[-1, -3, -4])

                # Print shapes for debugging
                tf.print("h_freq_fading shape:", tf.shape(h_freq_fading))
                tf.print("rx_tx_association shape:", tf.shape(self.stream_management.rx_tx_association))
                
                pathloss_all_pairs, pathloss_serving_cell = sionna.sys.get_pathloss(
                    h_freq_fading,
                    rx_tx_association=tf.convert_to_tensor(
                        self.stream_management.rx_tx_association
                    )
                )

                # Print pathloss shapes
                tf.print("pathloss_all_pairs shape:", tf.shape(pathloss_all_pairs))
                tf.print("pathloss_serving_cell shape:", tf.shape(pathloss_serving_cell))

                # Reduce over unnecessary dimensions for pathloss_serving_cell
                pathloss_serving_cell = tf.reduce_mean(pathloss_serving_cell, axis=[-2, -1])  # Average over antennas and OFDM symbols
                
                # Print reduced shape
                tf.print("reduced pathloss_serving_cell shape:", tf.shape(pathloss_serving_cell))

                pathloss_serving_cell = self._group_by_sector(
                    pathloss_serving_cell
                )

                if self.direction == 'uplink':
                    tx_power_per_ut = sionna.sys.open_loop_uplink_power_control(
                        pathloss_serving_cell,
                        num_allocated_sc,
                        alpha=alpha_ul,
                        p0_dbm=p0_dbm_ul,
                        ut_max_power_dbm=self.ut_max_power_dbm
                    )
                else:
                    one = tf.cast(1, pathloss_serving_cell.dtype)
                    rx_power_tot = tf.reduce_sum(
                        one / pathloss_all_pairs, axis=-2
                    )
                    rx_power_tot = self._group_by_sector(rx_power_tot)
                    interference_dl = rx_power_tot - one / pathloss_serving_cell
                    interference_dl *= sionna.phy.utils.dbm_to_watt(self.bs_max_power_dbm)

                    tx_power_per_ut, _ = sionna.sys.downlink_fair_power_control(
                        pathloss_serving_cell,
                        interference_dl + self.no,
                        num_allocated_sc,
                        bs_max_power_dbm=self.bs_max_power_dbm,
                        guaranteed_power_ratio=guaranteed_power_ratio_dl,
                        fairness=fairness_dl,
                        precision=self.precision
                    )
                tx_power = sionna.sys.spread_across_subcarriers(
                    tx_power_per_ut,
                    is_scheduled,
                    num_tx=self.num_tx_per_sector,
                    precision=self.precision
                )

                sinr = get_sinr(tx_power,
                                self.stream_management,
                                self.no,
                                self.direction,
                                h_freq_fading,
                                self.num_bs,
                                self.num_ut_per_sector,
                                self.num_streams_per_ut,
                                self.resource_grid)

                mcs_index = self.olla(num_allocated_re,
                                      harq_feedback=harq_feedback,
                                      sinr_eff=sinr_eff_feedback)
                
                num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
                    mcs_index,
                    sinr=sinr,
                    mcs_table_index=mcs_table_index,
                    mcs_category=int(self.direction == 'downlink')
                )

                sinr_eff_feedback = tf.where(num_allocated_re > 0,
                                             sinr_eff,
                                             tf.cast(0., self.rdtype))
                
                hist = record_results(hist,
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
            except tf.errors.InvalidArgumentError as e:
                print(f"SINR computation did not succeed at slot {slot}.\n"
                      f"Error message: {e}. Skipping slot...")
                hist = record_results(hist, slot, shape=[self.batch_size, self.num_bs, self.num_ut_per_sector], sim_failed=True)

            self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration
            self.channel_model.set_topology(
                ut_loc=self.ut_loc,
                bs_loc=self.bs_loc,
                ut_orientations=self.ut_orientations,
                bs_orientations=self.bs_orientations,
                ut_velocities=self.ut_velocities,
                los=self.los,
                bs_virtual_loc=self.bs_virtual_loc
            )

            return [slot + 1, hist, harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq]

        _, hist, *_ = tf.while_loop(
            lambda i, *_: i < num_slots,
            simulate_slot,
            [0, hist, last_harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq]
        )
        for key in hist:
            hist[key] = hist[key].stack()
        return hist
                

def get_cdf(values):
    values = np.array(values).flatten()
    n = len(values)
    sorted_val = np.sort(values)
    cumulative_prob = np.arange(1, n + 1) / n
    return sorted_val, cumulative_prob


def pairplot(dic, keys, suptitle=None, figsize=2.5):
    fig, axs = plt.subplots(len(keys), len(keys), figsize=[len(keys)*figsize]*2)
    for row, key_row in enumerate(keys):
        for col, key_col in enumerate(keys):
            ax = axs[row, col]
            ax.grid()
            if row == col:
                ax.hist(dic[key_row], bins=30,
                        color='skyblue', edgecolor='k',
                        linewidth=.5)
            elif col > row:
                fig.delaxes(ax)
            else:
                ax.scatter(dic[key_col], dic[key_row], s=16, color='skyblue', alpha=0.9, linewidth=.5, edgecolor='k')
            ax.set_ylabel(key_row)
            ax.set_xlabel(key_col)
    if suptitle is not None:
        fig.suptitle(suptitle, y=1, fontsize=17)
    fig.tight_layout()
    return fig, axs


def main():
    # 1、基本场景参数
    direction = 'downlink'
    scenario = 'satellite'  # 使用卫星场景
    beam_type = "service"   # 服务波束

    # 2、卫星波束参数
    num_rings = 1          # 生成7个波束
    num_ut_per_beam = 10   # 每个波束10个用户
    cell_radius = 22.6     # 波束半径为22.60km
    isd = cell_radius * np.sqrt(3)  # 波束间距 = 波束半径 * sqrt(3)
    max_bs_ut_dist = cell_radius  # 最大用户距离设为波束半径
    min_bs_ut_dist = 0  # 最小用户距离
    
    carrier_frequency = 3.5e9  # 载波频率3.5GHz

    coherence_time = 100  # 相干时间
    mcs_table_index = 1   # MCS表索引
    batch_size = 1        # 批处理大小

    # 卫星天线阵列
    bs_array = sionna.phy.channel.tr38901.PanelArray(
        num_rows_per_panel=20,    # 20x20阵列
        num_cols_per_panel=20,
        polarization='dual',      # 双极化
        polarization_type='VH',
        antenna_pattern='38.901',
        carrier_frequency=carrier_frequency,
        element_vertical_spacing=0.5,
        element_horizontal_spacing=0.5
    )

    # 用户天线
    ut_array = sionna.phy.channel.tr38901.PanelArray(
        num_rows_per_panel=1,     # 单天线
        num_cols_per_panel=1,
        polarization='single',    # 单极化
        polarization_type='V',
        antenna_pattern='omni',   # 全向天线
        carrier_frequency=carrier_frequency
    )

    # 功率配置
    system_bandwidth = 30e6  # 系统带宽30MHz
    eirp_density = 41.41    # EIRP功率密度(dBW/MHz)
    # 转换为dBm
    bs_max_power_dbm = eirp_density + 10*np.log10(system_bandwidth/1e6) + 30
    ut_max_power_dbm = 26  # 用户终端功率

    # 资源网格配置
    subcarrier_spacing = 15e3  # 子载波间隔15kHz
    num_ofdm_sym = 14        # 每时隙14个OFDM符号
    num_subcarriers = int(system_bandwidth/subcarrier_spacing)  # 子载波数
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=num_ofdm_sym,
        fft_size=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=1,  # 每波束一个发射天线
        num_streams_per_tx=ut_array.num_ant  # 每个用户的流数等于其天线数
    )
    
    # 时间参数
    slot_duration = 1e-3  # 时隙持续时间1ms
    num_slots = 1000  # 仿真时隙数

    # 卫星位置配置
    hex_radius = cell_radius * 1000  # 转换为米
    satellite_height = 500000  # 卫星高度500km
    
    # 创建六边形网格
    grid = sionna.sys.topology.HexGrid(
        num_rings=num_rings,
        cell_radius=hex_radius,
        cell_height=satellite_height
    )
    
    # 获取小区中心位置作为卫星位置
    cell_centers = grid.cell_loc.numpy()
    
    # 构建卫星位置字典
    custom_bs_positions = {}
    for i, center in enumerate(cell_centers):
        custom_bs_positions[i] = tuple(center)
            
    print(f"\n总卫星数: {len(custom_bs_positions)}")
    print(f"理论小区数: {sionna.sys.get_num_hex_in_grid(num_rings)}")
    
    # 创建系统级仿真器
    sls = SystemLevelSimulator(
        batch_size,
        num_rings,
        num_ut_per_beam,  # 每个波束的用户数
        carrier_frequency,
        resource_grid,
        scenario,
        direction,
        ut_array,
        bs_array,
        bs_max_power_dbm,
        ut_max_power_dbm,
        coherence_time,
        max_bs_ut_dist=max_bs_ut_dist*1000,  # 转换为米
        min_bs_ut_dist=min_bs_ut_dist*1000,  # 转换为米
        temperature=294,
        o2i_model='low',
        average_street_width=20.,
        average_building_height=10.,
        custom_bs_positions=custom_bs_positions,
        precision=None
    )

    # 打印每个波束的中心位置
    print("\n波束中心位置:")
    cell_centers = sls.grid.cell_loc.numpy()
    for i, center in enumerate(cell_centers):
        print(f"波束 {i}: (x={center[0]:.2f}, y={center[1]:.2f}, z={center[2]:.2f})")

    # 2D视图
    plt.figure(figsize=(10, 8))
    
    # 使用grid的show方法绘制波束边界和卫星位置
    grid.show(show_sectors=False,  # 不显示扇区
             show_mirrors=False,   # 不显示镜像小区
             color='b',           # 边界颜色
             label='beam')        # 图例标签改为beam
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Satellite System Topology with Beam Centers (2D view)')
    plt.legend()
    
    # 3D视图
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # 绘制用户位置
    ut_positions = sls.ut_loc[0].numpy()
    ax_3d.scatter(ut_positions[:, 0]/1000, ut_positions[:, 1]/1000, ut_positions[:, 2]/1000,
                 c='black', marker='x', label='User positions')
    
    # 绘制卫星位置
    cell_centers = grid.cell_loc.numpy()
    ax_3d.scatter(cell_centers[:, 0]/1000, cell_centers[:, 1]/1000, cell_centers[:, 2]/1000,
                 c='red', marker='^', s=100, label='Satellite positions')
    
    # 为每个卫星添加标签
    for i, center in enumerate(cell_centers):
        ax_3d.text(center[0]/1000, center[1]/1000, center[2]/1000, f'Beam {i}')
    
    # 绘制从卫星到地面的投影线
    for center in cell_centers:
        ax_3d.plot([center[0]/1000, center[0]/1000], 
                  [center[1]/1000, center[1]/1000], 
                  [0, center[2]/1000], 
                  'r--', alpha=0.3)
    
    # 绘制六边形波束的地面投影
    for cell_idx, cell in grid.grid.items():
        corners = cell.corners()  # 获取六边形顶点
        # 连接顶点形成六边形
        x_coords = [corner[0]/1000 for corner in corners] + [corners[0][0]/1000]
        y_coords = [corner[1]/1000 for corner in corners] + [corners[0][1]/1000]
        ax_3d.plot(x_coords, y_coords, [0]*len(x_coords), 'b-', alpha=0.5)
    
    # 设置坐标轴标签
    ax_3d.set_xlabel('X (kilometers)')
    ax_3d.set_ylabel('Y (kilometers)')
    ax_3d.set_zlabel('Z (kilometers)')
    
    # 设置视角
    ax_3d.view_init(elev=20, azim=45)
    
    # 调整Z轴的范围，使卫星和地面用户都能清晰显示
    z_min = 0
    z_max = satellite_height/1000 * 1.1  # 略高于卫星高度
    ax_3d.set_zlim(z_min, z_max)
    
    # 设置标题和图例
    ax_3d.set_title('Satellite System Topology (3D View)')
    ax_3d.legend()
    
    plt.show()

    # 运行仿真并收集结果
    num_slots = tf.constant(1000, tf.int32)
    bler_target = tf.constant(0.1, tf.float32)
    olla_delta_up = tf.constant(0.2, tf.float32)

    alpha_ul = tf.constant(1., tf.float32)
    p0_dbm_ul = tf.constant(-80., tf.float32)
    
    print("\n开始仿真...")
    hist = sls(num_slots,
               alpha_ul,
               p0_dbm_ul,
               bler_target,
               olla_delta_up)
    hist = clean_hist(hist)
    
    print("\n仿真完成，生成性能指标...")
    
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
    fig, axs = plt.subplots(3, 3, figsize=(8, 6.5))
    fig.suptitle('Per-user performance metrics', y=.99)

    for ii in range(3):
        for jj in range(3):
            ax = axs[ii, jj]
            metric = metrics[3*ii + jj]
            ax.plot(*get_cdf(results_avg[metric]))
            if metric == 'TBLER':
                ax.plot([bler_target]*2, [0, 1], '--k', label='target')
                ax.legend()
            if metric == 'TX power [dBm]':
                ax.set_xlim(ax.get_xlim()[0] - .5, ax.get_xlim()[1] + .5)
            ax.set_xlabel(metric)
            ax.grid()
            ax.set_ylabel('CDF')
    
    fig.tight_layout()
    plt.show()

    # 绘制性能指标相关性图
    fig, axs = pairplot(results_avg, ['Effective SINR [dB]', 'MCS', '# decoded bits / slot'], 
                        suptitle='MCS, SINR, and throughput')
    plt.show()

    fig, axs = pairplot(results_avg, ['TBLER', 'MCS', 'OLLA offset'], 
                        suptitle='TBLER, MCS, and OLLA offset')
    for ii in range(3):
        axs[ii, 0].plot([bler_target]*2, axs[ii, 0].get_ylim(), '--k')
    plt.show()

    fig, axs = pairplot(results_avg, ['# allocated REs / slot', 'PF metric', 'MCS'], 
                        suptitle='PF metric, allocated resources, and MCS')
    plt.show()

    # 打印平均性能指标
    print("\n平均性能指标:")
    for metric, values in results_avg.items():
        print(f"{metric}: {np.mean(values):.2f}")

if __name__ == "__main__":
    main()