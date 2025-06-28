import numpy as np
import tensorflow as tf
from sionna.phy import PI
from sionna.phy.constants import SPEED_OF_LIGHT  # 添加光速常量
from sionna.phy.channel.tr38901 import SystemLevelScenario

class SatelliteScenario(SystemLevelScenario):
    """卫星通信场景类,实现3GPP NTN规范的卫星信道特性"""
    
    def __init__(self, carrier_frequency, ut_array, bs_array, direction, o2i_model="low", height=500000., elevation=90., precision=None):
        # 在调用父类构造函数之前设置一些基本属性
        self._height = height
        self._elevation = elevation * PI / 180.
        super().__init__(carrier_frequency=carrier_frequency, ut_array=ut_array, bs_array=bs_array, direction=direction, o2i_model=o2i_model, precision=precision)

    def _compute_lsp_log_mean_std(self, dist_2d=None, h_ut=None, h_bs=None):
        """计算大尺度参数的对数均值和标准差
        
        这个方法有两种调用方式:
        1. 不带参数: 用于初始化时计算所有链路的参数
        2. 带参数: 用于计算特定链路的参数
        
        Parameters
        ----------
        dist_2d : tf.float | None
            2D距离 [m], 形状 [batch_size, num_bs, num_ut]
            
        h_ut : tf.float | None
            用户高度 [m], 形状 [batch_size, num_ut]
            
        h_bs : tf.float | None
            基站高度 [m], 形状 [batch_size, num_bs]
            
        Returns
        -------
        mean : tf.float
            对数均值
            
        std : tf.float
            对数标准差
        """
        if dist_2d is None:
            # 不带参数调用 - 用于初始化
            # 使用已经存储在类中的拓扑信息
            dist_2d = self.distance_2d  # [batch_size, num_bs, num_ut]
            h_ut = self.ut_loc[..., 2]  # [batch_size, num_ut]
            h_bs = self.bs_loc[..., 2]  # [batch_size, num_bs]
            
        # 为所有输入添加必要的维度
        dist_2d = tf.cast(dist_2d, self.rdtype)
        h_ut = tf.cast(h_ut, self.rdtype)
        h_bs = tf.cast(h_bs, self.rdtype)
        
        # 调整维度顺序，确保一致性
        # 将 dist_2d 转换为 [batch_size, num_bs, num_ut]
        if dist_2d.shape[-2] != h_bs.shape[-1]:
            dist_2d = tf.transpose(dist_2d, perm=[0, 2, 1])
            
        # 扩展维度以便广播
        # [batch_size, 1, num_ut]
        h_ut = tf.expand_dims(h_ut, axis=1)
        # [batch_size, num_bs, 1]
        h_bs = tf.expand_dims(h_bs, axis=-1)
        
        # 现在所有张量的维度都是兼容的：
        # dist_2d: [batch_size, num_bs, num_ut]
        # h_bs: [batch_size, num_bs, 1] -> 广播到 [batch_size, num_bs, num_ut]
        # h_ut: [batch_size, 1, num_ut] -> 广播到 [batch_size, num_bs, num_ut]
        dist_3d = tf.sqrt(dist_2d**2 + (h_bs - h_ut)**2)
        
        # 卫星场景的简化实现
        # 均值随距离对数增长
        mean = 20. * tf.math.log(dist_3d) / tf.math.log(10.)
        # 固定的标准差
        std = tf.ones_like(dist_2d, dtype=self.rdtype) * 4.0  # 4.0dB标准差
        
        return mean, std

    def clip_carrier_frequency_lsp(self, fc):
        """限制载波频率范围"""
        return fc  # 卫星场景不限制频率范围

    @property
    def los_parameter_filepath(self):
        """LOS参数文件路径"""
        return "satellite_los.json"  # 返回一个有效的文件名而不是None

    @property
    def nlos_parameter_filepath(self):
        """NLOS参数文件路径"""
        return "satellite_nlos.json"  # 返回一个有效的文件名而不是None

    @property
    def o2i_parameter_filepath(self):
        """室内外参数文件路径"""
        return "satellite_o2i.json"  # 返回一个有效的文件名而不是None

    def _load_params(self):
        """重写父类的参数加载方法"""
        # 设置所有LSP相关系数为0
        lsp_corr = {
            # ASD correlations
            "corrASDvsASA": 0.0,
            "corrASDvsDS": 0.0,
            "corrASDvsK": 0.0,
            "corrASDvsZSA": 0.0,
            "corrASDvsZSD": 0.0,
            "corrASDvsSF": 0.0,
            
            # ASA correlations
            "corrASAvsDS": 0.0,
            "corrASAvsK": 0.0,
            "corrASAvsZSA": 0.0,
            "corrASAvsZSD": 0.0,
            "corrASAvsSF": 0.0,
            
            # DS correlations
            "corrDSvsK": 0.0,
            "corrDSvsZSA": 0.0,
            "corrDSvsZSD": 0.0,
            "corrDSvsSF": 0.0,
            
            # K correlations
            "corrKvsZSA": 0.0,
            "corrKvsZSD": 0.0,
            "corrKvsSF": 0.0,
            "corrZSDvsK": 0.0,  # 添加缺失的参数
            
            # ZSA correlations
            "corrZSAvsZSD": 0.0,
            "corrZSAvsSF": 0.0,
            
            # ZSD correlations
            "corrZSDvsSF": 0.0,
            
            # SF correlations
            "corrSFvsK": 0.0,
            "corrSFvsZSA": 0.0,
            "corrSFvsZSD": 0.0,
            "corrSFvsASD": 0.0,
            "corrSFvsASA": 0.0,
            "corrSFvsDS": 0.0
        }

        # 设置LOS参数
        self._params_los = {
            **lsp_corr,  # 包含所有LSP相关系数
            "delay_spread": 1e-7,  # 示例值
            "k_factor": 10,        # 示例值
            "zoa_spread": 5,       # 示例值
            "zod_spread": 5,       # 示例值
            "aoa_spread": 10,      # 示例值
            "aod_spread": 10,      # 示例值
            # 添加其他必要的参数
            "mu_lgZSD": 0.0,
            "mu_lgZSA": 0.0,
            "mu_lgASD": 0.0,
            "mu_lgASA": 0.0,
            "mu_lgDS": -7.0,
            "mu_kf": 10.0,
            "sigma_lgZSD": 0.4,
            "sigma_lgZSA": 0.4,
            "sigma_lgASD": 0.4,
            "sigma_lgASA": 0.4,
            "sigma_lgDS": 0.4,
            "sigma_kf": 4.0,
            "sigma_sf": 4.0
        }
        
        # 设置NLOS参数
        self._params_nlos = {
            **lsp_corr,  # 包含所有LSP相关系数
            "delay_spread": 1e-6,  # 示例值
            "k_factor": 0,         # 示例值
            "zoa_spread": 10,      # 示例值
            "zod_spread": 10,      # 示例值
            "aoa_spread": 20,      # 示例值
            "aod_spread": 20,      # 示例值
            # 添加其他必要的参数
            "mu_lgZSD": 0.0,
            "mu_lgZSA": 0.0,
            "mu_lgASD": 0.0,
            "mu_lgASA": 0.0,
            "mu_lgDS": -6.0,
            "mu_kf": 0.0,
            "sigma_lgZSD": 0.4,
            "sigma_lgZSA": 0.4,
            "sigma_lgASD": 0.4,
            "sigma_lgASA": 0.4,
            "sigma_lgDS": 0.4,
            "sigma_kf": 4.0,
            "sigma_sf": 6.0
        }
        
        # 设置O2I参数
        self._params_o2i = {
            **lsp_corr,  # 包含所有LSP相关系数
            "o2i_loss_low": 10.0,    # 示例值
            "o2i_loss_high": 20.0,   # 示例值
            # 添加其他必要的参数
            "mu_lgZSD": 0.0,
            "mu_lgZSA": 0.0,
            "mu_lgASD": 0.0,
            "mu_lgASA": 0.0,
            "mu_lgDS": -6.5,
            "mu_kf": 5.0,
            "sigma_lgZSD": 0.4,
            "sigma_lgZSA": 0.4,
            "sigma_lgASD": 0.4,
            "sigma_lgASA": 0.4,
            "sigma_lgDS": 0.4,
            "sigma_kf": 4.0,
            "sigma_sf": 5.0
        }

    @property
    def min_2d_in(self):
        """最小2D距离"""
        return 0.0

    @property
    def max_2d_in(self):
        """最大2D距离"""
        return np.inf

    @property
    def rays_per_cluster(self):
        """每个簇的射线数"""
        return 20  # 卫星场景使用20条射线/簇

    def _compute_pathloss_basic(self):
        """计算基础路径损耗[dB]
        
        按照3GPP NTN规范计算:
        PL(d) = 32.45 + 20log10(f0 * d) + Fs + Lg + Ls
        
        其中:
        - f0: 载波频率(GHz)
        - d: 3D斜距距离(m)
        - Fs: 阴影衰落(暂不计算)
        - Lg + Ls: 大气吸收损耗和闪烁损耗(固定值5.5dB)
        """
        # 获取3D距离信息 [batch_size, num_bs, num_ut]
        distance_3d = self.distance_3d
        
        # 载波频率(GHz)
        fc = self.carrier_frequency/1e9  
        
        # 1. 基础自由空间损耗: 32.45 + 20log10(f0 * d)
        pl_basic = 32.45 + 20 * tf.math.log(fc * distance_3d) / tf.math.log(10.)
        
        # 2. 大气损耗和闪烁损耗(固定值5.5 dB)
        pl_atm_scint = tf.fill(pl_basic.shape, 5.5)
        
        # 3. 总路径损耗 (暂不包含阴影衰落)
        self._pl_b = pl_basic + pl_atm_scint

    def calculate_tx_power(self):
        """计算发射功率[dBW/MHz]"""
        return self.TX_POWER[self._beam_type]
        
    def calculate_antenna_gain(self, theta, phi):
        """计算发射天线增益[dB]
        
        Parameters
        ----------
        theta : [...], tf.float
            俯仰角 [rad]
        phi : [...], tf.float  
            方位角 [rad]
        """
        # 选择天线阵列大小
        array_size = self.ARRAY_SIZE[self._beam_type]
        
        # 计算相控阵天线增益
        wavelength = SPEED_OF_LIGHT / self.carrier_frequency
        d = wavelength / 2  # 天线单元间距
        
        # 计算阵列因子
        array_factor = tf.zeros_like(theta, dtype=tf.complex64)
        
        for m in range(array_size[0]):
            for n in range(array_size[1]):
                phase = 2 * PI * d / wavelength * (
                    m * tf.math.sin(theta) * tf.math.cos(phi) +
                    n * tf.math.sin(theta) * tf.math.sin(phi)
                )
                array_factor += tf.exp(tf.complex(0., phase))
                
        # 计算增益
        gain = 20 * tf.math.log(tf.abs(array_factor)) / tf.math.log(10.)
        
        return gain

    def calculate_rx_power(self, d_3d, theta, phi):
        """计算接收功率[dBW]"""
        
        # 1. 发射功率
        pt = self.calculate_tx_power()
        
        # 2. 发射天线增益
        gt = self.calculate_antenna_gain(theta, phi)
        
        # 3. 接收天线增益
        gr = self.UE_GT
        
        # 4. 路径损耗
        pl = self._pl_b
        
        # 计算接收功率
        pr = pt + gt + gr - pl
        
        return pr
        
    def los_probability(self, d_2d, h_ut):
        """计算LOS概率
        
        对于卫星场景,LOS概率主要取决于仰角和天气条件
        这里使用简化模型
        
        Parameters
        ----------
        d_2d : array of float
            2D距离 [m]
            
        h_ut : array of float
            用户高度 [m]
            
        Returns
        -------
        array of float
            LOS概率
        """
        # 基于仰角的简化LOS概率模型
        prob = np.cos(PI/2 - self._elevation)
        return np.full_like(d_2d, prob)
    
    def path_loss_los(self, d_2d, d_3d, h_ut, h_bs):
        """计算LOS路径损耗
        
        按照3GPP NTN规范计算:
        PL(d) = 32.45 + 20log10(f0 * d) + Fs + Lg + Ls
        
        Parameters
        ----------
        d_2d : array of float
            2D距离 [m]
            
        d_3d : array of float
            3D距离(斜距) [m]
            
        h_ut : array of float
            用户高度 [m]
            
        h_bs : array of float
            基站(卫星)高度 [m]
            
        Returns
        -------
        array of float
            路径损耗 [dB]
        """
        # 基础路径损耗: 32.45 + 20log10(f0 * d)
        # f0单位为GHz, d单位为m
        f0_ghz = self.carrier_frequency / 1e9  # 转换为GHz
        pl_basic = 32.45 + 20 * np.log10(f0_ghz * d_3d)
        
        # 大气损耗和闪烁损耗: 固定值5.5 dB
        pl_atm_scint = np.full_like(d_3d, 5.5)
        
        # 阴影衰落: 对数正态分布
        # 使用shadow_fading_std方法获取标准差
        sigma_sf = self.shadow_fading_std(los=True)  
        pl_shadow = np.random.normal(0, sigma_sf, d_3d.shape)
        
        return pl_basic + pl_atm_scint + pl_shadow
    
    def path_loss_nlos(self, d_2d, d_3d, h_ut, h_bs):
        """计算NLOS路径损耗
        
        对于卫星场景,NLOS通常意味着完全遮挡
        返回一个很大的损耗值
        
        Parameters
        ----------
        d_2d : array of float
            2D距离 [m]
            
        d_3d : array of float
            3D距离 [m]
            
        h_ut : array of float
            用户高度 [m]
            
        h_bs : array of float
            基站(卫星)高度 [m]
            
        Returns
        -------
        array of float
            路径损耗 [dB]
        """
        # NLOS情况下设置一个很大的损耗
        return np.full_like(d_2d, 500.0)
    
    def o2i_loss(self, d_2d):
        """计算室内外损耗
        
        Parameters
        ----------
        d_2d : array of float
            2D距离 [m]
            
        Returns
        -------
        tuple
            (损耗均值, 标准差)
        """
        # 低穿透损耗和标准差
        o2i_loss_low = 20.0
        sigma_o2i_low = 5.0
        
        # 高穿透损耗和标准差
        o2i_loss_high = 30.0
        sigma_o2i_high = 7.0
        
        # 根据场景选择损耗和标准差
        if self._beam_type == "low":
            o2i_loss = o2i_loss_low
            sigma = sigma_o2i_low
        else:
            o2i_loss = o2i_loss_high
            sigma = sigma_o2i_high
            
        return (np.full_like(d_2d, o2i_loss),
                np.full_like(d_2d, sigma))
    
    def shadow_fading_std(self, los):
        """阴影衰落标准差
        
        根据3GPP NTN规范设置标准差
        
        Parameters
        ----------
        los : bool
            是否LOS
                
        Returns
        -------
        float
            标准差 [dB]
        """
        if los:
            return 2.0  # LOS情况下的标准差
        else:
            return 4.0  # NLOS情况下的标准差
            
    def zod_offset(self, d_2d, h_ut):
        """计算ZOD偏移
        
        Parameters
        ----------
        d_2d : array of float
            2D距离 [m]
            
        h_ut : array of float
            用户高度 [m]
            
        Returns
        -------
        array of float
            ZOD偏移 [度]
        """
        # 基于仰角计算ZOD偏移
        return np.full_like(d_2d, 90 - self._elevation * 180/PI) 