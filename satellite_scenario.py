import numpy as np
import tensorflow as tf
from sionna.phy import PI
from sionna.phy.constants import SPEED_OF_LIGHT  # 添加光速常量
from sionna.phy.channel.tr38901 import SystemLevelScenario

class SatelliteScenario(SystemLevelScenario):
    """卫星通信场景类,实现3GPP NTN规范的卫星信道特性"""
    
    def __init__(self, carrier_frequency, ut_array, bs_array, 
                 direction, enable_pathloss=True, enable_shadow_fading=True,
                 beam_type="service", height=500000., elevation=90.,
                 precision=None):
        
        # 调用父类初始化
        super().__init__(carrier_frequency, 'low', ut_array, bs_array,
            direction, enable_pathloss, enable_shadow_fading,
            precision=precision)
            
        # 卫星场景特定参数
        self._height = tf.cast(height, self.rdtype)
        self._elevation = tf.cast(elevation * PI / 180., self.rdtype)
        self._beam_type = beam_type
        
        # 发射功率参数 (dBW/MHz)
        self.TX_POWER = {
            "service": 41.41,    # 服务波束
            "broadcast": 32.29   # 广播波束
        }
        
        # 天线阵列参数
        self.ARRAY_SIZE = {
            "service": [20, 20],  # 服务波束
            "broadcast": [7, 7]   # 广播波束
        }
        
        # UE天线G/T
        self.UE_GT = -33.62  # dB/K

    def _compute_pathloss_basic(self):
        """计算基础路径损耗[dB]"""
        
        # 获取距离信息
        distance_3d = self.distance_3d
        fc = self.carrier_frequency/1e9  # 载波频率(GHz)
        
        # 1. 基础自由空间损耗
        pl_basic = 32.45 + 20 * tf.math.log(fc * distance_3d) / tf.math.log(10.)
        
        # 2. 大气损耗和闪烁损耗(固定值5.5 dB)
        pl_atm_scint = tf.fill(pl_basic.shape, 5.5)
        
        # 3. 阴影衰落
        sigma_sf = self.shadow_fading_std(True)
        pl_shadow = tf.random.normal(pl_basic.shape, 0., sigma_sf, dtype=self.rdtype)
        
        # 总路径损耗
        self._pl_b = pl_basic + pl_atm_scint + pl_shadow

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