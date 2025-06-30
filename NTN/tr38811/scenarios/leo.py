"""Class implementing LEO (Low Earth Orbit) scenario from 3GPP TR38.811"""

import tensorflow as tf
import numpy as np

from sionna.phy import PI, SPEED_OF_LIGHT, config, dtypes
from sionna.phy.channel.tr38901 import PanelArray
from ..system_level_scenario import SystemLevelScenario
from sionna.phy.utils import log10

class LEO(SystemLevelScenario):
    """LEO (Low Earth Orbit) scenario class
    
    This class implements the LEO scenario from 3GPP TR38.811 specification.
    
    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    height : float
        Platform height [m]

    elevation : float
        Platform elevation angle [degree]

    ut_array : PanelArray
        Panel array configuration used by UTs

    bs_array : PanelArray
        Panel array configuration used by BSs (satellites)

    direction : str
        Link direction, either "uplink" or "downlink"

    enable_pathloss : bool, optional (default=True)
        If True, apply pathloss

    enable_atmospheric_loss : bool, optional (default=True)
        If True, apply atmospheric loss

    enable_rain_loss : bool, optional (default=True)
        If True, apply rain loss

    enable_clouds_fog_loss : bool, optional (default=True)
        If True, apply clouds and fog loss

    enable_scintillation_loss : bool, optional (default=True)
        If True, apply scintillation loss

    beam_type : str
        Type of beam, either "service" or "feeder"

    rain_rate : float
        Rain rate [mm/h]. Only used if enable_rain_loss is True.
        Defaults to 0.

    resource_grid : None | ResourceGrid, optional (default=None)
        Resource grid configuration used by the scenario

    precision : None (default) | "single" | "double"
        Precision used for internal calculations and outputs
    """
    def __init__(self,
                 carrier_frequency,
                 height,
                 elevation,
                 ut_array,
                 bs_array,
                 direction,
                 enable_pathloss=True,
                 enable_atmospheric_loss=True,
                 enable_rain_loss=True,
                 enable_clouds_fog_loss=True,
                 enable_scintillation_loss=True,
                 beam_type="service",
                 rain_rate=0.,
                 resource_grid=None,
                 precision=None):
        
        # Store LEO specific parameters first
        self._beam_type = beam_type
        self._rain_rate = None  # Will be set after parent class init
        
        # Store resource grid parameters
        if resource_grid is not None:
            self._num_subcarriers = resource_grid.fft_size
            self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        else:
            self._num_subcarriers = 2048  # Default value
            self._num_ofdm_symbols = 14   # Default value
        
        # Initialize the parent class
        super().__init__(carrier_frequency=carrier_frequency,
                        height=height,
                        elevation=elevation,
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction=direction,
                        enable_pathloss=enable_pathloss,
                        enable_atmospheric_loss=enable_atmospheric_loss,
                        enable_rain_loss=enable_rain_loss,
                        enable_clouds_fog_loss=enable_clouds_fog_loss,
                        enable_scintillation_loss=enable_scintillation_loss,
                        precision=precision)

        # Now set rain_rate using the correct dtype from parent class
        self._rain_rate = tf.cast(rain_rate, self.rdtype)

    @property
    def beam_type(self):
        """Type of beam, either "service" or "feeder" """
        return self._beam_type

    @property
    def rain_rate(self):
        """Rain rate [mm/h]"""
        return self._rain_rate

    @property
    def dtype(self):
        """Data type used for internal calculations"""
        return {
            'real_dtype': dtypes[self.precision]['tf']['rdtype'],
            'complex_dtype': dtypes[self.precision]['tf']['cdtype']
        }

    @property
    def rdtype(self):
        """Real data type used for internal calculations"""
        return dtypes[self.precision]['tf']['rdtype']

    @property
    def cdtype(self):
        """Complex data type used for internal calculations"""
        return dtypes[self.precision]['tf']['cdtype']

    @property
    def num_subcarriers(self):
        """Number of subcarriers"""
        return self._num_subcarriers

    @property
    def num_ofdm_symbols(self):
        """Number of OFDM symbols"""
        return self._num_ofdm_symbols

    def _initialize_leo_parameters(self):
        """Initialize LEO specific parameters according to TR 38.811"""
        
        # Orbital parameters
        self._orbital_period = 2 * PI * tf.sqrt((self.height + 6371e3)**3 / (6.67430e-11 * 5.972e24))
        self._orbital_velocity = 2 * PI * (self.height + 6371e3) / self._orbital_period

        # LSP parameters for LEO scenario
        if self.beam_type == "service":
            # Service link parameters (Table 6.6.2-1)
            self._lsp_log_mean = tf.constant([
                [-6.89],  # DS [log10(s)]
                [0.78],   # ASD [log10(degree)]
                [1.05],   # ASA [log10(degree)]
                [0],      # SF [dB]
                [9],      # K [dB]
                [0.92],   # ZSA [log10(degree)]
                [0.88]    # ZSD [log10(degree)]
            ], self.dtype['real_dtype'])

            self._lsp_log_std = tf.constant([
                [0.54],  # DS [log10(s)]
                [0.36],  # ASD [log10(degree)]
                [0.28],  # ASA [log10(degree)]
                [8],     # SF [dB]
                [3.5],   # K [dB]
                [0.41],  # ZSA [log10(degree)]
                [0.34]   # ZSD [log10(degree)]
            ], self.dtype['real_dtype'])

        else:  # feeder link parameters (Table 6.6.2-2)
            self._lsp_log_mean = tf.constant([
                [-7.39],  # DS
                [0.78],   # ASD
                [1.05],   # ASA
                [0],      # SF
                [12],     # K
                [0.92],   # ZSA
                [0.88]    # ZSD
            ], self.dtype['real_dtype'])

            self._lsp_log_std = tf.constant([
                [0.54],  # DS
                [0.36],  # ASD
                [0.28],  # ASA
                [6],     # SF
                [3],     # K
                [0.41],  # ZSA
                [0.34]   # ZSD
            ], self.dtype['real_dtype'])

    def get_param(self, param_name):
        """Get LEO scenario parameter by name

        Input
        ------
        param_name : str
            Name of the parameter

        Output
        -------
        : tf.float
            Value of the parameter
        """
        # LEO specific parameters from TR 38.811
        params = {
            # Cross-correlation between LSPs
            'corrASDvsDS': 0.4,
            'corrASAvsDS': 0.4,
            'corrASAvsSF': -0.4,
            'corrASDvsSF': -0.4,
            'corrDSvsSF': -0.4,
            'corrASDvsASA': 0.0,
            'corrASDvsK': 0.0,
            'corrASAvsK': 0.0,
            'corrDSvsK': 0.0,
            'corrSFvsK': 0.0,
            'corrZSDvsSF': -0.4,
            'corrZSAvsSF': -0.4,
            'corrZSDvsK': 0.0,
            'corrZSAvsK': 0.0,
            'corrZSDvsDS': 0.4,
            'corrZSAvsDS': 0.4,
            'corrZSDvsASD': 0.8,
            'corrZSAvsASD': 0.8,
            'corrZSDvsASA': 0.4,
            'corrZSAvsASA': 0.4,
            'corrZSDvsZSA': 0.0,

            # Correlation distances for LSPs
            'corrDistDS': 50,
            'corrDistASD': 50,
            'corrDistASA': 50,
            'corrDistSF': 50,
            'corrDistK': 50,
            'corrDistZSA': 50,
            'corrDistZSD': 50,

            # Delay spread
            'cDS': 20,  # Cluster delay spread in ns

            # Additional LEO parameters
            'orbital_velocity': self._orbital_velocity,  # Orbital velocity [m/s]
            'orbital_period': self._orbital_period,      # Orbital period [s]
            'min_elevation': self.elevation,             # Minimum elevation angle [deg]
            'satellite_height': self.height              # Satellite height [m]
        }

        return tf.cast(params.get(param_name, 0.0), self.dtype['real_dtype'])

    def compute_pathloss(self):
        """Compute path loss for LEO scenario

        The path loss model includes:
        - Free space path loss
        - Atmospheric loss
        - Rain loss
        - Clouds and fog loss
        - Scintillation loss

        Output
        -------
        : [batch size, num_ut, num_bs], tf.float
            Path loss [dB]
        """
        # Basic free space path loss
        wavelength = SPEED_OF_LIGHT/self.carrier_frequency
        fspl = 20*log10(4*PI*self._distance_3d/wavelength)

        # Initialize total path loss with free space path loss
        total_pl = fspl

        if self.atmospheric_loss_enabled:
            # Atmospheric loss (simplified model)
            # In reality, this depends on atmospheric composition, pressure, etc.
            atmospheric_loss = 0.1 * self._distance_3d/1000  # 0.1 dB/km
            total_pl += atmospheric_loss

        if self.rain_loss_enabled and self.rain_rate > 0:
            # Rain loss (simplified model)
            # In reality, this depends on frequency, polarization, etc.
            rain_loss = 0.2 * self.rain_rate * self._distance_3d/1000
            total_pl += rain_loss

        if self.clouds_fog_loss_enabled:
            # Clouds and fog loss (simplified model)
            clouds_loss = 0.05 * self._distance_3d/1000
            total_pl += clouds_loss

        if self.scintillation_loss_enabled:
            # Scintillation loss (simplified model)
            # Increases with frequency and decreases with elevation angle
            elevation_deg = self._elevation * 180/PI
            scintillation_loss = 0.5 * (self.carrier_frequency/1e9) * \
                                (90 - elevation_deg)/(90)
            total_pl += scintillation_loss

        if self.shadow_fading_enabled:
            # Shadow fading
            # Standard deviation depends on environment and elevation angle
            sf_std = 8.0  # dB, from TR 38.811
            shadow_fading = tf.random.normal(tf.shape(total_pl),
                                           mean=0.0,
                                           stddev=sf_std,
                                           dtype=self.dtype['real_dtype'])
            total_pl += shadow_fading

        return total_pl

    def set_topology(self,
                    ut_loc=None,
                    bs_loc=None,
                    ut_orientations=None,
                    bs_orientations=None,
                    ut_velocities=None,
                    los=None,
                    bs_virtual_loc=None):
        """Set the network topology
        
        Parameters
        ----------
        ut_loc : [batch_size, num_ut, 3], tf.float
            UTs locations

        bs_loc : [batch_size, num_bs, 3], tf.float
            BSs locations

        ut_orientations : [batch_size, num_ut, 3], tf.float
            UTs orientations [rad]

        bs_orientations : [batch_size, num_bs, 3], tf.float
            BSs orientations [rad]

        ut_velocities : [batch_size, num_ut, 3], tf.float
            UTs velocities [m/s]

        los : [batch_size, num_ut, num_bs], tf.bool
            LOS state. In NTN, typically always True.

        bs_virtual_loc : [batch_size, num_ut, num_bs, 3], tf.float
            Virtual BS locations for each UT [m]
        """
        # Call parent class set_topology
        super().set_topology(ut_loc,
                            bs_loc,
                            ut_orientations,
                            bs_orientations,
                            ut_velocities,
                            los,
                            bs_virtual_loc)

        # Compute distances and angles
        self._compute_distances_and_angles()

    def _compute_distances_and_angles(self):
        """Compute distances and angles between UTs and BSs
        
        This method computes:
        - 3D distances between UTs and BSs
        - Elevation angles
        - Azimuth angles
        """
        # Compute the relative positions
        # [batch_size, num_ut, num_bs, 3]
        ut_bs_relative_pos = tf.expand_dims(self.ut_loc, axis=2) - \
                            tf.expand_dims(self.bs_loc, axis=1)
        
        # 3D distance [batch_size, num_ut, num_bs]
        self._distance_3d = tf.norm(ut_bs_relative_pos, axis=-1)
        
        # Ground distance (2D projection) [batch_size, num_ut, num_bs]
        self._distance_2d = tf.norm(ut_bs_relative_pos[...,:2], axis=-1)
        
        # Height difference [batch_size, num_ut, num_bs]
        self._height_diff = tf.abs(ut_bs_relative_pos[...,2])
        
        # Elevation angle [batch_size, num_ut, num_bs]
        self._elevation = tf.asin(self._height_diff / self._distance_3d)
        
        # Azimuth angle [batch_size, num_ut, num_bs]
        self._azimuth = tf.atan2(ut_bs_relative_pos[...,1], 
                                ut_bs_relative_pos[...,0])

    def _compute_pathloss_basic(self):
        """Compute basic path loss according to TR38.811
        
        Returns
        -------
        : [batch_size, num_ut, num_bs], tf.float
            Basic path loss [dB]
        """
        # Wavelength
        wavelength = SPEED_OF_LIGHT/self.carrier_frequency
        
        # Free space path loss
        fspl = 20*log10(4*PI*self._distance_3d/wavelength)
        
        # Additional attenuation based on elevation angle
        # Convert elevation to degrees for the conditions
        elevation_deg = self._elevation * 180/PI
        
        # Initialize additional attenuation
        additional_loss = tf.zeros_like(fspl)
        
        # Apply elevation-dependent additional loss
        # For elevation angles between 5° and 90°
        mask = tf.logical_and(elevation_deg >= 5.0, elevation_deg <= 90.0)
        additional_loss = tf.where(mask,
            -0.7298 * elevation_deg + 66.5727,  # TR 38.811 Table 6.6.2-1
            additional_loss)
        
        # Total path loss
        return fspl + additional_loss

    def _compute_atmospheric_loss(self):
        """Compute atmospheric loss according to TR 38.811
        
        Output
        -------
        : [batch size, num_ut, num_bs], tf.float
            Atmospheric loss [dB]
        """
        # 根据TR 38.811计算大气损耗
        # 这里使用简化模型，实际应该考虑大气成分、压力等
        atmospheric_loss = 0.1 * self._distance_3d/1000  # 0.1 dB/km
        return atmospheric_loss

    def _compute_rain_loss(self):
        """Compute rain loss according to TR 38.811
        
        Output
        -------
        : [batch size, num_ut, num_bs], tf.float
            Rain loss [dB]
        """
        if self.rain_rate > 0:
            # 根据TR 38.811计算雨衰
            # 这里使用简化模型，实际应该考虑频率、极化等
            rain_loss = 0.2 * self.rain_rate * self._distance_3d/1000
        else:
            rain_loss = tf.zeros_like(self._distance_3d)
        return rain_loss

    def _compute_clouds_fog_loss(self):
        """Compute clouds and fog loss according to TR 38.811
        
        Output
        -------
        : [batch size, num_ut, num_bs], tf.float
            Clouds and fog loss [dB]
        """
        # 根据TR 38.811计算云和雾损耗
        # 这里使用简化模型，实际应该考虑云的类型、水含量等
        clouds_loss = 0.05 * self._distance_3d/1000
        return clouds_loss

    def _compute_scintillation_loss(self):
        """Compute scintillation loss according to TR38.811
        
        Returns
        -------
        : [batch_size, num_ut, num_bs], tf.float
            Scintillation loss [dB]
        """
        # Convert elevation to degrees for the calculation
        elevation_deg = self._elevation * 180/PI
        
        # Compute scintillation loss according to TR38.811 Section 6.6.2
        # The formula is: 0.5 * (f_GHz) * (90-elevation_deg)/90
        scintillation_loss = 0.5 * (self.carrier_frequency/1e9) * \
                            (90 - elevation_deg)/90
        
        # Only apply scintillation loss for elevation angles between 5° and 90°
        mask = tf.logical_and(elevation_deg >= 5.0, elevation_deg <= 90.0)
        scintillation_loss = tf.where(mask, scintillation_loss, 0.0)
        
        return scintillation_loss

    def _load_params(self):
        """Load scenario specific parameters
        
        This method loads all the parameters specific to the LEO scenario
        from TR 38.811 specification.
        """
        # 服务链路参数 (Table 6.6.2-1)
        if self.beam_type == "service":
            self.ds_mu = -6.89    # DS log mean [log10(s)]
            self.ds_sigma = 0.54  # DS log std
            self.asd_mu = 0.78    # ASD log mean [log10(degree)]
            self.asd_sigma = 0.36 # ASD log std
            self.asa_mu = 1.05    # ASA log mean [log10(degree)]
            self.asa_sigma = 0.28 # ASA log std
            self.sf_sigma = 8.0   # Shadow fading std [dB]
            self.k_mu = 9.0       # Ricean K-factor mean [dB]
            self.k_sigma = 3.5    # Ricean K-factor std [dB]
            self.zsa_mu = 0.92    # ZSA log mean [log10(degree)]
            self.zsa_sigma = 0.41 # ZSA log std
            self.zsd_mu = 0.88    # ZSD log mean [log10(degree)]
            self.zsd_sigma = 0.34 # ZSD log std

        # 馈电链路参数 (Table 6.6.2-2)
        else:
            self.ds_mu = -7.39    # DS log mean
            self.ds_sigma = 0.54  # DS log std
            self.asd_mu = 0.78    # ASD log mean
            self.asd_sigma = 0.36 # ASD log std
            self.asa_mu = 1.05    # ASA log mean
            self.asa_sigma = 0.28 # ASA log std
            self.sf_sigma = 6.0   # Shadow fading std
            self.k_mu = 12.0      # Ricean K-factor mean
            self.k_sigma = 3.0    # Ricean K-factor std
            self.zsa_mu = 0.92    # ZSA log mean
            self.zsa_sigma = 0.41 # ZSA log std
            self.zsd_mu = 0.88    # ZSD log mean
            self.zsd_sigma = 0.34 # ZSD log std

        # LSP之间的相关系数
        self.ds_sf_corr = -0.4    # DS vs SF correlation
        self.asd_sf_corr = -0.4   # ASD vs SF correlation
        self.asa_sf_corr = -0.4   # ASA vs SF correlation
        self.sf_k_corr = 0.0      # SF vs K correlation
        self.ds_k_corr = 0.0      # DS vs K correlation
        self.asd_k_corr = 0.0     # ASD vs K correlation
        self.asa_k_corr = 0.0     # ASA vs K correlation
        self.ds_asd_corr = 0.4    # DS vs ASD correlation
        self.ds_asa_corr = 0.4    # DS vs ASA correlation
        self.asd_asa_corr = 0.0   # ASD vs ASA correlation

        # 相关距离
        self.ds_corr_dist = 50    # DS correlation distance [m]
        self.asd_corr_dist = 50   # ASD correlation distance [m]
        self.asa_corr_dist = 50   # ASA correlation distance [m]
        self.sf_corr_dist = 50    # SF correlation distance [m]
        self.k_corr_dist = 50     # K correlation distance [m]
        self.zsa_corr_dist = 50   # ZSA correlation distance [m]
        self.zsd_corr_dist = 50   # ZSD correlation distance [m]

    def __call__(self, batch_size, num_time_samples=1, sampling_frequency=1.):
        """生成LEO场景的信道响应
        
        Parameters
        ----------
        batch_size : int
            批处理大小
        num_time_samples : int
            时间采样点数
        sampling_frequency : float
            采样频率 [Hz]
            
        Returns
        -------
        h : [batch_size, num_rx, num_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
            频域信道响应
        tau : [batch_size, num_rx, num_tx, num_paths=1], tf.float
            路径时延 [s]
        """
        # 打印调试信息
        tf.print("\nDebug LEO channel generation:")
        tf.print("- batch_size:", batch_size)
        tf.print("- ut_loc shape:", tf.shape(self.ut_loc))
        tf.print("- bs_loc shape:", tf.shape(self.bs_loc))
        tf.print("- ut_array.num_ant:", self.ut_array.num_ant)
        tf.print("- bs_array.num_ant:", self.bs_array.num_ant)
        
        # 1. 计算路径损耗
        total_loss = self._compute_total_path_loss()
        tf.print("- total_loss shape:", tf.shape(total_loss))
        
        # 2. 获取系统配置参数
        num_rx = self.ut_loc.shape[1]      # 接收端数量(用户数)
        num_tx = self.bs_loc.shape[1]      # 发射端数量(卫星数)
        num_rx_ant = self.ut_array.num_ant # 接收天线数
        num_tx_ant = self.bs_array.num_ant # 发射天线数
        
        # 如果是双极化天线，天线数量要乘2
        if self.bs_array.polarization == 'dual':
            num_tx_ant *= 2
        if self.ut_array.polarization == 'dual':
            num_rx_ant *= 2
        
        # 3. 生成标准7维信道矩阵
        h = tf.ones([
            batch_size,      # 批次大小
            num_rx,         # 接收端数量
            num_tx,         # 发射端数量
            num_rx_ant,     # 接收天线数
            num_tx_ant,     # 发射天线数
            self.num_ofdm_symbols,    # OFDM符号数
            self.num_subcarriers     # 子载波数
        ], dtype=self.cdtype)  # 使用复数类型
        
        tf.print("- h initial shape:", tf.shape(h))
        
        # 4. 应用路径损耗
        # 扩展维度以匹配信道矩阵维度
        total_loss = tf.expand_dims(total_loss, axis=-1)  # 子载波
        total_loss = tf.expand_dims(total_loss, axis=-1)  # OFDM符号
        total_loss = tf.expand_dims(total_loss, axis=-1)  # 发射天线
        total_loss = tf.expand_dims(total_loss, axis=-1)  # 接收天线
        
        tf.print("- total_loss expanded shape:", tf.shape(total_loss))
        
        # 应用路径损耗到信道矩阵
        h = h * tf.cast(tf.sqrt(total_loss), self.cdtype)
        
        tf.print("- h final shape:", tf.shape(h))
        
        # 5. 生成时延矩阵 (单径)
        tau = tf.zeros([batch_size, num_rx, num_tx, 1], dtype=self.rdtype)
        
        return h, tau
        
    def _compute_total_path_loss(self):
        """计算总路径损耗
        
        包括:
        - 自由空间路径损耗
        - 大气损耗
        - 雨衰
        - 云和雾损耗
        - 闪烁损耗
        
        Returns
        -------
        : [batch_size, num_rx, num_tx], tf.float
            总路径损耗 [线性单位]
        """
        # 基本自由空间路径损耗
        wavelength = SPEED_OF_LIGHT/self.carrier_frequency
        fspl = 20*log10(4*PI*self._distance_3d/wavelength)
        
        # 初始化总损耗
        total_loss_db = fspl
        
        # 添加各种额外损耗
        if self.atmospheric_loss_enabled:
            total_loss_db += self._compute_atmospheric_loss()
            
        if self.rain_loss_enabled:
            total_loss_db += self._compute_rain_loss()
            
        if self.clouds_fog_loss_enabled:
            total_loss_db += self._compute_clouds_fog_loss()
            
        if self.scintillation_loss_enabled:
            total_loss_db += self._compute_scintillation_loss()
            
        # 转换为线性单位
        return tf.pow(10.0, -total_loss_db/20.0)
