"""Class implementing LEO (Low Earth Orbit) scenario from 3GPP TR38.811"""

import tensorflow as tf
import numpy as np

from sionna.phy.utils import log10
from sionna.phy import PI, SPEED_OF_LIGHT
from ..system_level_scenario import SystemLevelScenario

class LEO(SystemLevelScenario):
    r"""
    Class implementing the LEO (Low Earth Orbit) scenario from 3GPP TR38.811.

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    ut_array : PanelArray
        Panel array configuration for the UTs

    bs_array : PanelArray
        Panel array configuration for the LEO satellites

    direction : str
        Link direction. Either "uplink" or "downlink"

    height : float
        Height of the LEO satellite [m]. Must be between 500km and 2000km.
        Defaults to 600km.

    elevation : float
        Minimum elevation angle [deg]. Must be between 10° and 90°.
        Defaults to 30°.

    beam_type : str
        Type of beam. Either "service" or "feeder".
        Defaults to "service".

    enable_pathloss : bool
        If `True`, apply pathloss. Otherwise doesn't.
        Defaults to `True`.

    enable_shadow_fading : bool
        If `True`, apply shadow fading. Otherwise doesn't.
        Defaults to `True`.

    enable_atmospheric_loss : bool
        If `True`, apply atmospheric loss. Otherwise doesn't.
        Defaults to `True`.

    enable_rain_loss : bool
        If `True`, apply rain loss. Otherwise doesn't.
        Defaults to `True`.

    enable_clouds_fog_loss : bool
        If `True`, apply clouds and fog loss. Otherwise doesn't.
        Defaults to `True`.

    enable_scintillation_loss : bool
        If `True`, apply scintillation loss. Otherwise doesn't.
        Defaults to `True`.

    rain_rate : float
        Rain rate [mm/h]. Defaults to 0.

    cloud_height : float
        Cloud height [m]. Defaults to 1000.

    batch_size : int
        Batch size. Defaults to 1.

    dtype : tf.DType
        Datatype for internal calculations and outputs.
        Defaults to tf.complex64.
    """

    def __init__(self,
                 carrier_frequency,
                 ut_array,
                 bs_array,
                 direction,
                 height=500e3,  # 500km default
                 elevation=30.,  # 30° default
                 beam_type="service",
                 enable_pathloss=True,
                 enable_shadow_fading=True,
                 enable_atmospheric_loss=True,
                 enable_rain_loss=True,
                 enable_clouds_fog_loss=True,
                 enable_scintillation_loss=True,
                 rain_rate=0.,
                 cloud_height=1000.,
                 batch_size=1,
                 dtype=tf.complex64):

        # Validate LEO specific parameters
        tf.assert_greater_equal(height, 500e3,
            message="LEO height must be >= 500km")
        tf.assert_less_equal(height, 2000e3,
            message="LEO height must be <= 2000km")
        tf.assert_greater_equal(elevation, 10.,
            message="LEO elevation must be >= 10°")
        tf.assert_less_equal(elevation, 90.,
            message="LEO elevation must be <= 90°")

        super().__init__(carrier_frequency=carrier_frequency,
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction=direction,
                        enable_pathloss=enable_pathloss,
                        enable_shadow_fading=enable_shadow_fading,
                        enable_atmospheric_loss=enable_atmospheric_loss,
                        enable_rain_loss=enable_rain_loss,
                        enable_clouds_fog_loss=enable_clouds_fog_loss,
                        enable_scintillation_loss=enable_scintillation_loss,
                        height=height,
                        elevation=elevation,
                        rain_rate=rain_rate,
                        cloud_height=cloud_height,
                        beam_type=beam_type,
                        batch_size=batch_size,
                        dtype=dtype)

        # Initialize LEO specific parameters
        self._initialize_leo_parameters()

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
            ], self.dtype.real_dtype)

            self._lsp_log_std = tf.constant([
                [0.54],  # DS [log10(s)]
                [0.36],  # ASD [log10(degree)]
                [0.28],  # ASA [log10(degree)]
                [8],     # SF [dB]
                [3.5],   # K [dB]
                [0.41],  # ZSA [log10(degree)]
                [0.34]   # ZSD [log10(degree)]
            ], self.dtype.real_dtype)

        else:  # feeder link parameters (Table 6.6.2-2)
            self._lsp_log_mean = tf.constant([
                [-7.39],  # DS
                [0.78],   # ASD
                [1.05],   # ASA
                [0],      # SF
                [12],     # K
                [0.92],   # ZSA
                [0.88]    # ZSD
            ], self.dtype.real_dtype)

            self._lsp_log_std = tf.constant([
                [0.54],  # DS
                [0.36],  # ASD
                [0.28],  # ASA
                [6],     # SF
                [3],     # K
                [0.41],  # ZSA
                [0.34]   # ZSD
            ], self.dtype.real_dtype)

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

        return tf.cast(params.get(param_name, 0.0), self.dtype.real_dtype)

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
            scintillation_loss = 0.5 * (self.carrier_frequency/1e9) * \
                                (90 - self._elevation_angles)/(90)
            total_pl += scintillation_loss

        if self.shadow_fading_enabled:
            # Shadow fading
            # Standard deviation depends on environment and elevation angle
            sf_std = 8.0  # dB, from TR 38.811
            shadow_fading = tf.random.normal(tf.shape(total_pl),
                                           mean=0.0,
                                           stddev=sf_std,
                                           dtype=self.dtype.real_dtype)
            total_pl += shadow_fading

        return total_pl
