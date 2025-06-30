# NTN/tr38811/system_level_scenario.py

import json
import tensorflow as tf
import numpy as np
from abc import abstractmethod
from sionna.phy import config, SPEED_OF_LIGHT, PI
from sionna.phy.block import Object
from sionna.phy.utils import log10, insert_dims, sample_bernoulli
from sionna.phy.channel.utils import rad_2_deg, wrap_angle_0_360
from sionna.phy.channel import ChannelModel
from sionna.phy.channel.tr38901 import PanelArray

class SystemLevelScenario(Object):
    r"""
    This class is used to set up the scenario for system level 3GPP NTN channel
    simulation based on TR 38.811.

    Scenarios for NTN system level channel simulation, such as LEO, MEO, GEO, and HAPS,
    are defined by implementing this base class.

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
        Panel array configuration used by BSs (satellites/HAPS)

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
                 precision=None):
        
        super().__init__(precision=precision)

        # Basic parameters
        self._carrier_frequency = tf.constant(carrier_frequency, self.rdtype)
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency, self.rdtype)
        self._height = tf.constant(height, self.rdtype)
        self._elevation = tf.constant(elevation * PI / 180.0, self.rdtype)  # Convert to radians

        # Arrays
        assert isinstance(ut_array, PanelArray), "'ut_array' must be an instance of PanelArray"
        assert isinstance(bs_array, PanelArray), "'bs_array' must be an instance of PanelArray"
        self._ut_array = ut_array
        self._bs_array = bs_array

        # Direction
        assert direction in ("uplink", "downlink"), "'direction' must be 'uplink' or 'downlink'"
        self._direction = direction

        # Loss components
        self._enable_pathloss = enable_pathloss
        self._enable_atmospheric_loss = enable_atmospheric_loss
        self._enable_rain_loss = enable_rain_loss
        self._enable_clouds_fog_loss = enable_clouds_fog_loss
        self._enable_scintillation_loss = enable_scintillation_loss

        # Topology
        self._ut_loc = None
        self._bs_loc = None
        self._bs_virtual_loc = None
        self._ut_orientations = None
        self._bs_orientations = None
        self._ut_velocities = None
        self._los = None  # NTN scenarios are typically LOS-dominant

        # Load parameters for this scenario
        self._load_params()

    @property
    def carrier_frequency(self):
        """Carrier frequency [Hz]"""
        return self._carrier_frequency

    @property
    def height(self):
        """Platform height [m]"""
        return self._height

    @property
    def elevation(self):
        """Platform elevation angle [rad]"""
        return self._elevation

    @property
    def direction(self):
        """Direction of communication"""
        return self._direction

    @property
    def pathloss_enabled(self):
        """True if pathloss is enabled"""
        return self._enable_pathloss

    @property
    def atmospheric_loss_enabled(self):
        """True if atmospheric loss is enabled"""
        return self._enable_atmospheric_loss

    @property
    def rain_loss_enabled(self):
        """True if rain loss is enabled"""
        return self._enable_rain_loss

    @property
    def clouds_fog_loss_enabled(self):
        """True if clouds and fog loss is enabled"""
        return self._enable_clouds_fog_loss

    @property
    def scintillation_loss_enabled(self):
        """True if scintillation loss is enabled"""
        return self._enable_scintillation_loss

    @property
    def lambda_0(self):
        """Wavelength [m]"""
        return self._lambda_0

    @property
    def batch_size(self):
        """Batch size"""
        return tf.shape(self._ut_loc)[0]

    @property
    def num_ut(self):
        """Number of UTs"""
        return tf.shape(self._ut_loc)[1]

    @property
    def num_bs(self):
        """Number of BSs (satellites/HAPS)"""
        return tf.shape(self._bs_loc)[1]

    @property
    def ut_loc(self):
        """Locations of UTs [m]. [batch_size, num_ut, 3]"""
        return self._ut_loc

    @property
    def bs_loc(self):
        """Locations of BSs [m]. [batch_size, num_bs, 3]"""
        return self._bs_loc

    @property
    def bs_virtual_loc(self):
        """Virtual locations of BSs for wraparound"""
        return self._bs_virtual_loc

    @property
    def ut_orientations(self):
        """Orientations of UTs [rad]. [batch_size, num_ut, 3]"""
        return self._ut_orientations

    @property
    def bs_orientations(self):
        """Orientations of BSs [rad]. [batch_size, num_bs, 3]"""
        return self._bs_orientations

    @property
    def ut_velocities(self):
        """UTs velocities [m/s]. [batch_size, num_ut, 3]"""
        return self._ut_velocities

    @property
    def ut_array(self):
        """PanelArray used by UTs"""
        return self._ut_array

    @property
    def bs_array(self):
        """PanelArray used by BSs"""
        return self._bs_array

    @property
    def los(self):
        """LOS state. In NTN, typically always True"""
        return self._los

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
            Virtual BS locations for wraparound
        """
        if ut_loc is not None:
            self._ut_loc = tf.cast(ut_loc, self.rdtype)
        if bs_loc is not None:
            self._bs_loc = tf.cast(bs_loc, self.rdtype)
        if ut_orientations is not None:
            self._ut_orientations = tf.cast(ut_orientations, self.rdtype)
        if bs_orientations is not None:
            self._bs_orientations = tf.cast(bs_orientations, self.rdtype)
        if ut_velocities is not None:
            self._ut_velocities = tf.cast(ut_velocities, self.rdtype)
        if los is not None:
            self._los = tf.cast(los, tf.bool)
        if bs_virtual_loc is not None:
            self._bs_virtual_loc = tf.cast(bs_virtual_loc, self.rdtype)

        # Update dependent quantities
        self._compute_distances_and_angles()

    @abstractmethod
    def _compute_pathloss_basic(self):
        """Compute basic path loss according to specific scenario"""
        pass

    @abstractmethod
    def _compute_atmospheric_loss(self):
        """Compute atmospheric loss according to specific scenario"""
        pass

    @abstractmethod
    def _compute_rain_loss(self):
        """Compute rain loss according to specific scenario"""
        pass

    @abstractmethod
    def _compute_clouds_fog_loss(self):
        """Compute clouds and fog loss according to specific scenario"""
        pass

    @abstractmethod
    def _compute_scintillation_loss(self):
        """Compute scintillation loss according to specific scenario"""
        pass

    @abstractmethod
    def _load_params(self):
        """Load scenario-specific parameters"""
        pass

    def _compute_distances_and_angles(self):
        """Compute distances and angles between UTs and BSs"""
        # Implementation will be added later
        pass