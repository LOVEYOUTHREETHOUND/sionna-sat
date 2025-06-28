"""Base class for implementing system level channel models from 3GPP TR38.811
specification for non-terrestrial networks"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sionna.phy.channel import ChannelModel
from sionna.phy.channel.utils import deg_2_rad
from . import LSPGenerator, RaysGenerator, Topology, ChannelCoefficientsGenerator

class SystemLevelChannel(ChannelModel):
    r"""
    Base class for implementing 3GPP TR 38.811 system level channel models for
    non-terrestrial networks (NTN), such as LEO, MEO, GEO and HAPS.

    Parameters
    -----------
    scenario : SystemLevelScenario
        Scenario for the NTN channel simulation

    always_generate_lsp : bool, optional (default=False)
        If True, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed.

    Input
    -----
    num_time_samples : int
        Number of time samples

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], tf.complex
        Path coefficients

    tau : [batch_size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """
    def __init__(self,
                 scenario,
                 always_generate_lsp=False,
                 precision=None):
        
        super().__init__(precision=scenario.precision)
        
        self._scenario = scenario
        self._lsp_sampler = LSPGenerator(scenario)
        self._ray_sampler = RaysGenerator(scenario)
        self._set_topology_called = False
        
        # Configure arrays based on link direction
        if scenario.direction == "uplink":
            tx_array = scenario.ut_array
            rx_array = scenario.bs_array
        else:  # downlink
            tx_array = scenario.bs_array
            rx_array = scenario.ut_array
            
        self._cir_sampler = ChannelCoefficientsGenerator(
            scenario.carrier_frequency,
            tx_array, rx_array,
            subclustering=True,  # NTN typically has simpler multipath
            precision=self.precision)
            
        # LSP generation control
        self._always_generate_lsp = always_generate_lsp

    def set_topology(self,
                    ut_loc=None,
                    bs_loc=None,
                    ut_orientations=None,
                    bs_orientations=None,
                    ut_velocities=None,
                    los=None,
                    bs_virtual_loc=None):
        """
        Set the network topology for NTN simulation

        Parameters
        ----------
        ut_loc : [batch_size, num_ut, 3], tf.float, optional
            Locations of UTs [m]

        bs_loc : [batch_size, num_bs, 3], tf.float, optional
            Locations of space stations (satellites/HAPS) [m]

        ut_orientations : [batch_size, num_ut, 3], tf.float, optional
            Orientations of UTs [rad]

        bs_orientations : [batch_size, num_bs, 3], tf.float, optional
            Orientations of space stations [rad]

        ut_velocities : [batch_size, num_ut, 3], tf.float, optional
            Velocities of UTs [m/s]

        los : [batch_size, num_ut, num_bs], tf.bool, optional
            LOS state (typically True for NTN)

        bs_virtual_loc : [batch_size, num_bs, num_ut, 3], tf.float, optional
            Virtual locations of space stations for each UT [m]
        """
        # Update scenario topology
        need_for_update = self._scenario.set_topology(ut_loc,
                                                    bs_loc,
                                                    ut_orientations,
                                                    bs_orientations,
                                                    ut_velocities,
                                                    los,
                                                    bs_virtual_loc)
        
        if need_for_update:
            # Update LSP and ray samplers
            self._lsp_sampler.topology_updated_callback()
            self._ray_sampler.topology_updated_callback()
            
            # Sample LSPs if not generating every time
            if not self._always_generate_lsp:
                self._lsp = self._lsp_sampler()
                
        if not self._set_topology_called:
            self._set_topology_called = True

    def __call__(self, num_time_samples, sampling_frequency):
        """Generate channel impulse responses"""
        
        # Sample LSPs if required
        if self._always_generate_lsp:
            lsp = self._lsp_sampler()
        else:
            lsp = self._lsp
            
        # Sample rays (typically simpler for NTN)
        rays = self._ray_sampler(lsp)
        
        # Create topology for CIR generation
        if self._scenario.direction == 'downlink':
            moving_end = 'rx'
            tx_orientations = self._scenario.bs_orientations
            rx_orientations = self._scenario.ut_orientations
        else:  # uplink
            moving_end = 'tx'
            tx_orientations = self._scenario.ut_orientations
            rx_orientations = self._scenario.bs_orientations
            
        topology = Topology(
            velocities=self._scenario.ut_velocities,
            moving_end=moving_end,
            los_aoa=deg_2_rad(self._scenario.los_aoa),
            los_aod=deg_2_rad(self._scenario.los_aod),
            los_zoa=deg_2_rad(self._scenario.los_zoa),
            los_zod=deg_2_rad(self._scenario.los_zod),
            los=self._scenario.los,
            distance_3d=self._scenario.distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations)

        # Get cluster delay spread in ns
        c_ds = self._scenario.get_param("cDS") * 1e-9

        # Handle uplink/downlink differences
        if self._scenario.direction == "uplink":
            # Transpose various parameters for uplink
            aoa, zoa = rays.aoa, rays.zoa
            aod, zod = rays.aod, rays.zod
            rays.aod = tf.transpose(aoa, [0, 2, 1, 3, 4])
            rays.zod = tf.transpose(zoa, [0, 2, 1, 3, 4])
            rays.aoa = tf.transpose(aod, [0, 2, 1, 3, 4])
            rays.zoa = tf.transpose(zod, [0, 2, 1, 3, 4])
            rays.powers = tf.transpose(rays.powers, [0, 2, 1, 3])
            rays.delays = tf.transpose(rays.delays, [0, 2, 1, 3])
            rays.xpr = tf.transpose(rays.xpr, [0, 2, 1, 3, 4])
            
            # Transpose topology parameters
            los_aod, los_aoa = topology.los_aod, topology.los_aoa
            los_zod, los_zoa = topology.los_zod, topology.los_zoa
            topology.los_aoa = tf.transpose(los_aod, [0, 2, 1])
            topology.los_aod = tf.transpose(los_aoa, [0, 2, 1])
            topology.los_zoa = tf.transpose(los_zod, [0, 2, 1])
            topology.los_zod = tf.transpose(los_zoa, [0, 2, 1])
            topology.los = tf.transpose(topology.los, [0, 2, 1])
            c_ds = tf.transpose(c_ds, [0, 2, 1])
            topology.distance_3d = tf.transpose(topology.distance_3d, [0, 2, 1])
            
            # Transpose LSP parameters
            k_factor = tf.transpose(lsp.k_factor, [0, 2, 1])
            atmospheric_loss = tf.transpose(lsp.atmospheric_loss, [0, 2, 1])
            rain_loss = tf.transpose(lsp.rain_loss, [0, 2, 1])
            clouds_fog_loss = tf.transpose(lsp.clouds_fog_loss, [0, 2, 1])
            scintillation_loss = tf.transpose(lsp.scintillation_loss, [0, 2, 1])
        else:
            k_factor = lsp.k_factor
            atmospheric_loss = lsp.atmospheric_loss
            rain_loss = lsp.rain_loss
            clouds_fog_loss = lsp.clouds_fog_loss
            scintillation_loss = lsp.scintillation_loss

        # Generate channel coefficients
        h, delays = self._cir_sampler(num_time_samples, sampling_frequency,
                                    k_factor, rays, topology, c_ds)

        # Apply NTN-specific losses
        h = self._apply_ntn_losses(h, atmospheric_loss, rain_loss,
                                clouds_fog_loss, scintillation_loss)

        # Reshape to match expected output format
        h = tf.transpose(h, [0, 2, 4, 1, 5, 3, 6])
        delays = tf.transpose(delays, [0, 2, 1, 3])

        # Stop gradients
        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)

        return h, delays

    def _apply_ntn_losses(self, h, atmospheric_loss, rain_loss,
                       clouds_fog_loss, scintillation_loss):
        """Apply NTN-specific losses to channel coefficients
        
        Parameters
        ----------
        h : [...], tf.complex
            Channel coefficients
        atmospheric_loss : [...], tf.float
            Atmospheric loss
        rain_loss : [...], tf.float
            Rain loss  
        clouds_fog_loss : [...], tf.float
            Clouds and fog loss
        scintillation_loss : [...], tf.float
            Scintillation loss
            
        Returns
        -------
        h : [...], tf.complex
            Channel coefficients with losses applied
        """
        if self._scenario.pathloss_enabled:
            # Get basic pathloss
            pl_db = self._lsp_sampler.sample_pathloss()
            if self._scenario.direction == 'uplink':
                pl_db = tf.transpose(pl_db, [0, 2, 1])
        else:
            pl_db = tf.constant(0.0, self.rdtype)

        # Apply atmospheric loss if enabled
        if self._scenario.atmospheric_loss_enabled:
            pl_db += atmospheric_loss

        # Apply rain loss if enabled  
        if self._scenario.rain_loss_enabled:
            pl_db += rain_loss

        # Apply clouds/fog loss if enabled
        if self._scenario.clouds_fog_loss_enabled:
            pl_db += clouds_fog_loss

        # Apply scintillation loss if enabled
        if self._scenario.scintillation_loss_enabled:
            pl_db += scintillation_loss

        # Convert from dB to linear and apply to coefficients
        gain = tf.math.pow(tf.constant(10., self.rdtype), -pl_db/20.)
        gain = tf.reshape(gain, tf.concat([tf.shape(gain),
            tf.ones([tf.rank(h)-tf.rank(gain)], tf.int32)], 0))
        h *= tf.complex(gain, tf.constant(0., self.rdtype))

        return h

    def show_topology(self, bs_index=0, batch_index=0):
        """Visualize the NTN network topology
        
        Parameters
        ----------
        bs_index : int, optional
            Index of space station to show (default: 0)
        batch_index : int, optional  
            Batch example to show (default: 0)
        """
        # Implementation similar to TR38.901 but adapted for NTN visualization
        pass
