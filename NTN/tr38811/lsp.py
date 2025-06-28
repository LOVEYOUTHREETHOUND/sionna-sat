"""
Class for sampling large scale parameters (LSPs) and pathloss following the
3GPP TR38.811 specifications for non-terrestrial networks
"""

import tensorflow as tf
from sionna.phy.utils import log10
from sionna.phy import config
from sionna.phy.block import Object

class LSP(Object):
    r"""
    Class for conveniently storing LSPs for NTN scenarios

    Parameters
    -----------
    ds : [batch size, num tx, num rx], `tf.float`
        RMS delay spread [s]

    asd : [batch size, num tx, num rx], `tf.float`
        azimuth angle spread of departure [deg]

    asa : [batch size, num tx, num rx], `tf.float`
        azimuth angle spread of arrival [deg]

    sf : [batch size, num tx, num rx], `tf.float`
        shadow fading

    k_factor : [batch size, num tx, num rx], `tf.float`
        Rician K-factor. Typically higher for NTN due to LOS dominance.

    zsa : [batch size, num tx, num rx], `tf.float`
        Zenith angle spread of arrival [deg]

    zsd: [batch size, num tx, num rx], `tf.float`
        Zenith angle spread of departure [deg]

    atmospheric_loss : [batch size, num tx, num rx], `tf.float`
        Atmospheric loss [dB]

    rain_loss : [batch size, num tx, num rx], `tf.float`
        Rain loss [dB]

    clouds_fog_loss : [batch size, num tx, num rx], `tf.float`
        Clouds and fog loss [dB]

    scintillation_loss : [batch size, num tx, num rx], `tf.float`
        Scintillation loss [dB]
    """
    def __init__(self, ds, asd, asa, sf, k_factor, zsa, zsd,
                 atmospheric_loss, rain_loss, clouds_fog_loss, scintillation_loss):
        super().__init__()
        self.ds = ds
        self.asd = asd
        self.asa = asa
        self.sf = sf
        self.k_factor = k_factor
        self.zsa = zsa
        self.zsd = zsd
        self.atmospheric_loss = atmospheric_loss
        self.rain_loss = rain_loss
        self.clouds_fog_loss = clouds_fog_loss
        self.scintillation_loss = scintillation_loss

class LSPGenerator(Object):
    """
    Sample large scale parameters (LSP) and pathloss for NTN scenarios.

    This class implements LSP generation following TR 38.811 specifications,
    including NTN-specific losses like atmospheric loss, rain loss, etc.

    Parameters
    ----------
    scenario : :class:`~sionna.ntn.tr38811.SystemLevelScenario`
        Scenario used to generate LSPs

    Input
    -----
    None

    Output
    ------
    An `LSP` instance storing realization of LSPs
    """
    def __init__(self, scenario):
        self._scenario = scenario
        super().__init__(precision=scenario.precision)

    def sample_pathloss(self):
        """
        Generate pathlosses [dB] for each space station-UT link.

        Input
        ------
        None

        Output
        -------
        A tensor with shape [batch size, number of space stations, number of UTs] 
        of pathloss [dB] for each space station-UT link
        """
        # Basic free space path loss
        pl_b = self._scenario.basic_pathloss

        # For NTN, we don't use O2I models as in TR38.901
        # Instead we apply NTN-specific losses in _apply_ntn_losses()
        return pl_b

    def __call__(self):
        """Generate LSPs for NTN scenario"""

        # LSPs are assumed to follow a log-normal distribution
        # Generate normal random variables
        s = config.tf_rng.normal(shape=[self._scenario.batch_size,
                                      self._scenario.num_bs,
                                      self._scenario.num_ut, 7],
                                      dtype=self.rdtype)

        # Apply cross-LSP correlation
        s = tf.expand_dims(s, axis=4)
        s = self._cross_lsp_correlation_matrix_sqrt@s
        s = tf.squeeze(s, axis=4)

        # Apply spatial correlation
        s = tf.expand_dims(tf.transpose(s, [0, 1, 3, 2]), axis=3)
        s = tf.matmul(s, self._spatial_lsp_correlation_matrix_sqrt,
                     transpose_b=True)
        s = tf.transpose(tf.squeeze(s, axis=3), [0, 1, 3, 2])

        # Scale LSPs to right mean and variance
        lsp_log_mean = self._scenario.lsp_log_mean
        lsp_log_std = self._scenario.lsp_log_std
        lsp_log = lsp_log_std*s + lsp_log_mean

        # Map to linear domain
        lsp = tf.math.pow(tf.constant(10., self.rdtype), lsp_log)

        # Generate NTN-specific losses
        atmospheric_loss = self._sample_atmospheric_loss()
        rain_loss = self._sample_rain_loss()
        clouds_fog_loss = self._sample_clouds_fog_loss()
        scintillation_loss = self._sample_scintillation_loss()

        # Create LSP object with all parameters
        # Note: For NTN we use higher limits for angle spreads due to larger distances
        lsp = LSP(ds=lsp[:, :, :, 0],
                 asd=tf.math.minimum(lsp[:, :, :, 1], 120.0),  # Increased from 104
                 asa=tf.math.minimum(lsp[:, :, :, 2], 120.0),  # Increased from 104
                 sf=lsp[:, :, :, 3],
                 k_factor=lsp[:, :, :, 4],
                 zsa=tf.math.minimum(lsp[:, :, :, 5], 60.0),  # Increased from 52
                 zsd=tf.math.minimum(lsp[:, :, :, 6], 60.0),  # Increased from 52
                 atmospheric_loss=atmospheric_loss,
                 rain_loss=rain_loss,
                 clouds_fog_loss=clouds_fog_loss,
                 scintillation_loss=scintillation_loss)

        return lsp

    def _sample_atmospheric_loss(self):
        """Sample atmospheric loss according to TR 38.811"""
        # Implementation based on TR 38.811 Section 6.5
        if not self._scenario.atmospheric_loss_enabled:
            return tf.zeros([self._scenario.batch_size,
                           self._scenario.num_bs,
                           self._scenario.num_ut], self.rdtype)

        # Get elevation angles from scenario
        elevation = self._scenario.elevation_angles
        
        # Calculate atmospheric loss based on elevation angle
        # This is a simplified model - actual implementation should follow TR 38.811
        atm_loss = 1.0 / tf.math.sin(elevation)  # Example calculation
        
        return atm_loss

    def _sample_rain_loss(self):
        """Sample rain loss according to TR 38.811"""
        # Implementation based on TR 38.811 Section 6.6
        if not self._scenario.rain_loss_enabled:
            return tf.zeros([self._scenario.batch_size,
                           self._scenario.num_bs,
                           self._scenario.num_ut], self.rdtype)

        # Get rain rate from scenario
        rain_rate = self._scenario.rain_rate
        elevation = self._scenario.elevation_angles
        
        # Calculate rain loss based on rain rate and elevation
        # This is a simplified model - actual implementation should follow TR 38.811
        rain_loss = rain_rate / tf.math.sin(elevation)  # Example calculation
        
        return rain_loss

    def _sample_clouds_fog_loss(self):
        """Sample clouds and fog loss according to TR 38.811"""
        # Implementation based on TR 38.811 Section 6.7
        if not self._scenario.clouds_fog_loss_enabled:
            return tf.zeros([self._scenario.batch_size,
                           self._scenario.num_bs,
                           self._scenario.num_ut], self.rdtype)

        # Get cloud parameters from scenario
        cloud_height = self._scenario.cloud_height
        elevation = self._scenario.elevation_angles
        
        # Calculate clouds/fog loss
        # This is a simplified model - actual implementation should follow TR 38.811
        clouds_loss = cloud_height / tf.math.sin(elevation)  # Example calculation
        
        return clouds_loss

    def _sample_scintillation_loss(self):
        """Sample scintillation loss according to TR 38.811"""
        # Implementation based on TR 38.811 Section 6.8
        if not self._scenario.scintillation_loss_enabled:
            return tf.zeros([self._scenario.batch_size,
                           self._scenario.num_bs,
                           self._scenario.num_ut], self.rdtype)

        # Get relevant parameters from scenario
        elevation = self._scenario.elevation_angles
        frequency = self._scenario.carrier_frequency
        
        # Calculate scintillation loss
        # This is a simplified model - actual implementation should follow TR 38.811
        scint_loss = 0.1 * frequency / tf.math.sin(elevation)  # Example calculation
        
        return scint_loss

    def topology_updated_callback(self):
        """
        Updates internal quantities. Must be called at every update of the
        scenario that changes the state of UTs or their locations.
        """
        # Pre-compute cross-LSP correlation matrix
        self._compute_cross_lsp_correlation_matrix()

        # Pre-compute LSP spatial correlation matrix
        self._compute_lsp_spatial_correlation_sqrt()

    def _compute_cross_lsp_correlation_matrix(self):
        """
        Compute cross-LSP correlation matrices for NTN scenario.
        
        The correlation values are adjusted for NTN characteristics,
        particularly the stronger correlation due to LOS dominance.
        """
        # Implementation similar to TR38.901 but with NTN-specific correlation values
        # Create identity matrix for 7 LSPs
        cross_lsp_corr_mat = tf.eye(7, 7, 
            batch_shape=[self._scenario.batch_size,
                        self._scenario.num_bs,
                        self._scenario.num_ut],
            dtype=self.rdtype)

        # Add correlation parameters from scenario
        def _add_param(mat, parameter_name, m, n):
            mask = tf.scatter_nd([[m, n], [n, m]],
                               tf.constant([1.0, 1.0], self.rdtype), [7, 7])
            mask = tf.reshape(mask, [1,1,1,7,7])
            update = self._scenario.get_param(parameter_name)
            update = tf.expand_dims(tf.expand_dims(update, axis=3), axis=4)
            mat = mat + update*mask
            return mat

        # Fill correlation matrix with NTN-specific values
        # Note: These correlations should be adjusted based on TR 38.811
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsDS', 0, 1)
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsDS', 0, 2)
        # Add other correlations...

        # Compute and store matrix square root
        self._cross_lsp_correlation_matrix_sqrt = tf.linalg.cholesky(
            cross_lsp_corr_mat)

    def _compute_lsp_spatial_correlation_sqrt(self):
        """
        Compute spatial correlation matrices for LSPs in NTN scenario.
        
        The correlation distances are adjusted for NTN characteristics,
        particularly the larger distances involved.
        """
        # Implementation similar to TR38.901 but with NTN-specific correlation distances
        # Create correlation matrices for each LSP
        filtering_matrices = []
        distance_scaling_matrices = []
        
        # Get correlation distances for each LSP
        for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                             'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
            
            filtering_matrix = tf.eye(self._scenario.num_ut,
                self._scenario.num_ut,
                batch_shape=[self._scenario.batch_size, self._scenario.num_bs],
                dtype=self.rdtype)
                
            # Get correlation distance from scenario
            distance_scaling_matrix = self._scenario.get_param(parameter_name)
            distance_scaling_matrix = tf.tile(
                tf.expand_dims(distance_scaling_matrix, axis=3),
                [1, 1, 1, self._scenario.num_ut])
            distance_scaling_matrix = -1./distance_scaling_matrix
            
            # For NTN, we mainly consider LOS state
            filtering_matrix = tf.ones_like(filtering_matrix)
            
            filtering_matrices.append(filtering_matrix)
            distance_scaling_matrices.append(distance_scaling_matrix)
            
        filtering_matrices = tf.stack(filtering_matrices, axis=2)
        distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)

        # Get 3D distances between UTs
        ut_dist_3d = self._scenario.matrix_ut_distance_3d
        ut_dist_3d = tf.expand_dims(tf.expand_dims(ut_dist_3d, axis=1), axis=2)

        # Compute correlation matrix using 3D distances
        spatial_lsp_correlation = (tf.math.exp(
            ut_dist_3d*distance_scaling_matrices)*filtering_matrices)

        # Compute and store matrix square root
        self._spatial_lsp_correlation_matrix_sqrt = tf.linalg.cholesky(
            spatial_lsp_correlation)
