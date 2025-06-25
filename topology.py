import tensorflow as tf
import numpy as np
from sionna.phy.utils import insert_dims, scalar_to_shaped_tensor, flatten_dims, sample_bernoulli
from sionna.phy import PI, config, dtypes
from sionna.phy.channel.utils import random_ut_properties, set_3gpp_scenario_parameters
from sionna.sys.topology import HexGrid, gen_hexgrid_topology

class CustomHexGrid(HexGrid):
    """扩展的六边形网格类,支持自定义基站位置
    
    继承自sionna.sys.topology.HexGrid,添加了自定义基站位置的功能
    
    Parameters
    ----------
    num_rings : int
        网格环数
        
    cell_radius : float | None (default)
        每个六边形小区的半径,定义为小区中心到任意角的距离
        
    cell_height : float (default: 0.)
        小区高度[m]
        
    isd : float | None (default)
        站间距。必须指定cell_radius或isd其中之一
        
    center_loc : [2], list | tuple (default: (0,0))
        网格中心坐标
        
    center_loc_type : 'offset' (default) | 'axial' | 'euclid'
        center_coord的坐标类型
        
    custom_bs_positions : dict | None (default)
        自定义基站位置字典,格式为:
        {cell_index: (x,y,z)}
        其中cell_index为小区索引,每个小区指定一个基站的位置
        
    precision : None (default) | "single" | "double"
        内部计算和输出使用的精度
    """
    def __init__(self,
                 num_rings,
                 cell_radius=None,
                 cell_height=0.,
                 isd=None,
                 center_loc=(0, 0),
                 center_loc_type='offset',
                 custom_bs_positions=None,
                 precision=None):
        # 在调用父类__init__之前先初始化_custom_bs_positions
        self._custom_bs_positions = custom_bs_positions
        self._original_cell_loc = None
        
        super().__init__(num_rings=num_rings,
                        cell_radius=cell_radius,
                        cell_height=cell_height,
                        isd=isd,
                        center_loc=center_loc,
                        center_loc_type=center_loc_type,
                        precision=precision)
        
    @property
    def cell_loc(self):
        """
        [num_cells, 3], float : 基站位置的欧几里得坐标[m]
        如果指定了custom_bs_positions,则返回自定义的基站位置
        否则返回默认的小区中心位置
        """
        if self._custom_bs_positions is None:
            # 使用默认的小区中心位置
            cell_loc = tf.convert_to_tensor([cell.coord_euclid
                                           for _, cell in self.grid.items()],
                                          dtype=self.rdtype)
            cell_height = tf.fill([cell_loc.shape[0], 1], self.cell_height)
            cell_loc = tf.concat([cell_loc, cell_height], axis=-1)
            self._original_cell_loc = cell_loc
            return cell_loc
        else:
            # 使用自定义基站位置
            bs_positions = []
            original_positions = []
            for cell_idx in range(len(self.grid)):
                cell = self.grid[cell_idx]
                # 保存原始小区中心位置
                original_pos = [cell.coord_euclid[0], cell.coord_euclid[1], self.cell_height]
                original_positions.append(original_pos)
                
                if cell_idx in self._custom_bs_positions:
                    # 使用自定义位置
                    bs_positions.append(self._custom_bs_positions[cell_idx])
                else:
                    # 使用默认小区中心位置
                    bs_positions.append(original_pos)
            
            # 保存原始小区中心位置用于用户放置
            self._original_cell_loc = tf.cast(original_positions, self.rdtype)
            # 返回基站位置
            return tf.cast(bs_positions, self.rdtype)

    def call(self,
             batch_size,
             num_ut_per_sector,
             min_bs_ut_dist,
             max_bs_ut_dist=None,
             min_ut_height=None,
             max_ut_height=None):
        """重写call方法以使用原始小区中心位置进行用户放置"""
        
        # 使用原始小区中心位置进行用户放置计算
        cell_loc_bcast = insert_dims(self._original_cell_loc, num_dims=1, axis=0)
        cell_loc_bcast = insert_dims(cell_loc_bcast, num_dims=2, axis=2)
        cell_loc_bcast = tf.cast(cell_loc_bcast, self.rdtype)

        # Random angles within half a sector, between [-pi/6; pi/6]
        alpha_half = config.tf_rng.uniform(shape=[batch_size,
                                                 len(self.grid),
                                                 3,  # n. sectors
                                                 num_ut_per_sector],
                                          minval=-PI/6.,
                                          maxval=PI/6.,
                                          dtype=self.rdtype)

        # 其余代码与父类相同
        r_max = tf.cast(self.isd, self.rdtype) / (2*tf.math.cos(alpha_half))
        if max_bs_ut_dist is not None:
            r_max = tf.minimum(r_max, tf.cast(max_bs_ut_dist, self.rdtype))

        r_min = tf.cast(min_bs_ut_dist, self.rdtype)
        distance = config.tf_rng.uniform(shape=[batch_size,
                                               len(self.grid),
                                               3,
                                               num_ut_per_sector],
                                       minval=r_min,
                                       maxval=r_max,
                                       dtype=self.rdtype)

        side = sample_bernoulli([batch_size, len(self.grid), 3, num_ut_per_sector],
                               tf.cast(0.5, self.rdtype),
                               precision=self.precision)
        side = tf.cast(side, self.rdtype)
        side = 2. * side + 1.
        alpha = alpha_half + side * PI/6.

        alpha_offset = tf.cast([0, 2*PI/3, 4*PI/3], self.rdtype)
        alpha_offset = insert_dims(alpha_offset, num_dims=2, axis=0)
        alpha_offset = insert_dims(alpha_offset, num_dims=1, axis=-1)
        alpha = alpha + alpha_offset

        ut_loc = tf.stack([distance * tf.math.cos(alpha),
                          distance * tf.math.sin(alpha)], axis=-1)
        ut_loc = ut_loc + cell_loc_bcast[..., :2]

        ut_height = config.tf_rng.uniform(shape=ut_loc.shape[:-1] + [1],
                                        minval=min_ut_height if min_ut_height is not None else 0.,
                                        maxval=max_ut_height if max_ut_height is not None else 0.,
                                        dtype=self.rdtype)
        ut_loc = tf.concat([ut_loc, ut_height], axis=-1)

        # 计算到所有基站的距离
        ut_loc_bcast = insert_dims(ut_loc, num_dims=2, axis=4)
        mirror_loc_bcast = insert_dims(self.mirror_cell_loc, num_dims=4, axis=0)
        mirror_loc_bcast = tf.tile(mirror_loc_bcast,
                                  multiples=[batch_size,
                                            len(self.grid),
                                            3,
                                            num_ut_per_sector,
                                            1, 1, 1])

        ut_mirror_cells_dist = tf.norm(ut_loc_bcast - tf.cast(mirror_loc_bcast, self.rdtype),
                                     ord='euclidean',
                                     axis=-1)

        wraparound_dist = tf.reduce_min(ut_mirror_cells_dist, axis=-1)
        wraparound_mirror_idx = tf.argmin(ut_mirror_cells_dist, axis=-1)
        mirror_cell_per_ut_loc = tf.gather(mirror_loc_bcast,
                                          wraparound_mirror_idx,
                                          axis=-2,
                                          batch_dims=5)

        return ut_loc, mirror_cell_per_ut_loc, wraparound_dist

def gen_custom_hexgrid_topology(batch_size,
                              num_rings,
                              num_ut_per_sector,
                              scenario,
                              custom_bs_positions=None,
                              min_bs_ut_dist=None,
                              max_bs_ut_dist=None,
                              isd=None,
                              bs_height=None,
                              min_ut_height=None,
                              max_ut_height=None,
                              indoor_probability=None,
                              min_ut_velocity=None,
                              max_ut_velocity=None,
                              downtilt_to_sector_center=True,
                              los=None,
                              return_grid=False,
                              precision=None):
    """生成带有自定义基站位置的六边形网格拓扑
    
    基于sionna.sys.gen_hexgrid_topology扩展,支持自定义基站位置
    
    Parameters
    ----------
    与gen_hexgrid_topology相同,另外添加:
    
    custom_bs_positions : dict | None (default)
        自定义基站位置字典,格式为:
        {cell_index: (x,y,z)}
        其中cell_index为小区索引,每个小区指定一个基站的位置
        
    Returns
    -------
    与gen_hexgrid_topology相同
    """
    # 设置3GPP场景参数
    params = set_3gpp_scenario_parameters(scenario,
                                        min_bs_ut_dist,
                                        isd,
                                        bs_height,
                                        min_ut_height,
                                        max_ut_height,
                                        indoor_probability,
                                        min_ut_velocity,
                                        max_ut_velocity,
                                        precision=precision)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, \
        indoor_probability, min_ut_velocity, max_ut_velocity = params

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # 创建自定义网格
    grid = CustomHexGrid(isd=isd,
                        cell_height=bs_height,
                        num_rings=num_rings,
                        custom_bs_positions=custom_bs_positions,
                        precision=precision)
    num_cells = grid.num_cells

    # 获取基站位置
    bs_loc = grid.cell_loc  # [num_cells, 3]
    # 每个基站重复3次作为3个扇区
    bs_loc = tf.repeat(bs_loc, 3, axis=0)  # [num_cells*3, 3]
    # [batch_size, num_cells*3, 3]
    bs_loc = tf.expand_dims(bs_loc, axis=0)
    bs_loc = tf.tile(bs_loc, [batch_size, 1, 1])

    # BS方向设置
    bs_yaw = tf.tile([tf.constant(PI/3.0, rdtype),
                      tf.constant(PI, rdtype),
                      tf.constant(5.0*PI/3.0, rdtype)], [num_cells])
    bs_yaw = insert_dims(bs_yaw, 1, axis=0)
    bs_yaw = tf.tile(bs_yaw, [batch_size, 1])
    bs_yaw = insert_dims(bs_yaw, 1, axis=-1)

    if downtilt_to_sector_center:
        sector_center = (min_bs_ut_dist + 0.5*isd) * 0.5
        bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    else:
        bs_downtilt = tf.cast(0, rdtype)

    bs_pitch = tf.fill([batch_size, num_cells*3, 1], bs_downtilt)
    bs_roll = tf.zeros([batch_size, num_cells*3, 1], rdtype)
    bs_orientations = tf.concat([bs_yaw, bs_pitch, bs_roll], axis=-1)

    # 放置用户
    ut_loc, bs_virtual_loc, _ = grid(batch_size,
                                    num_ut_per_sector,
                                    min_bs_ut_dist,
                                    max_bs_ut_dist=max_bs_ut_dist,
                                    min_ut_height=min_ut_height,
                                    max_ut_height=max_ut_height)
    
    ut_loc = flatten_dims(ut_loc, num_dims=3, axis=1)
    num_ut = ut_loc.shape[1]

    bs_virtual_loc = flatten_dims(bs_virtual_loc, num_dims=3, axis=1)
    bs_virtual_loc = tf.repeat(bs_virtual_loc, 3, axis=2)
    bs_virtual_loc = tf.transpose(bs_virtual_loc, [0, 2, 1, 3])

    # 用户状态
    ut_orientations, ut_velocities, in_state = \
        random_ut_properties(batch_size,
                           num_ut,
                           indoor_probability,
                           min_ut_velocity,
                           max_ut_velocity,
                           precision=precision)

    if return_grid:
        return ut_loc, bs_loc, ut_orientations, \
            bs_orientations, ut_velocities, in_state, los, bs_virtual_loc, grid
    else:
        return ut_loc, bs_loc, ut_orientations, \
            bs_orientations, ut_velocities, in_state, los, bs_virtual_loc 