import tensorflow as tf
import numpy as np
from sionna.phy.utils import insert_dims, scalar_to_shaped_tensor, flatten_dims, sample_bernoulli
from sionna.phy import PI, config, dtypes, Block, Object
from sionna.phy.channel.utils import random_ut_properties, set_3gpp_scenario_parameters
from sionna.sys.topology import HexGrid, Hexagon, convert_hex_coord

class CustomHexGrid(HexGrid):
    """扩展的六边形网格类,支持自定义卫星波束位置
    
    继承自sionna.sys.topology.HexGrid,添加了自定义卫星波束位置的功能。
    使用螺旋式布局算法确保六边形波束之间无缝连接。
    
    Parameters
    ----------
    num_rings : int
        网格环数
        
    cell_radius : float | None (default)
        每个六边形波束的半径,定义为波束中心到任意角的距离
        
    cell_height : float (default: 0.)
        卫星高度[m]
        
    isd : float | None (default)
        波束间距。必须指定cell_radius或isd其中之一
        
    center_loc : [2], list | tuple (default: (0,0))
        网格中心坐标
        
    center_loc_type : 'offset' (default) | 'axial' | 'euclid'
        center_coord的坐标类型
        
    custom_bs_positions : dict | None (default)
        自定义卫星波束位置字典,格式为:
        {beam_index: (x,y,z)}
        其中beam_index为波束索引,每个波束指定一个卫星的位置
        
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
        super().__init__(num_rings=num_rings,
                        cell_radius=cell_radius,
                        cell_height=cell_height,
                        isd=isd,
                        center_loc=center_loc,
                        center_loc_type=center_loc_type,
                        precision=precision)
        
        self.custom_bs_positions = custom_bs_positions
        
    def call(self,
             batch_size,
             num_ut_per_beam,
             min_bs_ut_dist,
             max_bs_ut_dist=None,
             min_ut_height=0.,
             max_ut_height=0.):
        """生成波束内的用户位置
        
        Args:
            batch_size: 批处理大小
            num_ut_per_beam: 每个波束的用户数
            min_bs_ut_dist: 最小用户距离
            max_bs_ut_dist: 最大用户距离
            min_ut_height: 最小用户高度
            max_ut_height: 最大用户高度
            
        Returns:
            ut_loc: 用户位置
            mirror_cell_per_ut_loc: 镜像波束位置
            wraparound_dist: 环绕距离
        """
        # 调用父类的call方法生成用户位置
        ut_loc, mirror_cell_per_ut_loc, wraparound_dist = super().call(
            batch_size,
            num_ut_per_beam,
            min_bs_ut_dist,
            max_bs_ut_dist,
            min_ut_height,
            max_ut_height
        )
        
        # 如果提供了自定义卫星位置,使用它们替换默认位置
        if self.custom_bs_positions is not None:
            for beam_idx, pos in self.custom_bs_positions.items():
                self.grid[beam_idx].coord_euclid = tf.constant(pos[:2], dtype=self.rdtype)
                
        return ut_loc, mirror_cell_per_ut_loc, wraparound_dist

def set_satellite_scenario_parameters(min_bs_ut_dist=None,
                                    isd=None,
                                    bs_height=None,
                                    min_ut_height=None,
                                    max_ut_height=None,
                                    indoor_probability=None,
                                    min_ut_velocity=None,
                                    max_ut_velocity=None,
                                    precision=None):
    """为卫星场景设置默认参数
    
    Parameters
    ----------
    min_bs_ut_dist : float | None
        最小卫星-用户距离 [m]
        
    isd : float | None
        波束间距 [m]
        
    bs_height : float | None
        卫星高度 [m]
        
    min_ut_height : float | None
        最小用户高度 [m]
        
    max_ut_height : float | None
        最大用户高度 [m]
        
    indoor_probability : float | None
        室内用户概率
        
    min_ut_velocity : float | None
        最小用户速度 [m/s]
        
    max_ut_velocity : float | None
        最大用户速度 [m/s]
        
    precision : str | None
        计算精度
        
    Returns
    -------
    tuple
        (min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,
         indoor_probability, min_ut_velocity, max_ut_velocity)
    """
    if precision is None:
        rdtype = tf.float32
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
        
    # 设置卫星场景的默认参数
    if min_bs_ut_dist is None:
        min_bs_ut_dist = tf.cast(35.0, rdtype)  # 最小距离35m
        
    if isd is None:
        isd = tf.cast(1000.0, rdtype)  # 波束间距1000m
        
    if bs_height is None:
        bs_height = tf.cast(500000.0, rdtype)  # 卫星高度500km
        
    if min_ut_height is None:
        min_ut_height = tf.cast(0.0, rdtype)  # 地面高度
        
    if max_ut_height is None:
        max_ut_height = tf.cast(10.0, rdtype)  # 最大建筑高度
        
    if indoor_probability is None:
        indoor_probability = tf.cast(0.8, rdtype)  # 80%室内用户
        
    if min_ut_velocity is None:
        min_ut_velocity = tf.cast(0.0, rdtype)  # 静止用户
        
    if max_ut_velocity is None:
        max_ut_velocity = tf.cast(3.0, rdtype)  # 最大3m/s (步行速度)
        
    return (min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,
            indoor_probability, min_ut_velocity, max_ut_velocity)

def gen_custom_hexgrid_topology(batch_size,
                              num_rings,
                              num_ut_per_sector,
                              scenario,
                              min_bs_ut_dist=None,
                              max_bs_ut_dist=None,
                              isd=None,
                              bs_height=None,
                              min_ut_height=None,
                              max_ut_height=None,
                              indoor_probability=None,
                              min_ut_velocity=None,
                              max_ut_velocity=None,
                              los=None,
                              return_grid=False,
                              custom_bs_positions=None,
                              downtilt_to_sector_center=True,
                              precision=None):
    """生成自定义六边形网格拓扑
    
    Parameters
    ----------
    batch_size : int
        批量大小
        
    num_rings : int
        环数
        
    num_ut_per_sector : int
        每个扇区/波束的用户数
        
    scenario : str
        场景类型 ('umi', 'uma', 'rma', 'satellite')
        
    min_bs_ut_dist : float | None
        最小基站/卫星-用户距离 [m]
        
    max_bs_ut_dist : float | None
        最大基站/卫星-用户距离 [m]
        
    isd : float | None
        站间距/波束间距 [m]
        
    bs_height : float | None
        基站高度/卫星高度 [m]
        
    min_ut_height : float | None
        最小用户高度 [m]
        
    max_ut_height : float | None
        最大用户高度 [m]
        
    indoor_probability : float | None
        室内用户概率
        
    min_ut_velocity : float | None
        最小用户速度 [m/s]
        
    max_ut_velocity : float | None
        最大用户速度 [m/s]
        
    los : bool | None
        是否强制LOS
        
    return_grid : bool
        是否返回网格对象
        
    custom_bs_positions : dict | None
        自定义基站/卫星位置
        
    downtilt_to_sector_center : bool
        是否将天线下倾指向扇区中心
        
    precision : str | None
        计算精度
    """
    
    # 根据场景类型选择参数设置函数
    if scenario == 'satellite':
        params = set_satellite_scenario_parameters(
            min_bs_ut_dist,
            isd,
            bs_height,
            min_ut_height,
            max_ut_height,
            indoor_probability,
            min_ut_velocity,
            max_ut_velocity,
            precision=precision
        )
        # 卫星场景不需要下倾
        downtilt_to_sector_center = False
    else:
        params = set_3gpp_scenario_parameters(
            scenario,
            min_bs_ut_dist,
            isd,
            bs_height,
            min_ut_height,
            max_ut_height,
            indoor_probability,
            min_ut_velocity,
            max_ut_velocity,
            precision=precision
        )
    
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

    # 获取基站/卫星位置
    bs_loc = grid.cell_loc  # [num_cells, 3]
    
    # 只在非卫星场景下重复3次作为3个扇区
    if scenario != 'satellite':
        bs_loc = tf.repeat(bs_loc, 3, axis=0)  # [num_cells*3, 3]
        num_sectors = 3
    else:
        num_sectors = 1
        
    # [batch_size, num_cells*num_sectors, 3]
    bs_loc = tf.expand_dims(bs_loc, axis=0)
    bs_loc = tf.tile(bs_loc, [batch_size, 1, 1])

    # BS/卫星方向设置
    if scenario != 'satellite':
        # 地面场景: 3个扇区的方向
        bs_yaw = tf.tile([tf.constant(PI/3.0, rdtype),
                         tf.constant(PI, rdtype),
                         tf.constant(5.0*PI/3.0, rdtype)], [num_cells])
    else:
        # 卫星场景: 指向地面
        bs_yaw = tf.zeros([num_cells], dtype=rdtype)
        
    bs_yaw = insert_dims(bs_yaw, 1, axis=0)
    bs_yaw = tf.tile(bs_yaw, [batch_size, 1])
    bs_yaw = insert_dims(bs_yaw, 1, axis=-1)

    if downtilt_to_sector_center:
        sector_center = (min_bs_ut_dist + 0.5*isd) * 0.5
        bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    else:
        bs_downtilt = tf.cast(0, rdtype)

    bs_pitch = tf.fill([batch_size, num_cells*num_sectors, 1], bs_downtilt)
    bs_roll = tf.zeros([batch_size, num_cells*num_sectors, 1], rdtype)
    bs_orientations = tf.concat([bs_yaw, bs_pitch, bs_roll], axis=-1)

    # 放置用户
    ut_loc, bs_virtual_loc, _ = grid(batch_size,
                                    num_ut_per_sector,
                                    min_bs_ut_dist,
                                    max_bs_ut_dist=max_bs_ut_dist,
                                    min_ut_height=min_ut_height,
                                    max_ut_height=max_ut_height)
    
    # 修改: 根据场景类型决定是否展平维度
    if scenario == 'satellite':
        # 卫星场景: 直接展平用户位置
        ut_loc = tf.reshape(ut_loc, [batch_size, -1, 3])
    else:
        # 地面场景: 保持原有逻辑
        ut_loc = flatten_dims(ut_loc, num_dims=3, axis=1)
    
    num_ut = ut_loc.shape[1]

    bs_virtual_loc = flatten_dims(bs_virtual_loc, num_dims=3, axis=1)
    if scenario != 'satellite':
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