import tensorflow as tf
from sionna.phy.channel.tr38901 import SystemLevelChannel
from satellite_scenario import SatelliteScenario

class Satellite(SystemLevelChannel):
    """卫星通信信道模型类
    
    继承自TR38.901的SystemLevelChannel,实现卫星特有的信道特性
    
    Parameters
    ----------
    carrier_frequency : float
        载波频率 [Hz]
        
    ut_array : PanelArray
        用户终端天线阵列
        
    bs_array : PanelArray
        卫星天线阵列
        
    direction : str
        'uplink' or 'downlink'
        
    height : float
        卫星高度 [m]
        
    elevation : float
        卫星仰角 [度]
        
    enable_pathloss : bool
        是否启用路径损耗计算
        
    enable_shadow_fading : bool
        是否启用阴影衰落
        
    o2i_model : str
        室内外损耗模型, "low" or "high"
        
    precision : tf.DType
        数据类型
    """
    
    def __init__(self, carrier_frequency, ut_array, bs_array,
                 direction, enable_pathloss=True, enable_shadow_fading=True,
                 beam_type="service", height=500000., elevation=90.,
                 precision=None):

        # 创建场景对象 - 传递所有必需的参数
        scenario = SatelliteScenario(
            carrier_frequency=carrier_frequency,
            ut_array=ut_array,
            bs_array=bs_array,
            direction=direction,
            o2i_model="low",  # 使用默认值
            height=height,
            elevation=elevation,
            precision=precision
        )
        
        # 调用父类初始化 - 只传递父类需要的参数
        super().__init__(
            scenario=scenario,
            always_generate_lsp=False,  # 使用默认值
            precision=precision
        )