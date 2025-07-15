[[仿真实验源码]]

sionna 官方文档：[nvlabs.github.io/sionna/](https://nvlabs.github.io/sionna/)

# 框架搭建步骤

1. 根据论文中确定好的参数创建星座，仿真一个 snapshot（10s，包含长度为 1ms 的时隙一共 10000 个），将每个时隙对应的卫星位置进行输出；
   1. 输出为什么形式（ECEF 坐标系下）？表格？以什么形式被 python 所读取？
      - STK 默认导出`Fixed Position`即为 ECEF 坐标
      - 无需额外转换即可直接用于 Python 仿真
   2. 输出时间细粒度如何？其实只需要每个时隙开头对应的每个卫星的位置？
   3. 创建一个==支持灵活配置==的读取卫星位置的类，具体功能应该包括：1、stk 输出应该为每 1ms 一个位置，该类中应该支持定义 time slot 的长度，即支持定义仿真地大致计算量；
2. 创建地面拓扑结构：根据论文中给的两个目标区域，按照给出的 SC 参数半径绘制球面小区。对于两个目标区域内的所有小区分别进行编号并对应给出 SC 中心位置坐标。
   1. 使用 h3 库绘制两个目标区域内的小区（等大的正六边形）（已有代码），两个区域分别编号并给出每个编号对应的小区中心；
   2. 将对应的小区中心转换为地心地固坐标系下坐标：应该有专门的 python 库。输出处理后的对应 map；常用转换工具：pythonfrom pymap3d import ecef2geodetic, geodetic2ecef
   3. 创建一个用户撒点函数：基于 h3 还是不需要基于 h3？主要功能应该为每个 snapshot 调用一次，将用户在 100 米为半径的圆内重新撒点，两个用户位置可以重复。输出为用户编号与用户位置（ECEF）的对应 map；
3. 搞清楚 simulate_slot 函数的调用逻辑与具体实现的功能；修改该函数，实现一个可以复用的时隙仿真器（该函数应该包含每个 time slot 应该进行的仿真步骤，包括 1、按照轮询波束调度规则进行波束调度；2、）。对于该函数应该在函数（类？）中初始化一个计数器，计数器到达 snapshot/time_slot 时需要执行更新网络拓扑结构的函数。
4. 实现==计算 SINR==函数，该函数在 simulate_slot 函数中被调用，应该实现的具体功能应该是：输入为 1、卫星位置矩阵 1800；2、分别的两个目标区域内的 SC 位置矩阵；3、在该时隙内重新初始化的用户位置矩阵；4、波束调度情况（也用矩阵进行表示）；。输出应该为每个用户对应的 SINR？ 该函数定位为一个系统级别的根据当前仿真系统参数计算所有用户 SINR 的可以复用的函数。具体参数应该定义在一个外部的 json 中。
5. 初始化一个外部的参数 json 文件，其中定义了计算文档[ucnuh4fziz5j.feishu.cn/wiki/SBqdwifMaiBtxDkbKYhczfCHnsh?fromScene=spaceOverview](https://ucnuh4fziz5j.feishu.cn/wiki/SBqdwifMaiBtxDkbKYhczfCHnsh?fromScene=spaceOverview)中所有相关的参数以及参数解释；方便进行灵活配置。
6.

# 系统结构设计

|     |     |
| --- | --- |
|     |     |

# 实际操作步骤

1. 根据论文中确定好的参数创建星座，仿真一个 snapshot（10s，包含长度为 1ms 的时隙一共 10000 个），将每个时隙对应的卫星位置进行输出；
   1. 输出为什么形式（ECEF 坐标系下）？表格？以什么形式被 python 所读取？输出为 txt 报告格式；==TODO 应该进一步处理数据，将报告文件处理为表格形式。==
   2. 输出时间细粒度如何？其实只需要每个时隙开头对应的每个卫星的位置？对应的应该就是每个时隙开头。
   3. 创建一个==支持灵活配置==的读取卫星位置的类，具体功能应该包括：1、stk 输出应该为每 1ms 一个位置，该类中应该支持定义 time slot 的长度，即支持定义仿真地大致计算量；==TODO 在主文件中进行实现==

具体导出参数设置：

| ​**参数**​       | ​**推荐值**​           | ​**物理意义**​        | ​**选择依据**​                               |
| ---------------- | ---------------------- | --------------------- | -------------------------------------------- |
| ​**总时长**​     | 15 分钟 (900 秒)       | 1/4 典型 LEO 轨道周期 | 覆盖卫星从进入 → 过顶 → 离开区域的全过程     |
| ​**时间步长**​   | 1 秒                   | 位置更新频率          | 满足快照级更新需求（10 秒/次）同时控制数据量 |
| ​**卫星数量**​   | 1800 颗                | Walker 星座规模       | 完整星座数据                                 |
| ​**数据点总量**​ | 900 × 1800 = 1,620,000 | 位置记录数            | 可管理的数据规模                             |

![[Pasted image 20250715105652.png]]
==stk 中是不是不支持生成步长为 ms 级别的位置报告信息？==

设置生成全星座卫星地心地固位置信息，设置时间为 1 天，时间步长为 1s，生成 txt 如下：
![[Pasted image 20250715115145.png]]

创建一个支持灵活配置读取参数的类：
设计：

2. 创建地面拓扑结构：根据论文中给的两个目标区域，按照给出的 SC 参数半径绘制球面小区。对于两个目标区域内的所有小区分别进行编号并对应给出 SC 中心位置坐标。
   1. 使用 h3 库绘制两个目标区域内的小区（等大的正六边形）（已有代码），两个区域分别编号并给出每个编号对应的小区中心；
   2. 将对应的小区中心转换为地心地固坐标系下坐标：应该有专门的 python 库。输出处理后的对应 map；常用转换工具：pythonfrom pymap3d import ecef2geodetic, geodetic2ecef
   3. 创建一个用户撒点函数：基于 h3 还是不需要基于 h3？主要功能应该为每个 snapshot 调用一次，将用户在 100 米为半径的圆内重新撒点，两个用户位置可以重复。输出为用户编号与用户位置（ECEF）的对应 map；

# 主要存在的难点

1.

# TBD

1. 确定一下 sionna 是否可以使用 gpu 加速；
2. 了解一下 python 中类的初始化以及含义；
3. 地心地固坐标系；
4.

# sionna.phy.block:

## ChannelMatrix 类：当前搭建仿真框架时不需要使用

- ​**功能**​：生成并更新 OFDM 信道矩阵，并应用衰落模型。
- ​**初始化**​：设置资源网格、相干时间、批大小等参数，初始化衰落参数。
- ​**call 方法**​：生成当前批次的 OFDM 信道频率响应。
- ​**update 方法**​：根据时隙更新信道矩阵，如果时隙是相干时间的整数倍，则更新信道矩阵，否则保持原信道矩阵。
- ​**apply_fading 方法**​：应用衰落模型到信道矩阵上，模拟信道衰落。

# 流管理函数  `get_stream_management`

什么叫做流管理？

- ​**功能**​：根据通信方向（下行或上行）配置流管理。
- ​**下行**​：每个发射机（基站）对应多个接收机（用户终端），每个基站服务多个用户。
- ​**上行**​：每个接收机（基站）对应多个发射机（用户终端），每个用户终端独立发射。
- ​**返回**​：流管理对象，用于管理 MIMO 流。

# SINR 计算函数 重点修改

- ​**功能**​：计算信号与干扰加噪声比（SINR）。
- ​**输入**​：发射功率、流管理、噪声功率、方向、信道矩阵等。
- ​**下行**​：使用 RZF（正则化迫零）预编码。
- ​**上行**​：使用单位矩阵预编码（即无预编码）。
- ​**计算**​：使用 LMMSE（线性最小均方误差）后均衡器计算 SINR。
- ​**输出**​：SINR 值，经过整形和转置以匹配维度。

# 估计可达速率函数  `estimate_achievable_rate`

- **功能**​：根据==SINR==估计可达速率（香农容量公式）。
- ​**输入**​：SINR（dB）、OFDM 符号数、子载波数。
- ​**计算**​：将 SINR 从 dB 转换为线性值，然后计算速率（bps/Hz）。
- ​**输出**​：估计的可达速率。
  ==可以使用在当前框架中。==

# 结果历史记录函数

包括初始化历史记录  `init_result_history`、记录结果  `record_results`、清理历史记录  `clean_hist`。

- ​**功能**​：用于在仿真过程中记录和整理结果，如路径损耗、发射功率、SINR 等。

1. 什么叫做初始化历史记录函数？
2. 最后要不要在当前框架中进行复用？

# 系统级仿真器类  `SystemLevelSimulator`

==重点修改部分==

```class SystemLevelSimulator(sionna.phy.Block):
    def __init__(self, batch_size, num_rings, num_ut_per_sector, carrier_frequency, resource_grid, scenario, direction, ut_array, bs_array, bs_max_power_dbm, ut_max_power_dbm, coherence_time, pf_beta=0.98, max_bs_ut_dist=None, min_bs_ut_dist=None, temperature=294, o2i_model='low', average_street_width=20.0, average_building_height=5.0, precision=None):
        # 初始化参数，设置场景、方向、天线阵列等
        # 设置流管理、噪声功率等
        # 设置信道模型和拓扑

    def _setup_channel_model(self, scenario, carrier_frequency, o2i_model, ut_array, bs_array, average_street_width, average_building_height):
        # 根据场景（umi, uma, rma）设置信道模型

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist):
        # 设置拓扑（基站和用户终端的位置、方向等）

    def _reset(self, bler_target, olla_delta_up):
        # 重置仿真状态（如HARQ反馈、SINR反馈等）

    def _group_by_sector(self, tensor):
        # 将张量按扇区分组

    @tf.function(jit_compile=True)
    def call(self, num_slots, alpha_ul, p0_dbm_ul, bler_target, olla_delta_up, mcs_table_index=1, fairness_dl=0, guaranteed_power_ratio_dl=0.5):
        # 执行仿真
        # 初始化历史记录
        # 重置状态
        # 生成信道矩阵
        # 循环仿真每个时隙
        #   更新信道矩阵
        #   应用衰落
        #   估计可达速率
        #   调度用户
        #   计算路径损耗
        #   计算发射功率（上行或下行）
        #   计算SINR
        #   选择MCS（调制编码方案）
        #   解码并更新HARQ反馈
        #   记录结果
        # 返回历史记录
```

# 辅助函数

- `get_cdf`: 计算经验 CDF（累积分布函数）。
- `pairplot`: 绘制多个变量之间的成对关系图。
  ==可以考虑复用==

# 主函数 main

## 主要流程：

1. ​**初始化**​：设置仿真参数，包括场景、天线阵列、资源网格等。
2. ​**创建仿真器**​：初始化系统级仿真器，设置信道模型和拓扑。
3. ​**运行仿真**​：循环多个时隙，每个时隙更新信道、调度用户、计算 SINR、选择 MCS、记录结果。
4. ​**结果处理**​：计算性能指标（如误块率、平均速率等）。
5. ​**可视化**​：绘制 CDF 和成对关系图，分析系统性能。

# 拓展知识

## 计算坐标系

1. 地心地固坐标系（Earth-Centered, Earth-Fixed，ECEF），简称地心坐标系。
2. 地理坐标系统（Geographic Coordinate System，GCS）[1](https://blog.csdn.net/NobodyWu/article/details/81158298#fn:1 "See footnote")，坐标系是地心坐标系，用经纬度表示球面上的点。
3. 世界大地测量系统（World Geodetic System, WGS），比如 WGS84，是一种地理坐标系统，用于全球定位系统（GPS）。
4. 投影坐标系统（Projection Coordinate System，PCS）[2](https://blog.csdn.net/NobodyWu/article/details/81158298#fn:2 "See footnote")，，在二维平面上用米表示位置。
5. 通用横轴墨卡托投影（Universal Transverse Mercator，UTM），是一种投影方法。
6. ECEF 坐标系：![[Pasted image 20250715150706.png]]
