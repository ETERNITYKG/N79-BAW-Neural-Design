import datetime

# 记录开始时间
start_time = datetime.datetime.now()
print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 在导入TensorFlow之前设置日志级别，禁止信息和警告输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部输出, 1=不输出INFO, 2=不输出INFO和WARNING, 3=不输出所有
import tensorflow as tf
from tensorflow.keras.models import load_model
import skrf as rf
from sko.GA import GA # 导入遗传算法
from sko.tools import set_run_mode

# 使用遗传算法进行优化，控制sko库实现多线程优化（win下多线程，Linux自动多进程）

# 设置 matplotlib 支持中文
rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 配置TensorFlow，限制显存使用
# 方法1：仅在需要时分配显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置TensorFlow仅在需要时分配显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # print("已设置TensorFlow按需分配显存")
    except RuntimeError as e:
        print(f"设置TensorFlow显存配置时出错: {e}")

# 方法2：限制TensorFlow使用的显存比例（如果方法1不起作用）
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]  # 限制为1GB显存
# )


# 统一配置类
class FilterConfig:
    """滤波器配置类，集中管理所有配置参数"""

    def __init__(self):
        # 频率配置
        self.freq_start = 4.221e9  # 起始频率 (Hz)
        self.freq_stop = 5.131e9  # 终止频率 (Hz)
        self.freq_points = 911  # 频率点数

        # 模型配置
        self.model_dir = "output_model"

        # 电感配置 (单位: nH, 0表示不接电感)
        self.inductors = {
            "L_input": 1,  # 输入端口与第一个谐振器之间的电感
            "L_output": 1,  # 最后一个Tee与输出端口之间的电感
            "L_fp1": 1,  # 并联谐振器fp1与接地之间的电感
            "L_fp2": 1,  # 并联谐振器fp2与接地之间的电感
            "L_fp3": 1,  # 并联谐振器fp3与接地之间的电感
            "L_fp4": 1  # 并联谐振器fp4与接地之间的电感
        }

        # 滤波器性能目标
        self.targets = [
            # S11目标 - 在通带内应小于等于-15 dB
            FilterTarget("回波损耗目标1", "S11", (4.556, 4.796), -15, "max", weight=5.0),
            FilterTarget("回波损耗目标2", "S11", (4.556, 4.796), -20, "max", weight=1.0),

            # S21目标 - 在通带内应大于等于-3 dB
            FilterTarget("插入损耗目标1", "S21", (4.556, 4.796), -3, "min", weight=5.0),
            FilterTarget("插入损耗目标2", "S21", (4.556, 4.796), -1, "min", weight=1.0),

            # S21目标 - 在阻带内应小于等于-30 dB
            FilterTarget("阻带抑制目标1", "S21", (4.221, 4.506), -30, "max", weight=3),
            FilterTarget("阻带抑制目标2", "S21", (4.846, 5.131), -30, "max", weight=3),
            FilterTarget("阻带抑制目标3", "S21", (4.221, 4.506), -40, "max", weight=1),
            FilterTarget("阻带抑制目标4", "S21", (4.846, 5.131), -40, "max", weight=1),
        ]

        # 默认谐振器参数 - 可以被优化算法修改
        # 参数顺序: [顶电极厚度（im），压电层厚度（im），面积（kum2）]
        self.resonator_params = {
            # 四个串联谐振器参数
            "fs1": np.array([1610, 3147, 1.15]),
            "fs2": np.array([1566, 3147, 1.4]),
            "fs3": np.array([1566, 3147, 9.25]),
            "fs4": np.array([1616, 3147, 1.1]),

            # 四个并联谐振器参数
            "fp1": np.array([1922, 3147, 2.45]),
            "fp2": np.array([1894, 3147, 0.85]),
            "fp3": np.array([1894, 3147, 0.3]),
            "fp4": np.array([1944, 3147, 1.15])
        }

        # --- 新增：优化参数配置 ---
        self.optimization_params = self._define_optimization_params()

    def get_freq_points(self):
        """获取频率点数组"""
        return np.linspace(self.freq_start, self.freq_stop, self.freq_points)


    def get_resonator_names(self):
        """获取所有谐振器名称列表"""
        return ["fs1", "fp1", "fs2", "fp2", "fs3", "fp3", "fs4", "fp4"]

    def update_resonator_params(self, param_dict):
        """
        更新谐振器参数

        参数:
            param_dict (dict): 包含谐振器名称和参数的字典
        """
        for name, params in param_dict.items():
            if name in self.resonator_params:
                self.resonator_params[name] = np.array(params)
            else:
                print(f"警告: 未知的谐振器名称 {name}")

    def update_inductor_params(self, inductor_dict):
        """
        更新电感参数

        参数:
            inductor_dict (dict): 包含电感名称和值的字典
        """
        for name, value in inductor_dict.items():
            if name in self.inductors:
                self.inductors[name] = value
            else:
                print(f"警告: 未知的电感名称 {name}")

    def _define_optimization_params(self):
        """定义所有优化变量的范围 (lb, ub) 和步长 (step)"""
        params = {}
        resonator_names = self.get_resonator_names()
        inductor_names = list(self.inductors.keys())

        # 谐振器参数范围和步长
        # [压电层厚度（im），顶电极厚度（im），面积（kum2）]
        # 变量范围和步长
        resonator_lb = [1500, 3146, 0.1]
        resonator_ub = [1900, 3148, 10]
        resonator_step = [1, 1, 0.05]

        for name in resonator_names:
            params[f"{name}_VA"] = {'lb': resonator_lb[0], 'ub': resonator_ub[0], 'step': resonator_step[0]}
            params[f"{name}_MT"] = {'lb': resonator_lb[1], 'ub': resonator_ub[1], 'step': resonator_step[1]}
            params[f"{name}_Area"] = {'lb': resonator_lb[2], 'ub': resonator_ub[2], 'step': resonator_step[2]}

        # 电感参数范围和步长 (单位: nH)
        # 假设范围和步长，你需要根据实际情况调整
        inductor_lb = 0.0
        inductor_ub = 10.0
        inductor_step = 0.001

        for name in inductor_names:
            params[name] = {'lb': inductor_lb, 'ub': inductor_ub, 'step': inductor_step}

        return params


    def get_optimization_bounds_and_steps(self):
        """获取优化算法所需的lb, ub, step列表"""
        lb = []
        ub = []
        step = []
        param_order = []  # 记录参数顺序，用于后续转换

        resonator_names = self.get_resonator_names()
        inductor_names = list(self.inductors.keys())

        # 确保参数顺序一致
        # 1. 谐振器参数 (VA, MT, Area) * 8
        for name in resonator_names:
            va_key = f"{name}_VA"
            mt_key = f"{name}_MT"
            area_key = f"{name}_Area"
            lb.extend([self.optimization_params[va_key]['lb'], self.optimization_params[mt_key]['lb'],
                       self.optimization_params[area_key]['lb']])
            ub.extend([self.optimization_params[va_key]['ub'], self.optimization_params[mt_key]['ub'],
                       self.optimization_params[area_key]['ub']])
            step.extend([self.optimization_params[va_key]['step'], self.optimization_params[mt_key]['step'],
                         self.optimization_params[area_key]['step']])
            param_order.extend([va_key, mt_key, area_key])

        # 2. 电感参数 * 6
        for name in inductor_names:
            lb.append(self.optimization_params[name]['lb'])
            ub.append(self.optimization_params[name]['ub'])
            step.append(self.optimization_params[name]['step'])
            param_order.append(name)

        return np.array(lb), np.array(ub), np.array(step), param_order



# 归一化函数
def normalize(data, min_val, max_val):
    # 确保输入是NumPy数组
    data_array = np.array(data, dtype=float)
    min_val_array = np.array(min_val, dtype=float)
    max_val_array = np.array(max_val, dtype=float)
    return (data_array - min_val_array) / (max_val_array - min_val_array)


# 反归一化函数
def unnormalize(data, min_val, max_val):
    # 确保输入是NumPy数组
    data_array = np.array(data, dtype=float)
    min_val_array = np.array(min_val, dtype=float)
    max_val_array = np.array(max_val, dtype=float)
    return data_array * (max_val_array - min_val_array) + min_val_array


# 加载模型和归一化参数
def load_model_and_params(model_dir):
    model_path = os.path.join(model_dir, 'final_model.h5')
    norm_params_path = os.path.join(model_dir, 'norm_params.npy')

    try:
        model = load_model(model_path, compile=False)
        norm_params = np.load(norm_params_path, allow_pickle=True).item()
        print(f"模型和归一化参数已成功加载")
        return model, norm_params
    except Exception as e:
        print(f"加载模型或归一化参数时发生错误: {e}")
        return None, None


# 预测谐振器的S参数
def predict_resonator_s_params(model, input_params, norm_params, freq_points):
    # 归一化输入参数
    input_min_vals = np.array(norm_params['input_min_vals'], dtype=float)
    input_max_vals = np.array(norm_params['input_max_vals'], dtype=float)
    output_min_vals = np.array(norm_params['output_min_vals'], dtype=float)
    output_max_vals = np.array(norm_params['output_max_vals'], dtype=float)

    # 向量化归一化操作
    normalized_input = normalize(input_params, input_min_vals, input_max_vals)

    # 预测S参数
    normalized_output = model.predict(normalized_input.reshape(1, -1), verbose=0).flatten()

    # 反归一化输出
    num_groups = 4  # S11_dB, S11_phase, S21_dB, S21_phase
    group_size = len(normalized_output) // num_groups

    # 重塑输出数据
    output_reshaped = normalized_output.reshape(num_groups, group_size)

    # 向量化反归一化操作 - 确保维度匹配
    output_min_vals_reshaped = output_min_vals.reshape(-1, 1)
    output_max_vals_reshaped = output_max_vals.reshape(-1, 1)
    unnormalized_output = unnormalize(output_reshaped, output_min_vals_reshaped, output_max_vals_reshaped)

    # 提取S参数
    s11_db = unnormalized_output[0]
    s11_phase = unnormalized_output[1]
    s21_db = unnormalized_output[2]
    s21_phase = unnormalized_output[3]

    # 转换为复数形式
    s11 = 10 ** (s11_db / 20) * np.exp(1j * np.deg2rad(s11_phase))
    s21 = 10 ** (s21_db / 20) * np.exp(1j * np.deg2rad(s21_phase))

    # 创建S参数矩阵 (对称网络: S11=S22, S21=S12)
    s_params = np.zeros((len(freq_points), 2, 2), dtype=complex)
    s_params[:, 0, 0] = s11  # S11
    s_params[:, 0, 1] = s21  # S12
    s_params[:, 1, 0] = s21  # S21
    s_params[:, 1, 1] = s11  # S22

    return s_params, s11_db, s11_phase, s21_db, s21_phase


# 创建谐振器网络
def create_resonator_network(s_params, freq_points, name="Resonator"):
    # 创建频率对象
    freq = rf.Frequency.from_f(freq_points, unit='Hz')

    # 创建网络对象
    network = rf.Network()
    network.s = s_params
    network.frequency = freq
    network.name = name  # 为网络添加名称

    return network


# 级联谐振器形成滤波器
def create_filter_network(resonators, freq_points, inductors=None):
    """
    创建滤波器网络

    参数:
        resonators: 谐振器网络列表
        freq_points: 频率点数组
        inductors: 电感参数字典，如果为None则不添加电感
    """
    # 创建频率对象
    freq = rf.Frequency.from_f(freq_points, unit='Hz')

    # 定义端口和接地点
    port1 = rf.Circuit.Port(freq, name='Port1', z0=50)
    port2 = rf.Circuit.Port(freq, name='Port2', z0=50)

    # 为每个并联谐振器创建独立的GND节点
    GND1 = rf.Circuit.Ground(freq, name='GND1')
    GND2 = rf.Circuit.Ground(freq, name='GND2')
    GND3 = rf.Circuit.Ground(freq, name='GND3')
    GND4 = rf.Circuit.Ground(freq, name='GND4')

    # 创建理想的Tee节点
    # 使用media.DefinedGammaZ0创建理想的无损Tee
    media = rf.media.DefinedGammaZ0(freq)
    tee1 = media.tee(name='Tee1')
    tee2 = media.tee(name='Tee2')
    tee3 = media.tee(name='Tee3')
    tee4 = media.tee(name='Tee4')

    # 创建电感元件
    L_input = None
    L_output = None
    L_fp1 = None
    L_fp2 = None
    L_fp3 = None
    L_fp4 = None

    if inductors is not None:
        # 只有当电感值大于0时才创建电感元件
        if 'L_input' in inductors and inductors['L_input'] > 0:
            L_input = rf.Circuit.SeriesImpedance(freq,
                                                 z0=50,
                                                 Z=1j * 2 * np.pi * freq.f * inductors['L_input'] * 1e-9,
                                                 name='L_input')
            #print(f"添加输入电感: {inductors['L_input']} nH")

        if 'L_output' in inductors and inductors['L_output'] > 0:
            L_output = rf.Circuit.SeriesImpedance(freq,
                                                  z0=50,
                                                  Z=1j * 2 * np.pi * freq.f * inductors['L_output'] * 1e-9,
                                                  name='L_output')
            #print(f"添加输出电感: {inductors['L_output']} nH")

        # 创建并联电感
        if 'L_fp1' in inductors and inductors['L_fp1'] > 0:
            L_fp1 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp1'] * 1e-9,
                                               name='L_fp1')
            #print(f"添加并联电感 L_fp1: {inductors['L_fp1']} nH")

        if 'L_fp2' in inductors and inductors['L_fp2'] > 0:
            L_fp2 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp2'] * 1e-9,
                                               name='L_fp2')
            #print(f"添加并联电感 L_fp2: {inductors['L_fp2']} nH")

        if 'L_fp3' in inductors and inductors['L_fp3'] > 0:
            L_fp3 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp3'] * 1e-9,
                                               name='L_fp3')
            #print(f"添加并联电感 L_fp3: {inductors['L_fp3']} nH")

        if 'L_fp4' in inductors and inductors['L_fp4'] > 0:
            L_fp4 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp4'] * 1e-9,
                                               name='L_fp4')
            #print(f"添加并联电感 L_fp4: {inductors['L_fp4']} nH")

    # 四阶滤波器连接关系:
    # 若有输入电感 L_input: Port1 -> L_input -> fs1
    # 若无输入电感 L_input: Port1 -> fs1
    # fs1 -> Tee1
    # 若有并联电感 L_fp1: Tee1 -> fp1 -> L_fp1 -> GND1
    # 若无并联电感 L_fp1: Tee1 -> fp1 -> GND1
    # Tee1 -> fs2 -> Tee2
    # 若有并联电感 L_fp2: Tee2 -> fp2 -> L_fp2 -> GND2
    # 若无并联电感 L_fp2: Tee2 -> fp2 -> GND2
    # Tee2 -> fs3 -> Tee3
    # 若有并联电感 L_fp3: Tee3 -> fp3 -> L_fp3 -> GND3
    # 若无并联电感 L_fp3: Tee3 -> fp3 -> GND3
    # Tee3 -> fs4 -> Tee4
    # 若有并联电感 L_fp4: Tee4 -> fp4 -> L_fp4 -> GND4
    # 若无并联电感 L_fp4: Tee4 -> fp4 -> GND4
    # 若有输出电感 L_output: Tee4 -> L_output -> Port2
    # 若无输出电感 L_output: Tee4 -> Port2

    # 定义连接列表
    connexions = []

    # 根据是否有输入电感决定连接方式
    if L_input is not None:
        connexions.extend([
            [(port1, 0), (L_input, 0)],  # Port1 -> L_input
            [(L_input, 1), (resonators[0], 0)]  # L_input -> fs1
        ])
    else:
        connexions.append([(port1, 0), (resonators[0], 0)])  # Port1 -> fs1

    # 中间部分连接 - 根据是否有并联电感决定连接方式
    connexions.extend([
        [(resonators[0], 1), (tee1, 0)],  # fs1 -> Tee1
        [(tee1, 1), (resonators[1], 0)],  # Tee1 -> fp1
    ])

    # fp1与GND1之间的连接，根据是否有并联电感决定
    if L_fp1 is not None:
        connexions.extend([
            [(resonators[1], 1), (L_fp1, 0)],  # fp1 -> L_fp1
            [(L_fp1, 1), (GND1, 0)]  # L_fp1 -> GND1
        ])
    else:
        connexions.append([(resonators[1], 1), (GND1, 0)])  # fp1 -> GND1

    connexions.extend([
        [(tee1, 2), (resonators[2], 0)],  # Tee1 -> fs2
        [(resonators[2], 1), (tee2, 0)],  # fs2 -> Tee2
        [(tee2, 1), (resonators[3], 0)],  # Tee2 -> fp2
    ])

    # fp2与GND2之间的连接，根据是否有并联电感决定
    if L_fp2 is not None:
        connexions.extend([
            [(resonators[3], 1), (L_fp2, 0)],  # fp2 -> L_fp2
            [(L_fp2, 1), (GND2, 0)]  # L_fp2 -> GND2
        ])
    else:
        connexions.append([(resonators[3], 1), (GND2, 0)])  # fp2 -> GND2

    connexions.extend([
        [(tee2, 2), (resonators[4], 0)],  # Tee2 -> fs3
        [(resonators[4], 1), (tee3, 0)],  # fs3 -> Tee3
        [(tee3, 1), (resonators[5], 0)],  # Tee3 -> fp3
    ])

    # fp3与GND3之间的连接，根据是否有并联电感决定
    if L_fp3 is not None:
        connexions.extend([
            [(resonators[5], 1), (L_fp3, 0)],  # fp3 -> L_fp3
            [(L_fp3, 1), (GND3, 0)]  # L_fp3 -> GND3
        ])
    else:
        connexions.append([(resonators[5], 1), (GND3, 0)])  # fp3 -> GND3

    connexions.extend([
        [(tee3, 2), (resonators[6], 0)],  # Tee3 -> fs4
        [(resonators[6], 1), (tee4, 0)],  # fs4 -> Tee4
        [(tee4, 1), (resonators[7], 0)],  # Tee4 -> fp4
    ])

    # fp4与GND4之间的连接，根据是否有并联电感决定
    if L_fp4 is not None:
        connexions.extend([
            [(resonators[7], 1), (L_fp4, 0)],  # fp4 -> L_fp4
            [(L_fp4, 1), (GND4, 0)]  # L_fp4 -> GND4
        ])
    else:
        connexions.append([(resonators[7], 1), (GND4, 0)])  # fp4 -> GND4

    # 根据是否有输出电感决定连接方式
    if L_output is not None:
        connexions.extend([
            [(tee4, 2), (L_output, 0)],  # Tee4 -> L_output
            [(L_output, 1), (port2, 0)]  # L_output -> Port2
        ])
    else:
        connexions.append([(tee4, 2), (port2, 0)])  # Tee4 -> Port2

    # 创建电路对象
    circuit = rf.Circuit(connexions)

    # 计算整体网络的 S 参数
    network = circuit.network
    #print("成功创建四阶滤波器网络")
    return network


# 绘制滤波器的S参数
def plot_filter_s_params(filter_network, title="滤波器S参数"):
    plt.figure(figsize=(12, 8))

    # 获取频率和S参数(dB)
    freq_ghz = filter_network.frequency.f / 1e9
    s21_db = filter_network.s_db[:, 1, 0]
    s11_db = filter_network.s_db[:, 0, 0]

    # --- 将大于0的dB值限制为0 ---
    s21_db_clipped = np.clip(s21_db, None, 0)
    s11_db_clipped = np.clip(s11_db, None, 0)

    # 绘制S21 (dB) - 使用限制后的值
    plt.subplot(2, 1, 1)
    plt.plot(freq_ghz, s21_db_clipped, 'b-', linewidth=2) # 修改：使用 s21_db_clipped
    plt.title(f"{title} - S21")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('S21 (dB)')
    plt.grid(True)

    # 绘制S11 (dB) - 使用限制后的值
    plt.subplot(2, 1, 2)
    plt.plot(freq_ghz, s11_db_clipped, 'r-', linewidth=2) # 修改：使用 s11_db_clipped
    plt.title(f"{title} - S11")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('S11 (dB)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 绘制谐振器的S参数
def plot_resonator_s_params(freq_points, s11_db, s11_phase, s21_db, s21_phase, title="谐振器S参数"):
    plt.figure(figsize=(12, 10))

    # 绘制S11 (dB)
    plt.subplot(2, 2, 1)
    plt.plot(freq_points / 1e9, s11_db, 'r-', linewidth=2)
    plt.title(f"{title} - S11 (dB)")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('S11 (dB)')
    plt.grid(True)

    # 绘制S11 (相位)
    plt.subplot(2, 2, 2)
    plt.plot(freq_points / 1e9, s11_phase, 'r-', linewidth=2)
    plt.title(f"{title} - S11 (相位)")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('相位 (度)')
    plt.grid(True)

    # 绘制S21 (dB)
    plt.subplot(2, 2, 3)
    plt.plot(freq_points / 1e9, s21_db, 'b-', linewidth=2)
    plt.title(f"{title} - S21 (dB)")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('S21 (dB)')
    plt.grid(True)

    # 绘制S21 (相位)
    plt.subplot(2, 2, 4)
    plt.plot(freq_points / 1e9, s21_phase, 'b-', linewidth=2)
    plt.title(f"{title} - S21 (相位)")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('相位 (度)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 添加目标评估相关的类和函数
class FilterTarget:
    """滤波器性能目标类"""

    def __init__(self, name, param_type, freq_range_ghz, target_value, target_type="max", weight=1.0):
        """
        初始化滤波器目标

        参数:
            name (str): 目标名称
            param_type (str): 参数类型，如 'S11', 'S21'
            freq_range_ghz (tuple): 频率范围 (start_ghz, end_ghz)，单位为GHz
            target_value (float): 目标值
            target_type (str): 目标类型，'max'表示参数应小于等于目标值，'min'表示参数应大于等于目标值
            weight (float): 目标权重，默认为1.0，权重越高表示该目标越重要
        """
        self.name = name
        self.param_type = param_type
        self.freq_range_ghz = freq_range_ghz
        self.target_value = target_value
        self.target_type = target_type
        self.weight = weight
        # 频率索引将在计算时根据实际频率确定
        self.freq_indices = None

    def calculate_error(self, network):
        """
        计算当前网络参数与目标之间的加权均方误差

        参数:
            network: 滤波器网络对象

        返回:
            error (float): 加权均方误差
            violations (int): 违反目标的点数
            weighted_error (float): 应用权重后的误差
        """
        # 将频率范围(GHz)转换为频点索引
        freq_ghz = network.frequency.f / 1e9
        start_freq, end_freq = self.freq_range_ghz

        # 找到对应频率范围的索引
        indices = np.where((freq_ghz >= start_freq) & (freq_ghz <= end_freq))[0]

        if len(indices) == 0:
            print(f"警告: 在{start_freq}-{end_freq} GHz范围内未找到频点")
            return 0, 0, 0

        # 保存索引供显示使用
        self.freq_indices = (indices[0], indices[-1])

        # 获取相应的S参数
        if self.param_type == 'S11':
            actual_values = network.s_db[indices, 0, 0]
        elif self.param_type == 'S21':
            actual_values = network.s_db[indices, 1, 0]
        else:
            raise ValueError(f"不支持的参数类型: {self.param_type}")

        # 根据目标类型计算误差
        if self.target_type == "max":
            # 对于最大值目标，只有当实际值大于目标值时才计算误差
            violations = actual_values > self.target_value
            errors = np.where(violations, actual_values - self.target_value, 0)
        else:  # "min"
            # 对于最小值目标，只有当实际值小于目标值时才计算误差
            violations = actual_values < self.target_value
            errors = np.where(violations, self.target_value - actual_values, 0)

        # 计算均方误差
        mse = np.mean(errors ** 2) if np.any(violations) else 0
        # 计算加权误差
        weighted_mse = mse * self.weight
        num_violations = np.sum(violations)

        return mse, num_violations, weighted_mse

    def __str__(self):
        target_type_str = "小于等于" if self.target_type == "max" else "大于等于"
        freq_indices_str = f" (频点索引: {self.freq_indices})" if self.freq_indices else ""
        weight_str = f", 权重: {self.weight}"
        return f"{self.name}: {self.param_type} 在 {self.freq_range_ghz[0]}-{self.freq_range_ghz[1]} GHz 范围内应{target_type_str} {self.target_value} dB{freq_indices_str}{weight_str}"


def evaluate_filter_performance(filter_network, targets):
    """
    评估滤波器性能

    参数:
        filter_network: 滤波器网络对象
        targets (list): FilterTarget对象列表

    返回:
        total_weighted_mse (float): 总加权均方误差
        total_violations (int): 总违反点数
        results (list): 每个目标的评估结果
    """
    results = []
    total_mse = 0
    total_weighted_mse = 0
    total_violations = 0

    for target in targets:
        mse, violations, weighted_mse = target.calculate_error(filter_network)
        total_mse += mse
        total_weighted_mse += weighted_mse
        total_violations += violations

        results.append({
            'target': target,
            'mse': mse,
            'weighted_mse': weighted_mse,
            'violations': violations
        })

    return total_mse, total_weighted_mse, total_violations, results


def print_evaluation_results(results, total_mse, total_weighted_mse, total_violations):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("滤波器性能评估结果:")
    print("=" * 60)

    for result in results:
        target = result['target']
        mse = result['mse']
        weighted_mse = result['weighted_mse']
        violations = result['violations']

        print(f"\n目标: {target}")
        if violations > 0:
            print(f"  - 均方误差 (MSE): {mse:.6f}")
            print(f"  - 加权均方误差: {weighted_mse:.6f}")
            print(f"  - 违反点数: {violations}")
        else:
            print("  - 已满足目标要求")

    print("\n" + "-" * 60)
    print(f"总均方误差 (Total MSE): {total_mse:.6f}")
    print(f"总加权均方误差 (Total Weighted MSE): {total_weighted_mse:.6f}")
    print(f"总违反点数: {total_violations}")
    print("=" * 60)


class FilterOptimizer:
    """滤波器优化器类，封装评估逻辑"""

    def __init__(self, config: FilterConfig):
        self.config = config
        self.model, self.norm_params = load_model_and_params(config.model_dir)
        if self.model is None or self.norm_params is None:
            raise ValueError("无法加载模型或归一化参数")
        self.freq_points = self.config.get_freq_points()
        self.resonator_names = self.config.get_resonator_names()
        self.inductor_names = list(self.config.inductors.keys())
        self.param_order = self.config.get_optimization_bounds_and_steps()[3]  # 获取参数顺序
        self.eval_count = 0
        self.best_score = float('inf')
        self.best_params_vec = None
        self.best_filter_network = None

    def _vector_to_params(self, x):
        """将优化向量转换为谐振器和电感参数字典"""
        resonator_params_dict = {}
        inductor_params_dict = {}
        current_idx = 0

        # 解析谐振器参数
        for res_name in self.resonator_names:
            va = x[current_idx]
            mt = x[current_idx + 1]
            area = x[current_idx + 2]
            resonator_params_dict[res_name] = np.array([va, mt, area])
            current_idx += 3

        # 解析电感参数
        for ind_name in self.inductor_names:
            inductor_params_dict[ind_name] = x[current_idx]
            current_idx += 1

        return resonator_params_dict, inductor_params_dict

    def evaluate(self, x):
        """
        评估给定参数向量 x 的滤波器性能，用作优化目标函数。

        参数:
            x (array): 优化参数向量，顺序由 param_order 决定。

        返回:
            float: 总加权均方误差。
        """
        self.eval_count += 1
        resonator_params_dict, inductor_params_dict = self._vector_to_params(x)

        # 预测所有谐振器的S参数
        resonator_networks = []
        for name in self.resonator_names:
            params = resonator_params_dict[name]
            try:
                s_params, _, _, _, _ = predict_resonator_s_params(
                    self.model, params, self.norm_params, self.freq_points
                )
                network = create_resonator_network(s_params, self.freq_points, name=f"{name}_Resonator")
                resonator_networks.append(network)
            except Exception as e:
                print(f"错误: 预测谐振器 {name} 时出错: {e}")
                return float('inf')  # 返回一个很大的误差值

        if len(resonator_networks) != len(self.resonator_names):
            print(f"错误: 未能成功创建所有 {len(self.resonator_names)} 个谐振器网络")
            return float('inf')

        # 创建滤波器网络
        try:
            current_filter_network = create_filter_network(
                resonator_networks,
                self.freq_points,
                inductors=inductor_params_dict  # 传入当前电感值
            )
        except Exception as e:
            print(f"错误: 创建滤波器网络时出错: {e}")
            return float('inf')

        # 评估滤波器性能
        total_mse, total_weighted_mse, total_violations, results = evaluate_filter_performance(
            current_filter_network, self.config.targets
        )

        # 记录最佳结果
        if total_weighted_mse < self.best_score:
            self.best_score = total_weighted_mse
            self.best_params_vec = x.copy()
            self.best_filter_network = current_filter_network
            print(f"评估次数: {self.eval_count}, 新最佳分数: {total_weighted_mse:.6f}, 违反点数: {total_violations}")

        elif self.eval_count % 50 == 0:  # 每50次评估打印一次当前状态
            print(f"评估次数: {self.eval_count}, 当前分数: {total_weighted_mse:.6f}, 违反点数: {total_violations}")

        # 优化目标是最小化加权均方误差
        return total_weighted_mse

    def get_best_results(self):
        """获取最佳优化结果"""
        if self.best_params_vec is None:
            return None, None, None, None

        best_resonator_params, best_inductor_params = self._vector_to_params(self.best_params_vec)
        return best_resonator_params, best_inductor_params, self.best_score, self.best_filter_network

    def print_best_params(self):
        """打印最佳参数"""
        best_resonator_params, best_inductor_params, best_score, _ = self.get_best_results()
        if best_resonator_params is None:
            print("尚未找到有效解。")
            return

        print("\n" + "=" * 60)
        print(f"优化完成 - 最佳加权MSE: {best_score:.6f}")
        print("=" * 60)

        print("\n最佳谐振器参数:")
        for name, params in best_resonator_params.items():
            # 假设参数顺序是 VA, MT, Area
            print(f"  {name}: VA={params[0]:.0f}, MT={params[1]:.0f}, Area={params[2]:.4f}")

        print("\n最佳电感参数:")
        for name, value in best_inductor_params.items():
            print(f"  {name}: {value:.4f} nH")

        print("=" * 60)


def main():
    # 记录开始时间
    script_start_time = datetime.datetime.now()
    print(f"脚本启动时间: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 初始化 ---
    config = FilterConfig()
    optimizer = FilterOptimizer(config)

    # 获取优化参数边界和步长
    lb, ub, step, param_order = config.get_optimization_bounds_and_steps()
    n_dim = len(lb)  # 参数维度

    print(f"优化维度: {n_dim}")
    # print("参数顺序:", param_order)
    # print("下界 (lb):", lb)
    # print("上界 (ub):", ub)
    # print("步长 (step):", step)

    # --- 遗传算法优化 ---
    print("\n" + "=" * 60)
    print("开始遗传算法优化 (GA)")
    print("=" * 60)

    ga_start_time = datetime.datetime.now()

    # 设置运行模式为多进程
    set_run_mode(optimizer.evaluate, 'multiprocessing') 

    # 配置遗传算法参数
    ga = GA(func=optimizer.evaluate, # 目标函数
            n_dim=n_dim,         # 变量维度
            size_pop=50,         # 种群大小 (示例值, 可调整)
            max_iter=100,        # 最大迭代次数 (示例值, 可调整)
            prob_mut=0.01,       # 变异概率 (示例值, 可调整)
            lb=lb,               # 变量下界
            ub=ub,               # 变量上界
            precision=step       # 使用 step 作为精度，模拟离散变量
           )

    best_x, best_y = ga.run()

    ga_end_time = datetime.datetime.now()
    ga_time = ga_end_time - ga_start_time

    print(f"\n遗传算法优化完成，耗时: {ga_time.total_seconds():.2f}秒")
    print(f"总评估次数: {optimizer.eval_count}")

    # --- 结果处理 ---
    # 从 optimizer 获取记录的最佳结果（SA本身的best_x/best_y可能不是整个过程中的最优）
    best_resonator_params, best_inductor_params, best_score, best_filter_network = optimizer.get_best_results()

    if best_resonator_params is not None and best_inductor_params is not None:
        output_dir = "output_result"
        os.makedirs(output_dir, exist_ok=True) # 创建文件夹，如果不存在
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 修改文件扩展名为 .txt
        output_file = os.path.join(output_dir, f"best_params_ga_{timestamp}.txt")
        try:
            # 使用 'w' 模式打开文件进行写入
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"优化算法: 遗传算法 (GA)\n")
                f.write(f"最佳分数 (加权MSE): {best_score:.6f}\n")
                f.write(f"保存时间: {timestamp}\n")
                f.write("\n" + "=" * 60 + "\n")
                f.write("最佳谐振器参数:\n")
                f.write("=" * 60 + "\n")
                for name, params in best_resonator_params.items():
                    # 格式化输出谐振器参数
                    f.write(f"  {name}:\n")
                    f.write(f"    压电层厚度 (VA): {params[0]:.0f}\n")
                    f.write(f"    顶电极厚度 (MT): {params[1]:.0f}\n")
                    f.write(f"    面积 (Area): {params[2]:.4f}\n")
                f.write("\n" + "=" * 60 + "\n")
                f.write("最佳电感参数 (nH):\n")
                f.write("=" * 60 + "\n")
                for name, value in best_inductor_params.items():
                    # 格式化输出电感参数
                    f.write(f"  {name}: {value:.4f}\n")
                f.write("=" * 60 + "\n")

            print(f"最佳参数已保存到文本文件: {output_file}")
        except Exception as e:
            print(f"保存最佳参数到文本文件时出错: {e}")

        # --- 保存最佳参数为 NPZ 文件 ---
        npz_output_file = os.path.join(output_dir, f"best_params_ga_{timestamp}.npz")
        try:
            np.savez(npz_output_file,
                     resonator_params=best_resonator_params,
                     inductor_params=best_inductor_params,
                     best_score=np.array(best_score))
            print(f"最佳参数已保存到二进制文件: {npz_output_file}")
        except Exception as e:
            print(f"保存最佳参数到二进制文件时出错: {e}")

    if best_filter_network is not None:
        optimizer.print_best_params()

        # 绘制最佳滤波器响应
        print("\n绘制最佳滤波器响应:")
        plot_filter_s_params(best_filter_network, title=f"优化后的滤波器 (GA - Score: {best_score:.4f})")

        # 绘制收敛曲线 (GA自带)
        plt.figure()
        plt.plot(ga.generation_best_Y, 'b-', linewidth=2) 
        plt.title("遗传算法收敛曲线") 
        plt.xlabel("迭代次数")
        plt.ylabel("最佳加权MSE")
        plt.grid(True)
        plt.show()
    else:
        print("\n优化未找到有效解。")

    # 记录结束时间并计算总耗时
    script_end_time = datetime.datetime.now()
    total_elapsed_time = script_end_time - script_start_time
    print(f"\n脚本结束时间: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"脚本总耗时: {total_elapsed_time.total_seconds():.2f} 秒")


if __name__ == "__main__":
    main()