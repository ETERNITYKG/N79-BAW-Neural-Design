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

# V7加入并联电感，至此滤波器所有组成部分已构建完成

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
        self.freq_start = 4.3e9  # 起始频率 (Hz)
        self.freq_stop = 5.3e9  # 终止频率 (Hz)
        self.freq_points = 1001  # 频率点数

        # 模型配置
        self.model_dir = "output_model"

        # 电感配置 (单位: nH, 0表示不接电感)
        self.inductors = {
            "L_input": 0,  # 输入端口与第一个谐振器之间的电感
            "L_output": 0 ,  # 最后一个Tee与输出端口之间的电感
            "L_fp1": 7.2297,  # 并联谐振器fp1与接地之间的电感
            "L_fp2": 10,  # 并联谐振器fp2与接地之间的电感
            "L_fp3": 1.8998,  # 并联谐振器fp3与接地之间的电感
            "L_fp4": 3.7515  # 并联谐振器fp4与接地之间的电感
        }

        # 滤波器性能目标
        self.targets = [
            # S11目标 - 在通带内应小于等于-15 dB，权重1.0
            FilterTarget("回波损耗目标", "S11", (4.45, 4.75), -15, "max", weight=1.0),

            # S21目标 - 在通带内应大于等于-3 dB，权重2.0（更重要）
            FilterTarget("插入损耗目标", "S21", (4.55, 4.65), -3, "min", weight=2.0),

            # S21目标 - 在阻带内应小于等于-20 dB，权重1.5
            FilterTarget("阻带抑制目标1", "S21", (4.321, 4.4), -20, "max", weight=1.5),
            FilterTarget("阻带抑制目标2", "S21", (4.8, 5.031), -20, "max", weight=1.5),
        ]

        # 默认谐振器参数 - 可以被优化算法修改
        # 参数顺序: [压电层厚度（im），顶电极厚度（im），面积（kum2）]
        self.resonator_params = {
            # 四个串联谐振器参数
            "fs1": np.array([3533, 1132, 0.4993]),
            "fs2": np.array([3554, 1442, 5.0000]),
            "fs3": np.array([3800, 1597, 4.3349]),
            "fs4": np.array([3576, 1305, 0.2426]),

            # 四个并联谐振器参数
            "fp1": np.array([3800, 1129, 4.8205]),
            "fp2": np.array([3464, 1100, 2.3585]),
            "fp3": np.array([3766, 1319, 2.9315]),
            "fp4": np.array([3800, 1129, 2.5686])
        }

    def get_freq_points(self):
        """获取频率点数组"""
        return np.linspace(self.freq_start, self.freq_stop, self.freq_points)

    def get_all_resonator_params(self):
        """获取所有谐振器参数列表"""
        return [
            self.resonator_params["fs1"],
            self.resonator_params["fp1"],
            self.resonator_params["fs2"],
            self.resonator_params["fp2"],
            self.resonator_params["fs3"],
            self.resonator_params["fp3"],
            self.resonator_params["fs4"],
            self.resonator_params["fp4"]
        ]

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
        model = load_model(model_path, compile = False)
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
            print(f"添加输入电感: {inductors['L_input']} nH")

        if 'L_output' in inductors and inductors['L_output'] > 0:
            L_output = rf.Circuit.SeriesImpedance(freq,
                                                  z0=50,
                                                  Z=1j * 2 * np.pi * freq.f * inductors['L_output'] * 1e-9,
                                                  name='L_output')
            print(f"添加输出电感: {inductors['L_output']} nH")

        # 创建并联电感
        if 'L_fp1' in inductors and inductors['L_fp1'] > 0:
            L_fp1 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp1'] * 1e-9,
                                               name='L_fp1')
            print(f"添加并联电感 L_fp1: {inductors['L_fp1']} nH")

        if 'L_fp2' in inductors and inductors['L_fp2'] > 0:
            L_fp2 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp2'] * 1e-9,
                                               name='L_fp2')
            print(f"添加并联电感 L_fp2: {inductors['L_fp2']} nH")

        if 'L_fp3' in inductors and inductors['L_fp3'] > 0:
            L_fp3 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp3'] * 1e-9,
                                               name='L_fp3')
            print(f"添加并联电感 L_fp3: {inductors['L_fp3']} nH")

        if 'L_fp4' in inductors and inductors['L_fp4'] > 0:
            L_fp4 = rf.Circuit.SeriesImpedance(freq,
                                               z0=50,
                                               Z=1j * 2 * np.pi * freq.f * inductors['L_fp4'] * 1e-9,
                                               name='L_fp4')
            print(f"添加并联电感 L_fp4: {inductors['L_fp4']} nH")

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
    print("成功使用 Circuit 类创建四阶滤波器网络")
    return network


# 绘制滤波器的S参数
def plot_filter_s_params(filter_network, title="滤波器S参数"):
    plt.figure(figsize=(12, 8))

    # 绘制S21 (dB)
    plt.subplot(2, 1, 1)
    plt.plot(filter_network.frequency.f / 1e9, filter_network.s_db[:, 1, 0], 'b-', linewidth=2)
    plt.title(f"{title} - S21")
    plt.xlabel('频率 (GHz)')
    plt.ylabel('S21 (dB)')
    plt.grid(True)

    # 绘制S11 (dB)
    plt.subplot(2, 1, 2)
    plt.plot(filter_network.frequency.f / 1e9, filter_network.s_db[:, 0, 0], 'r-', linewidth=2)
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


# 为优化算法提供的接口
class FilterOptimizer:
    """滤波器优化器类，为优化算法提供接口"""

    def __init__(self, config=None):
        """
        初始化优化器

        参数:
            config (FilterConfig): 滤波器配置对象，如果为None则创建默认配置
        """
        self.config = config if config is not None else FilterConfig()
        self.model = None
        self.norm_params = None
        self.filter_network = None
        self.freq_points = self.config.get_freq_points()

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载模型和归一化参数"""
        self.model, self.norm_params = load_model_and_params(self.config.model_dir)
        if self.model is None or self.norm_params is None:
            raise ValueError("无法加载模型或归一化参数")

    def _predict_all_resonators(self, resonator_params=None):
        """
        预测所有谐振器的S参数

        参数:
            resonator_params (list): 谐振器参数列表，如果为None则使用配置中的默认参数

        返回:
            list: 谐振器网络对象列表
        """
        if resonator_params is None:
            resonator_params = self.config.get_all_resonator_params()

        resonator_names = self.config.get_resonator_names()
        resonator_networks = []

        # 顺序处理每个谐振器
        for params, name in zip(resonator_params, resonator_names):
            try:
                s_params, s11_db, s11_phase, s21_db, s21_phase = predict_resonator_s_params(
                    self.model, params, self.norm_params, self.freq_points
                )
                print(f"预测谐振器 {name} 的S参数完成")

                # 创建谐振器网络
                network = create_resonator_network(s_params, self.freq_points, name=f"{name}_Resonator")
                resonator_networks.append(network)
            except Exception as e:
                print(f"预测谐振器 {name} 时出错: {e}")
                return None

        return resonator_networks

    def evaluate_params(self, param_dict=None, inductor_dict=None):
        """
        评估给定参数的滤波器性能

        参数:
            param_dict (dict): 包含谐振器名称和参数的字典，如果为None则使用配置中的默认参数
            inductor_dict (dict): 包含电感名称和值的字典，如果为None则使用配置中的默认电感值

        返回:
            float: 总加权均方误差
            int: 总违反点数
            dict: 详细评估结果
        """
        # 如果提供了新参数，更新配置
        if param_dict is not None:
            self.config.update_resonator_params(param_dict)

        # 如果提供了新的电感参数，更新配置
        if inductor_dict is not None:
            self.config.update_inductor_params(inductor_dict)

        # 预测所有谐振器的S参数
        resonator_networks = self._predict_all_resonators()
        if resonator_networks is None or len(resonator_networks) < 8:
            print("错误: 无法创建足够的谐振器网络")
            return float('inf'), float('inf'), None

        # 创建滤波器网络，传入电感参数
        self.filter_network = create_filter_network(
            resonator_networks,
            self.freq_points,
            inductors=self.config.inductors  # 这里传入电感参数
        )

        # 评估滤波器性能
        total_mse, total_weighted_mse, total_violations, results = evaluate_filter_performance(
            self.filter_network, self.config.targets
        )

        return total_weighted_mse, total_violations, {
            'total_mse': total_mse,
            'results': results,
            'filter_network': self.filter_network
        }

    def plot_current_filter(self):
        """绘制当前滤波器的S参数"""
        if self.filter_network is None:
            print("错误: 尚未创建滤波器网络，请先调用evaluate_params")
            return

        plot_filter_s_params(self.filter_network, title="当前滤波器")

    def print_current_results(self, total_mse, total_weighted_mse, total_violations, results):
        """打印当前评估结果"""
        print_evaluation_results(results, total_mse, total_weighted_mse, total_violations)


def main():
    # 创建配置对象
    config = FilterConfig()

    print(f"使用电感配置: 输入电感={config.inductors['L_input']}nH, 输出电感={config.inductors['L_output']}nH")
    print(f"并联电感配置: L_fp1={config.inductors['L_fp1']}nH, L_fp2={config.inductors['L_fp2']}nH, "
          f"L_fp3={config.inductors['L_fp3']}nH, L_fp4={config.inductors['L_fp4']}nH")

    # 创建优化器
    optimizer = FilterOptimizer(config)

    # 评估默认参数，包括电感
    total_weighted_mse, total_violations, eval_results = optimizer.evaluate_params()

    # 打印评估结果
    if eval_results:
        optimizer.print_current_results(
            eval_results['total_mse'],
            total_weighted_mse,
            total_violations,
            eval_results['results']
        )

        # 绘制滤波器S参数
        optimizer.plot_current_filter()

    print("分析完成！")

    # 记录结束时间并计算耗时
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed_time.total_seconds():.2f} 秒")


if __name__ == "__main__":
    main()