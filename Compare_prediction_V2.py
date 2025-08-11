import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorflow.keras.models import load_model

# 设置 matplotlib 支持中文
rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False


# 反归一化函数
def unnormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


# 归一化函数
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


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


# 预测S参数
def predict_s_params(model, input_params, norm_params):
    # 归一化输入参数
    input_min_vals = norm_params['input_min_vals']
    input_max_vals = norm_params['input_max_vals']
    output_min_vals = norm_params['output_min_vals']
    output_max_vals = norm_params['output_max_vals']

    normalized_input = np.zeros_like(input_params, dtype=float)
    for i in range(len(input_params)):
        normalized_input[i] = normalize(input_params[i], input_min_vals[i], input_max_vals[i])

    # 预测S参数
    normalized_output = model.predict(normalized_input.reshape(1, -1)).flatten()

    # 反归一化输出
    num_groups = 4  # S11_dB, S11_phase, S21_dB, S21_phase
    group_size = len(normalized_output) // num_groups

    # 重塑输出数据
    output_reshaped = normalized_output.reshape(num_groups, group_size)
    unnormalized_output = np.zeros_like(output_reshaped)

    for i in range(num_groups):
        unnormalized_output[i] = unnormalize(output_reshaped[i], output_min_vals[i], output_max_vals[i])

    return unnormalized_output, group_size


# 绘制S参数
def plot_s_params(s_params, freq_points, title="谐振器S参数"):
    s11_db = s_params[0]
    s11_phase = s_params[1]
    s21_db = s_params[2]
    s21_phase = s_params[3]

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


# 比较训练数据和预测数据
def compare_with_training_data(input_params, s_params, group_size):
    # 读取训练数据
    try:
        input_data = np.loadtxt('S_Parameter_input_ADS.txt', delimiter=' ')
        output_data = np.loadtxt('S_Parameter_output_ADS.txt', delimiter=' ')

        # 检查输入参数是否与训练数据完全匹配
        exact_match = False
        match_idx = -1

        for i in range(len(input_data)):
            if np.array_equal(input_data[i], input_params):
                exact_match = True
                match_idx = i
                break

        if not exact_match:
            print("错误：输入参数与训练数据不完全匹配")
            return False

        print(f"找到完全匹配的训练数据，索引: {match_idx}")
        print(f"训练数据输入参数: {input_data[match_idx]}")

        # 提取对应的输出数据
        train_output = output_data[match_idx].reshape(4, group_size)

        # 绘制对比图
        labels = ['S11_dB', 'S11_phase', 'S21_dB', 'S21_phase']

        plt.figure(figsize=(12, 15))
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(train_output[i], 'r-', label=f'训练数据 {labels[i]}')
            plt.plot(s_params[i], 'b--', label=f'预测数据 {labels[i]}')
            plt.title(f'{labels[i]} 训练数据 vs 预测数据')
            plt.xlabel('频率点')
            plt.ylabel(labels[i])
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # 计算误差
        mse = np.mean((train_output - s_params) ** 2)
        print(f"均方误差 (MSE): {mse}")

        for i in range(4):
            param_mse = np.mean((train_output[i] - s_params[i]) ** 2)
            print(f"{labels[i]} MSE: {param_mse}")

        return True
    except Exception as e:
        print(f"比较训练数据时出错: {e}")
        return False


def main():
    # 加载模型和归一化参数
    model_dir = "output_model"
    model, norm_params = load_model_and_params(model_dir)

    if model is None or norm_params is None:
        print("无法继续，请确保模型和归一化参数文件存在")
        return

    # 输入参数值
    default_params = np.array([1565, 3147, 3])
    print(f"使用参数: {default_params}")

    # 创建频率点
    freq_start = 4.221e9
    freq_stop = 5.131e9
    freq_points = np.linspace(freq_start, freq_stop, 911)  # num个频率点

    # 预测S参数
    s_params, group_size = predict_s_params(model, default_params, norm_params)

    # 绘制S参数
    plot_s_params(s_params, freq_points, title="预测的谐振器S参数")

    # 与训练数据比较 - 如果不匹配则退出
    if not compare_with_training_data(default_params, s_params, group_size):
        print("分析终止：输入参数必须与训练数据完全匹配")
        return

    print("分析完成！")


if __name__ == "__main__":
    main()