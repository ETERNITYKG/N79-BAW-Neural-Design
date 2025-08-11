import os
# 在导入TensorFlow之前设置日志级别，禁止信息和警告输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部输出, 1=不输出INFO, 2=不输出INFO和WARNING, 3=不输出所有
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf


# V3版本 增加注意力机制

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

# 设置 matplotlib 支持中文
rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重复
#np.random.seed(88)

# === Define the custom weighted MSE loss ===
def make_weighted_mse(weight_ranges, weight_factor, group_size, n_groups=4):
    # Create a weights array for 4*group_size outputs (4 groups of group_size=911)
    weights = np.ones(group_size * n_groups, dtype=np.float32)

    for g in range(n_groups):
        for lo, hi in weight_ranges:
            start = g * group_size + lo
            end = g * group_size + hi
            weights[start:end + 1] = weight_factor  # Inclusive range

    weights_tf = tf.constant(weights)

    # Define the custom loss function
    def weighted_mse(y_true, y_pred):
        sq_error = tf.square(y_true - y_pred)
        weighted_sq_error = sq_error * weights_tf
        return tf.reduce_mean(weighted_sq_error)  # scalar loss

    return weighted_mse

# 归一化函数
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


# 反归一化函数
def unnormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


# 确保目录存在
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 读取数据的函数
def read_data(input_file, output_file):
    # 读取输入数据，指定逗号为分隔符
    input_data = np.loadtxt(input_file, delimiter=' ')
    # 读取输出数据，指定逗号为分隔符
    output_data = np.loadtxt(output_file, delimiter=' ')

    return input_data, output_data


# 归一化输入数据
def normalize_input_data(input_data, min_vals, max_vals):
    normalized_input = np.zeros_like(input_data)
    num_features = input_data.shape[1]  # 假设每行是一个样本，列是不同的特征

    for i in range(num_features):
        feature_data = input_data[:, i]
        min_val = min_vals[i]
        max_val = max_vals[i]
        # 归一化每个特征
        normalized_input[:, i] = normalize(feature_data, min_val, max_val)

    return normalized_input


# 归一化输出数据
def normalize_output_data(output_data, min_vals, max_vals, num_groups, group_size):
    # 每一行的输出包含num_groups*group_size个值，需要分成num_groups组，每组group_size个值

    normalized_output = np.zeros_like(output_data)

    for i in range(num_groups):
        # 切分数据：每组group_size个值
        group_data = output_data[:, i * group_size: (i + 1) * group_size]
        min_val = min_vals[i]
        max_val = max_vals[i]
        # 归一化该组数据
        normalized_output[:, i * group_size: (i + 1) * group_size] = normalize(group_data, min_val, max_val)

    return normalized_output


# 创建BP模型
def create_model(input_dim, output_dim):
    model = Sequential([
        # 输入层
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # 隐藏层
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # 隐藏层
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # 隐藏层
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        # 隐藏层
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        # 隐藏层
        Dense(8192, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # 输出层
        Dense(output_dim, activation='linear')
    ])

    model.summary()
    return model


# 创建一个自定义的回调函数，用于在训练过程中显示进度条
class TQDMProgressBar(Callback):

    def __init__(self, total_epochs):
        super(TQDMProgressBar, self).__init__()
        self.epochs = total_epochs
        self.pbar = tqdm(total=self.epochs, desc='Training Progress')

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()


# 绘制训练和验证损失曲线
def plot_loss_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失 (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制预测值与实际值的对比图
def plot_predictions(model, input_data, output_data, min_input, max_input, min_output, max_output,
                     group_size, dataset_type='训练集', num_samples=2):
    indices = np.random.choice(len(input_data), size=num_samples, replace=False)
    num_performance_params = len(min_output)  # 4组性能参数（S11_dB, S11_phase, S21_dB, S21_phase）

    labels = ['S11_dB', 'S11_phase', 'S21_dB', 'S21_phase']  # 标签名称

    for idx in indices:
        input_sample = input_data[idx].reshape(1, -1)
        actual = output_data[idx].reshape(num_performance_params, group_size)
        predicted = model.predict(input_sample).flatten().reshape(num_performance_params, group_size)

        # 反归一化输出数据
        actual_unnormalized = np.copy(actual)
        predicted_unnormalized = np.copy(predicted)

        for i in range(num_performance_params):
            actual_unnormalized[i] = unnormalize(actual[i], min_output[i], max_output[i])
            predicted_unnormalized[i] = unnormalize(predicted[i], min_output[i], max_output[i])

        # 反归一化输入数据
        input_unnormalized = np.copy(input_sample)
        for i in range(input_sample.shape[1]):
            input_unnormalized[0, i] = unnormalize(input_sample[0, i], min_input[i], max_input[i])

        # 创建子图 - 2列布局：左侧为曲线对比图，右侧为误差柱状图
        fig, axs = plt.subplots(num_performance_params, 2, figsize=(15, 5 * num_performance_params),
                                gridspec_kw={'width_ratios': [2, 1]})

        if num_performance_params == 1:
            axs = [axs]  # 确保 axs 是一个列表

        for i in range(num_performance_params):
            # 左侧：曲线对比图
            axs[i, 0].plot(actual_unnormalized[i], label=f'实际值 {labels[i]}', marker='o')
            axs[i, 0].plot(predicted_unnormalized[i], label=f'预测值 {labels[i]}', marker='x')
            axs[i, 0].set_title(f'{dataset_type}样本 {idx} 的 {labels[i]} 实际值 vs 预测值', fontsize=12)
            axs[i, 0].set_xlabel('频率点')
            axs[i, 0].set_ylabel(labels[i])
            axs[i, 0].legend()
            axs[i, 0].grid(True)

            # 右侧：误差柱状图
            errors = np.abs(actual_unnormalized[i] - predicted_unnormalized[i])
            # 为了可视化效果，只显示部分点的误差
            sample_indices = np.linspace(0, len(errors) - 1, min(100, len(errors))).astype(int)
            axs[i, 1].bar(sample_indices, errors[sample_indices], color='r', alpha=0.7, width=5.0)
            axs[i, 1].set_title(f'{labels[i]} 预测误差', fontsize=12)
            axs[i, 1].set_xlabel('采样点')
            axs[i, 1].set_ylabel('绝对误差')
            axs[i, 1].grid(True, axis='y')

            # 添加平均误差和最大误差标注
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            axs[i, 1].axhline(y=mean_error, color='g', linestyle='--', label=f'平均误差: {mean_error:.4f}')
            axs[i, 1].axhline(y=max_error, color='b', linestyle='--', label=f'最大误差: {max_error:.4f}')
            axs[i, 1].legend()

        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)  # 调整子图间距
        plt.subplots_adjust(hspace=0.3)  # 手动调整子图之间的垂直间距

        print(f"输入数据 (归一化):\n{input_sample}")
        print(f"输入数据 (反归一化):\n{input_unnormalized}")

        plt.show()

        print(f"{dataset_type}样本 {idx} 的实际值与预测值对比:")
        print(f"实际值:\n{actual_unnormalized}")
        print(f"预测值:\n{predicted_unnormalized}")
        print(
            f"平均绝对误差:\n{[np.mean(np.abs(actual_unnormalized[i] - predicted_unnormalized[i])) for i in range(num_performance_params)]}")
        print(
            f"最大绝对误差:\n{[np.max(np.abs(actual_unnormalized[i] - predicted_unnormalized[i])) for i in range(num_performance_params)]}\n")

def main():
    # 输入输出文件路径
    input_file = 'S_Parameter_input_ADS.txt'
    output_file = 'S_Parameter_output_ADS.txt'

    # 手动输入每个特征的最小值和最大值
    # 初始化基础值
    input_min_base = [1500, 3148, 0.1]  # 输入数据的最小值
    input_max_base = [2000, 3158, 10]  # 输入数据的最大值
    output_min_base = [-60, -100, -60, -100]  # 输出数据的最小值
    output_max_base = [0, 100, 0, 100]  # 输出数据的最大值

    # 添加边界余量
    margin = 0.001
    input_min_vals = [x - margin for x in input_min_base]
    input_max_vals = [x + margin for x in input_max_base]
    output_min_vals = [x - margin for x in output_min_base]
    output_max_vals = [x + margin for x in output_max_base]

    # 调用函数读取数据
    input_data, output_data = read_data(input_file, output_file)

    # 打印读取到的数据（或者进行其他处理）
    print("Input Data Shape:", input_data.shape)
    print("Output Data Shape:", output_data.shape)

    # 归一化输入数据
    normalized_input_data = normalize_input_data(input_data, input_min_vals, input_max_vals)
    # print("Normalized Input Data Sample:", normalized_input_data[:5])
    print("Normalized Input Data Shape:", normalized_input_data.shape)

    # 定义输出数据的组数和组大小
    num_groups = 4
    group_size = 911
    # 归一化输出数据
    normalized_output_data = normalize_output_data(output_data, output_min_vals, output_max_vals, num_groups,
                                                   group_size)
    # print("Normalized Output Data Sample:", normalized_output_data[:5])
    print("Normalized Output Data Shape:", normalized_output_data.shape)

    # 划分数据集
    input_train, input_test, output_train, output_test = train_test_split(
        normalized_input_data, normalized_output_data, test_size=0.1)

    print(f"训练集输入大小: {input_train.shape}")
    print(f"训练集输出大小: {output_train.shape}")
    print(f"测试集输入大小: {input_test.shape}")
    print(f"测试集输出大小: {output_test.shape}")
    print("训练集输入 Sample:", input_train[:5])
    print("训练集输出 Sample:", output_train[:5])
    print("测试集输入 Sample:", input_test[:5])
    print("测试集输出 Sample:", output_test[:5])

    # 创建输出目录
    output_model_dir = "output_model"
    ensure_directory(output_model_dir)

    # 构建模型
    model = create_model(input_dim=input_train.shape[1], output_dim=output_train.shape[1])

    # Create the weighted loss function
    loss_fn = make_weighted_mse(weight_ranges=[(200, 680)], weight_factor=3.0, group_size=group_size, n_groups=4)

    # Compile the model with the custom loss
    model.compile(optimizer=Adam(), loss=loss_fn, metrics=['mae'])

    # # 编译模型
    # model.compile(loss='mean_squared_error', optimizer=Adam())

    # 创建检查点回调
    checkpoint_dir = 'checkpoints'
    ensure_directory(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model_weights_epoch_{epoch:04d}.h5")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        period=1000  # 每period个epoch保存一次
    )

    # 训练参数
    max_epochs = 20000  # 根据需要调整
    batch_size = 10000    # 根据数据量调整

    # 创建 tqdm 回调
    tqdm_callback = TQDMProgressBar(total_epochs=max_epochs)

    # 创建早停回调以防训练过长
    early_stop = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True, verbose=1)

    # 训练模型
    model_trained = model.fit(
        input_train, output_train,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split=0.2,
        callbacks=[checkpoint, tqdm_callback, early_stop],
        verbose=0
    )

    # 保存模型和权重
    final_model_path = os.path.join(output_model_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"模型已保存到 {final_model_path}。")

    final_model_weights_path = os.path.join(output_model_dir, 'final_model_weights.h5')
    model.save_weights(final_model_weights_path)
    print(f"权重已保存到 {final_model_weights_path}。")

    # 保存归一化参数
    norm_params = {
        'input_min_vals': input_min_vals,
        'input_max_vals': input_max_vals,
        'output_min_vals': output_min_vals,
        'output_max_vals': output_max_vals
    }
    np.save(os.path.join(output_model_dir, 'norm_params.npy'), norm_params)
    print(f"归一化参数已保存到 {os.path.join(output_model_dir, 'norm_params.npy')}。")

    # 打印训练信息
    trained_epochs = len(model_trained.history['loss'])
    final_loss = model_trained.history['val_loss'][-1]
    print(f"\n训练结束。实际训练的 epoch 数量: {trained_epochs}")
    print(f"最终的验证损失 (val_loss): {final_loss:.4f}")

    # 绘制损失曲线
    plot_loss_curves(model_trained.history)

    # 模型目录
    output_model_dir = "output_model"
    # 加载保存的模型
    final_model_path = os.path.join(output_model_dir, 'final_model.h5')
    try:
        loaded_model = load_model(final_model_path, compile = False)
        # model.compile(loss='binary_crossentropy',
        #               optimizer=Ada,
        #               metrics=[HammingScore])  #HammingScore是自定义的metric,调用自编写loss方法
        # model = keras.models.load_model('model.h5', custom_objects={'HammingScore': HammingScore} )
        print(f"模型已从 {final_model_path} 成功加载。")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")

    # 绘制训练集预测
    plot_predictions(loaded_model, input_train, output_train, input_min_vals, input_max_vals, output_min_vals,
                     output_max_vals, group_size, dataset_type='训练集', num_samples=2)

    # 绘制测试集预测
    plot_predictions(loaded_model, input_test, output_test, input_min_vals, input_max_vals, output_min_vals,
                     output_max_vals, group_size, dataset_type='测试集', num_samples=6)


if __name__ == "__main__":
    main()
