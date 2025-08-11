# 导入所需的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
import os            # 用于处理文件路径

#V3为完全版本，完美实现COMSOL数据转换


# 定义一个函数，用于读取CSV文件并将其转换为DataFrame
def read_csv_to_dataframe(file_path):
    try:
        # 使用pandas的read_csv函数读取CSV文件
        df = pd.read_csv(file_path)
        # 打印数据的前几行，以便查看数据样本
        print("数据样本:")
        print(df.head())
        # 返回读取的DataFrame
        return df
    except Exception as e:
        # 如果读取文件时发生错误，打印错误信息并返回None
        print(f"读取CSV文件失败: {e}")
        return None

# 定义一个函数，用于处理DataFrame中的数据
def process_data(df):
    # 如果DataFrame为空，打印提示信息并返回None
    if df is None:
        print("数据为空，无法处理。")
        return None, None

    # 获取频率列的列名（假设频率列的列名为'freq (GHz)'）
    freq_column = 'freq (GHz)'
    # 获取结构参数列的列名（即频率列之前的所有列）
    structure_columns = df.columns[:df.columns.get_loc(freq_column)]
    # 获取性能参数列的列名（即频率列之后的所有列）
    performance_columns = df.columns[df.columns.get_loc(freq_column) + 1:]

    # 获取唯一的输入结构参数组合，并去除重复项
    input_data = df[structure_columns].drop_duplicates().values

    # 准备一个空列表，用于存储输出数据
    output_data = []

    # 遍历每个唯一的输入结构参数组合
    for structure in input_data:
        # 创建一个布尔掩码，用于筛选出当前结构参数的所有行
        mask = (df[structure_columns] == structure).all(axis=1)
        # 使用掩码筛选出对应的行，并按频率列进行排序
        grouped = df[mask].sort_values(by=freq_column)

        # 提取每个性能参数列（S11_dB, S11_phase, S21_dB, S21_phase）的值
        S11_dB_values = grouped['S11_dB (dB)'].values
        S11_phase_values = grouped['S11_phase (deg)'].values
        S21_dB_values = grouped['S21_dB (dB)'].values
        S21_phase_values = grouped['S21_phase (deg)'].values

        # 重新排列输出顺序
        output_values = np.concatenate([S11_dB_values, S11_phase_values, S21_dB_values, S21_phase_values])

        # 将新的输出值添加到输出数据列表中
        output_data.append(output_values)

    # 将输出数据列表转换为NumPy数组
    output_data_combined = np.array(output_data)
    # 返回输入数据和输出数据
    return input_data, output_data_combined

# 定义一个函数，用于将输入数据和输出数据保存到文件中
def save_to_file(input_data, output_data, file_name):
    try:
        # 使用NumPy的savetxt函数将输入数据保存到文本文件中
        np.savetxt(f"{file_name}_input.txt", input_data, delimiter=' ', fmt='%f')
        # 使用NumPy的savetxt函数将输出数据保存到文本文件中
        np.savetxt(f"{file_name}_output.txt", output_data, delimiter=' ', fmt='%f')
        # 打印成功保存文件的提示信息
        print("文件保存成功。")
    except Exception as e:
        # 如果保存文件时发生错误，打印错误信息
        print(f"保存文件时出错: {e}")

# 主程序入口
if __name__ == "__main__":
    # 定义CSV文件的路径
    file_path = 'S_Parameter.csv'
    # 调用read_csv_to_dataframe函数读取CSV文件
    df = read_csv_to_dataframe(file_path)
    # 如果成功读取到数据
    if df is not None:
        # 调用process_data函数处理数据
        input_data, output_data = process_data(df)
        # 如果处理后的输入数据和输出数据都不为空
        if input_data is not None and output_data is not None:
            # 获取文件的基本名称（不包含扩展名）
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            # 调用save_to_file函数将数据保存到文件中
            save_to_file(input_data, output_data, base_name)