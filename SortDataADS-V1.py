import os
import numpy as np
import itertools

#V1为完全版本，完美实现ADS数据转换

def read_input_variables(file_path):
    """读取输入变量，但排除频率变量"""
    input_vars = []
    recording = False
    current_var = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == 'VAR_LIST_BEGIN':
                recording = True
                current_var = []
            elif line == 'VAR_LIST_END':
                recording = False
                input_vars.append(current_var)
            elif recording and line:
                try:
                    # 去除可能的前导空格并转换为浮点数
                    value = float(line.strip())
                    current_var.append(value)
                except ValueError:
                    pass  # 忽略无法转换为浮点数的行
            elif line == 'BEGIN':
                break  # 遇到BEGIN行，停止读取输入变量

    # 排除最后一组变量（频率变量）
    if len(input_vars) > 0:
        input_vars = input_vars[:-1]

    return input_vars

def generate_input_combinations(input_vars):
    """生成所有输入变量的组合，从右到左遍历"""
    # 使用itertools.product生成所有组合，不需要反转输入变量的顺序
    combinations = list(itertools.product(*input_vars))
    # 不需要反转组合顺序
    return combinations

def read_s_parameters(file_path, freq_num=3):
    """读取S参数数据"""
    s11_data = []
    s21_data = []
    
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # 查找BEGIN行的索引
    begin_indices = [i for i, line in enumerate(content) if line.strip() == 'BEGIN']
    
    if len(begin_indices) < 2:
        raise ValueError("文件中没有足够的BEGIN-END数据块")
    
    # 读取S11数据（第一个BEGIN-END块）
    i = begin_indices[0] + 1
    while i < len(content) and content[i].strip() != 'END':
        line = content[i].strip()
        if line:
            try:
                # 分割逗号分隔的数据
                parts = line.split(',')
                if len(parts) >= 2:
                    db = float(parts[0].strip())
                    phase = float(parts[1].strip())
                    s11_data.append((db, phase))
            except ValueError:
                pass  # 忽略无法解析的行
        i += 1
    
    # 读取S21数据（第二个BEGIN-END块）
    i = begin_indices[1] + 1
    while i < len(content) and content[i].strip() != 'END':
        line = content[i].strip()
        if line:
            try:
                parts = line.split(',')
                if len(parts) >= 2:
                    db = float(parts[0].strip())
                    phase = float(parts[1].strip())
                    s21_data.append((db, phase))
            except ValueError:
                pass  # 忽略无法解析的行
        i += 1
    
    return s11_data, s21_data


def organize_output_data(s11_data, s21_data, input_combinations, freq_num):
    """组织输出数据"""
    output_data = []

    # 确保数据长度匹配
    total_combinations = len(input_combinations)
    if len(s11_data) != total_combinations * freq_num or len(s21_data) != total_combinations * freq_num:
        raise ValueError(
            f"S参数数据长度不匹配: S11={len(s11_data)}, S21={len(s21_data)}, 预期={total_combinations * freq_num}")

    # 为每个输入组合组织输出数据
    for i in range(total_combinations):
        row = []
        # 按频率顺序添加S11的dB值
        for j in range(freq_num):
            idx = i * freq_num + j
            row.append(s11_data[idx][0])

        # 按频率顺序添加S11的相位值
        for j in range(freq_num):
            idx = i * freq_num + j
            row.append(s11_data[idx][1])

        # 按频率顺序添加S21的dB值
        for j in range(freq_num):
            idx = i * freq_num + j
            row.append(s21_data[idx][0])

        # 按频率顺序添加S21的相位值
        for j in range(freq_num):
            idx = i * freq_num + j
            row.append(s21_data[idx][1])

        output_data.append(row)

    return output_data

def main():
    file_path = 'ADS_Resonator2.cti'
    freq_num = 911  # 频率点数
    
    # 读取输入变量
    input_vars = read_input_variables(file_path)
    print(f"读取到 {len(input_vars)} 组输入变量")
    
    # 生成所有输入组合
    input_combinations = generate_input_combinations(input_vars)
    print(f"生成了 {len(input_combinations)} 种输入组合")
    
    # 保存输入数据到文件
    with open('S_Parameter_input_ADS.txt', 'a') as f:
        for combo in input_combinations:
            f.write(' '.join(map(str, combo)) + '\n')

    print("输入数据已保存到 S_Parameter_input_ADS.txt 文件中")

    # 读取S参数数据
    s11_data, s21_data = read_s_parameters(file_path, freq_num)
    print(f"读取到 {len(s11_data)} 个S11数据点和 {len(s21_data)} 个S21数据点")
    
    # 组织输出数据
    output_data = organize_output_data(s11_data, s21_data, input_combinations, freq_num)

    print(f"组织了 {len(output_data)} 行输出数据")
    
    # 保存输出数据到文件
    with open('S_Parameter_output_ADS.txt', 'a') as f:
        for row in output_data:
            f.write(' '.join(map(str, row)) + '\n')

    print("输出数据已保存到 S_Parameter_output_ADS.txt 文件中")
    
    print("数据处理完成，已保存")

if __name__ == "__main__":
    main()