import pandas as pd
import re

# 1. 设置文件路径
# input_file = r'C:\Users\960\Desktop\Top2\results\mednext\visualization_slices\1.txt'  # 这里换成你实际的 txt 文件名
input_file = r'C:\Users\960\Desktop\DinoUNETR\1.txt'
output_file = 'metrics_analysis.csv'

# 2. 定义正则表达式
# 匹配格式: [Case: ...] Depth: ... | MAE: ... | MSE: ... | RMSE: ... | PSNR: ... dB

pattern = re.compile(
    r"\[Case:\s*(?P<Case>.*?)\]\s*"
    r"Depth:\s*(?P<Depth>\d+)\s*\|\s*"
    r"MAE:\s*(?P<MAE>[\d.]+)\s*\|\s*"
    r"MSE:\s*(?P<MSE>[\d.]+)\s*\|\s*"
    r"RMSE:\s*(?P<RMSE>[\d.]+)\s*\|\s*"
    r"PSNR:\s*(?P<PSNR>[\d.]+)\s*dB"
)
data = []

# 3. 读取并解析文件
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            match = pattern.search(line)
            if match:
                data.append(match.groupdict())
            else:
                print(f"警告：无法解析行 -> {line}")
except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}，请确认文件名是否正确。")
    exit()

# 4. 转换为 DataFrame
if not data:
    print("未提取到任何数据，请检查文件内容格式。")
else:
    df = pd.DataFrame(data)

    # 将数值列转换为浮点数，方便排序
    numeric_cols = ['MAE', 'MSE', 'RMSE', 'PSNR']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])


    # 6. 保存为 CSV
    df.to_csv(output_file, index=False)
    
    print(f"提取完成！结果已保存至 {output_file}")