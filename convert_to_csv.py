import csv
import re
import os

def parse_metrics(file_path, output_csv):
    """
    Parses the metrics.txt file and converts it to a CSV file.
    """
    # Regex to match the data lines
    # Example line: [0522c0002+15Ag+MOS_21176] MAE: 1.9494 | MSE: 16.0461 | RMSE: 4.0058 | PSNR: 25.5562 | SSIM: 0.9419
    pattern = re.compile(r'\[(?P<id>.*?)\] MAE: (?P<mae>[\d\.]+) \| MSE: (?P<mse>[\d\.]+) \| RMSE: (?P<rmse>[\d\.]+) \| PSNR: (?P<psnr>[\d\.]+) \| SSIM: (?P<ssim>[\d\.]+)')

    results = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        match = pattern.search(line)
        if match:
            results.append(match.groupdict())

    if not results:
        print("No metrics found in the file.")
        return

    # Write to CSV
    keys = ['id', 'mae', 'mse', 'rmse', 'psnr', 'ssim']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"Successfully converted {len(results)} rows to {output_csv}")

if __name__ == "__main__":
    # Define paths
    input_file = r"C:\Users\960\Desktop\baseline\results\baseline_nah&lung\metrics.txt"
    output_file = r"C:\Users\960\Desktop\baseline\results\baseline_nah&lung\metrics.csv"
    
    parse_metrics(input_file, output_file)
