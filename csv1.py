import re
import pandas as pd

def parse_global_metrics(content):
    """Parses global_metrics.txt content."""
    data = {}
    # Split by [Case: to handle multi-line blocks easily
    chunks = content.split('[Case: ')
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        try:
            # Extract Case Name (it's at the start of the chunk until ']')
            case_name = chunk.split(']')[0].strip()
            
            # Extract metrics using regex
            mae_match = re.search(r'MAE:\s+([\d\.]+)', chunk)
            mse_match = re.search(r'MSE:\s+([\d\.]+)', chunk)
            rmse_match = re.search(r'RMSE:\s+([\d\.]+)', chunk)
            psnr_match = re.search(r'PSNR:\s+([\d\.]+)', chunk)
            
            if case_name and mae_match:
                data[case_name] = {
                    'Global_MAE': float(mae_match.group(1)),
                    'Global_MSE': float(mse_match.group(1)) if mse_match else None,
                    'Global_RMSE': float(rmse_match.group(1)) if rmse_match else None,
                    'Global_PSNR': float(psnr_match.group(1)) if psnr_match else None,
                }
        except Exception:
            continue
    return data

def parse_metrics_txt(content):
    """Parses global_metrics.txt content."""
    data = {}
    # Split by [Case: to handle multi-line blocks easily
    chunks = content.split('[Case: ')
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        try:
            # Extract Case Name (it's at the start of the chunk until ']')
            case_name = chunk.split(']')[0].strip()
            
            # Extract metrics using regex
            mae_match = re.search(r'MAE:\s+([\d\.]+)', chunk)
            mse_match = re.search(r'MSE:\s+([\d\.]+)', chunk)
            rmse_match = re.search(r'RMSE:\s+([\d\.]+)', chunk)
            psnr_match = re.search(r'PSNR:\s+([\d\.]+)', chunk)
            
            if case_name and mae_match:
                data[case_name] = {
                    'Metrics_MAE': float(mae_match.group(1)) if mae_match else None,
                    'Metrics_MSE': float(mse_match.group(1)) if mse_match else None,
                    'Metrics_RMSE': float(rmse_match.group(1)) if rmse_match else None,
                    'Metrics_PSNR': float(psnr_match.group(1)) if psnr_match else None,
                }
        except Exception:
            continue
    return data

# Main Execution
try:
    with open(r'C:\Users\960\Desktop\DinoUNETR\results\result_default_full_256_meddino_changechannel\global_metrics.txt', 'r') as f:
        global_content = f.read()
    with open(r'C:\Users\960\Desktop\Top2\results\mednext\global_metrics.txt', 'r') as f:
        global_content2 = f.read()

    # Parse data
    global_data = parse_global_metrics(global_content)
    global_data2 = parse_metrics_txt(global_content2)

    # Merge data
    all_cases = set(global_data.keys()) | set(global_data2.keys())
    combined_list = []

    for case in all_cases:
        row = {'Case': case}
        if case in global_data:
            row.update(global_data[case])
        if case in global_data2:
            row.update(global_data2[case])
        combined_list.append(row)

    # Create DataFrame and Reorder
    df = pd.DataFrame(combined_list)
    cols = ['Case', 'Metrics_MAE', 'Metrics_RMSE', 'Metrics_MSE', 
              'Metrics_PSNR', 'Global_MAE', 'Global_MSE', 'Global_RMSE', 'Global_PSNR']
            
    # Add missing columns if any
    for col in cols:
        if col not in df.columns:
            df[col] = None

    df = df[cols].sort_values(by='Case')
    df.to_csv('metrics_comparison.csv', index=False)
    print("CSV generated successfully.")
    
except FileNotFoundError:
    print("Please ensure 'global_metrics.txt' and 'metrics.txt' are in the same directory.")