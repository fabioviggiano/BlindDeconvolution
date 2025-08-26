# analyze_results.py
# NOTE: Comments have been translated to English to remove all special characters
# and definitively solve file encoding issues.

import os
import json
import pandas as pd

# --- Configuration ---
# The folder where the script will look for metadata files.
RESULTS_DIR = 'results'
# The name of the CSV file that will be generated.
REPORT_FILENAME = 'report.csv'

def analyze():
    """
    Loads all JSON metadata from the results folder, aggregates them
    into a Pandas DataFrame, prints a report, and saves a CSV file.
    """
    all_results = []
    print(f"Scanning folder '{RESULTS_DIR}'...")

    # 1. Scan all files in the results folder
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if filename.endswith('_metadata.json'):
            filepath = os.path.join(RESULTS_DIR, filename)
            
            # 2. Read each JSON file
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # 3. "Flatten" the JSON structure into a single-level dictionary
                    flat_data = {
                        'image': data.get('parameters', {}).get('image'),
                        'method': data.get('method'),
                        'kernel_size': data.get('parameters', {}).get('kernel_size'),
                        'lambda_prior': data.get('parameters', {}).get('lambda_prior'),
                        'lambda_kernel': data.get('parameters', {}).get('lambda_kernel'),
                        'iterations': data.get('parameters', {}).get('iterations'),
                        'blur_type': data.get('parameters', {}).get('blur_type', 'N/A'),
                        'motion_len': data.get('parameters', {}).get('motion_len', 'N/A'),
                        'gaussian_sigma': data.get('parameters', {}).get('gaussian_sigma', 'N/A'),
                        'exec_time_sec': data.get('execution_time_sec'),
                        'input_sharpness': data.get('input_sharpness'),
                        'output_sharpness': data.get('output_sharpness'),
                        'psnr': data.get('metrics', {}).get('psnr'),
                        'ssim': data.get('metrics', {}).get('ssim'),
                        'source_file': filename 
                    }
                    all_results.append(flat_data)
            except json.JSONDecodeError:
                print(f"Warning: Could not read JSON file, it might be corrupted: {filename}")
            except Exception as e:
                print(f"Unexpected error while processing file {filename}: {e}")

    if not all_results:
        print(f"No '_metadata.json' files found in the '{RESULTS_DIR}' folder.")
        print("Make sure you have run at least one test with 'main.py'.")
        return

    # 4. Create a Pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # 5. Set display options for the terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)
    pd.set_option('display.precision', 4)

    print("\n--- Summary of All Executed Tests ---")
    
    columns_to_show = [
        'image', 'kernel_size', 'lambda_prior', 'lambda_kernel', 
        'blur_type', 'motion_len', 'psnr', 'ssim', 'output_sharpness'
    ]
    display_columns = [col for col in columns_to_show if col in df.columns]
    print(df[display_columns])
    
    # 6. Save the full report to a CSV file
    try:
        df.sort_values(by=['image', 'psnr'], ascending=[True, False]).to_csv(REPORT_FILENAME, index=False)
        print(f"\nFull report successfully saved to '{REPORT_FILENAME}'")
        print("You can open this file with Excel to filter, sort, and analyze the data.")
    except Exception as e:
        print(f"\nError while saving the CSV report: {e}")


if __name__ == "__main__":
    print("Executing results analysis script...")
    try:
        import pandas
    except ImportError:
        print("\nWARNING: The 'pandas' library is not installed.")
        print("Please run this from your terminal: pip install pandas")
        exit()
        
    analyze()