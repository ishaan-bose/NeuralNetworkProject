import os
import csv
import sys

def get_csv_dimensions(filepath):
    """
    Calculates the number of rows and columns in a CSV file 
    without loading the entire file into memory.
    """
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # 1. Get column count from the first row
            try:
                first_row = next(reader)
                cols = len(first_row)
            except StopIteration:
                # File is empty
                return 0, 0
            
            # 2. Count remaining rows
            # We start at 1 because we already consumed the first row
            rows = 1 + sum(1 for _ in reader)
            
            return rows, cols
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None, None

def scan_directory():
    """
    Scans the current directory (no subdirectories) for CSV files
    and prints their dimensions.
    """
    # Get only the files in the current working directory
    current_dir = os.getcwd()
    print(f"Scanning CSV files in: {current_dir}\n")
    print(f"{'Filename':<40} | {'Rows':>10} | {'Cols':>10}")
    print("-" * 66)

    csv_files_found = 0
    
    # os.listdir only lists items in the top level (non-recursive)
    for entry in os.listdir(current_dir):
        # Join path and check if it's a file (to ignore folders)
        full_path = os.path.join(current_dir, entry)
        
        if os.path.isfile(full_path) and entry.lower().endswith('.csv'):
            csv_files_found += 1
            rows, cols = get_csv_dimensions(full_path)
            
            if rows is not None:
                # Truncate filename if it's too long for the display table
                display_name = (entry[:37] + '..') if len(entry) > 39 else entry
                print(f"{display_name:<40} | {rows:>10,} | {cols:>10,}")

    if csv_files_found == 0:
        print("No CSV files found in the current directory.")
    else:
        print("-" * 66)
        print(f"Total CSV files scanned: {csv_files_found}")

if __name__ == "__main__":
    scan_directory()