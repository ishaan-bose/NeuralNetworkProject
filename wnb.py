import csv
import random
import sys
import os

def generate_biased_csv(filename, rows, cols):
    """
    Generates a CSV file with values biased toward zero using a 
    Normal distribution, clipped to the range [-1, 1].
    """
    # Ensure the filename ends with .csv
    if not filename.lower().endswith('.csv'):
        filename += '.csv'

    print(f"-> Generative Task: {filename} ({rows}x{cols})")

    try:
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for i in range(rows):
                row = []
                for _ in range(cols):
                    # random.gauss(mu, sigma)
                    # mu=0 (centered at zero)
                    # sigma=0.33 (puts ~99% of values within [-1, 1] before clipping)
                    val = random.gauss(0, 0.005)
                    
                    # Clip to ensure we stay strictly within [-1, 1]
                    val = max(-1.0, min(1.0, val))
                    
                    # Format to 6 decimal places
                    row.append(f"{val:.6f}")
                
                writer.writerow(row)
                
        print(f"   Successfully generated '{filename}'.")

    except PermissionError:
        print(f"   Error: Permission denied when writing to '{filename}'.")
    except Exception as e:
        print(f"   An unexpected error occurred: {e}")

def process_specifications():
    spec_file = "weightsAndBiasesSpecifications.txt"
    
    if not os.path.exists(spec_file):
        print(f"Error: Could not find '{spec_file}' in the current directory.")
        return

    print(f"Reading specifications from: {spec_file}")
    print("-" * 50)

    try:
        with open(spec_file, 'r') as f:
            # Read all lines and strip whitespace, ignoring empty lines
            lines = [line.strip() for line in f if line.strip()]

        total_lines = len(lines)
        
        # Check if we have complete triplets
        if total_lines % 3 != 0:
            print(f"Warning: The file has {total_lines} lines, which is not a multiple of 3.")
            print("Some data at the end might be incomplete and will be skipped.\n")

        # Iterate through the lines in steps of 3
        for i in range(0, total_lines, 3):
            # Ensure we have a full triplet (Name, Rows, Cols)
            if i + 2 >= total_lines:
                break
                
            filename = lines[i]
            rows_str = lines[i+1]
            cols_str = lines[i+2]
            
            try:
                rows = int(rows_str)
                cols = int(cols_str)
                
                if rows <= 0 or cols <= 0:
                    print(f"   Skipping '{filename}': Rows and columns must be positive integers.")
                    continue
                    
                generate_biased_csv(filename, rows, cols)
                
            except ValueError:
                print(f"   Error parsing dimensions for '{filename}'. Expected integers, got ({rows_str}, {cols_str}).")

        print("-" * 50)
        print("Batch generation complete.")

    except Exception as e:
        print(f"Critical error reading specification file: {e}")

if __name__ == "__main__":
    process_specifications()