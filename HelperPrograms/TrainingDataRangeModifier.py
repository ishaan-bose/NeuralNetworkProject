import csv
import os
import sys
import tempfile
import shutil

def process_csv_in_place(filename):
    """
    Reads a 2-column CSV, divides the second column by 100.0,
    and formats it to exactly 4 decimal places, preserving the sign.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    # Create a temporary file to write processed data safely
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filename), text=True)
    os.close(fd)

    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            with open(temp_path, mode='w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                
                row_count = 0
                for row in reader:
                    if not row:
                        continue
                    
                    # Ensure the row has at least 2 columns
                    if len(row) >= 2:
                        label = row[0]
                        try:
                            # Convert second column to float
                            value = float(row[1])
                            
                            # Divide by 100.0
                            new_value = value / 100.0
                            
                            # Format to exactly 4 decimal places
                            # f-string :.4f preserves the sign automatically (+ or -)
                            row[1] = f"{new_value:.4f}"
                        except ValueError:
                            # If the second column isn't a number (e.g. header), leave it
                            pass
                        
                        writer.writerow(row[:2]) # Ensure we only write 2 columns
                        row_count += 1
                
        # Replace the original file with the temporary one
        shutil.move(temp_path, filename)
        print(f"Successfully processed {row_count} rows in '{filename}'.")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    user_file = input("Enter the CSV filename to process: ").strip()
    if user_file:
        process_csv_in_place(user_file)
    else:
        print("Filename cannot be empty.")