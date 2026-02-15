import sys
import os

def clean_and_load_data(filename):
    """
    Reads the file, removes unwanted characters, and returns a list of integers.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    with open(filename, 'r') as f:
        content = f.read()

    # Filter: keep only '0', '1', and ','
    cleaned_content = "".join([c for c in content if c in "01,"])
    
    # Split by comma
    raw_elements = cleaned_content.split(',')
    
    # Convert to int, ignoring empty strings (handles trailing comma case)
    data = [int(x) for x in raw_elements if x.strip() != ""]
    
    return data

def generate_fen(data):
    if len(data) != 740:
        print(f"Error: Expected 740 elements, but found {len(data)}.")
        sys.exit(1)

    # Initialize an empty board (64 squares)
    # Index 0 = a8, Index 63 = h1 (FEN order)
    board = [None] * 64

    # --- 1. Map Standard Pieces (64 squares each) ---
    # Order: r, n, b, q, k (Black), R, N, B, Q, K (White)
    pieces = ['r', 'n', 'b', 'q', 'k', 'R', 'N', 'B', 'Q', 'K']
    current_index = 0
    
    for piece_char in pieces:
        for i in range(64):
            if data[current_index + i] == 1:
                board[i] = piece_char
        current_index += 64 # Move to next piece type

    # --- 2. Map Pawns (48 squares each) ---
    # Pawns exist on Rank 7 through Rank 2 (Indices 8 to 55 in our 0-63 board)
    # We skip Rank 8 (indices 0-7) and Rank 1 (indices 56-63)
    
    # Black Pawns
    pawn_start_index = 640
    for i in range(48):
        if data[pawn_start_index + i] == 1:
            # Map input index 0->7 (Rank 7) to board index 8->15
            board[i + 8] = 'p'
            
    # White Pawns
    pawn_start_index = 688
    for i in range(48):
        if data[pawn_start_index + i] == 1:
            board[i + 8] = 'P'

    # --- 3. Construct Piece Placement String ---
    fen_rows = []
    for rank in range(8):
        empty_count = 0
        row_string = ""
        for file in range(8):
            square_index = rank * 8 + file
            piece = board[square_index]
            
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_string += str(empty_count)
                    empty_count = 0
                row_string += piece
        
        # Flush remaining empty squares at end of row
        if empty_count > 0:
            row_string += str(empty_count)
        
        fen_rows.append(row_string)

    piece_placement = "/".join(fen_rows)

    # --- 4. Castling Rights (Last 4 elements) ---
    # Indices: 736 (K), 737 (Q), 738 (k), 739 (q)
    castling = ""
    if data[736] == 1: castling += "K"
    if data[737] == 1: castling += "Q"
    if data[738] == 1: castling += "k"
    if data[739] == 1: castling += "q"
    
    if castling == "":
        castling = "-"

    # --- 5. Final FEN Assembly ---
    # Fixed values: Active color (w), En Passant (-), Halfmove (0), Fullmove (1)
    final_fen = f"{piece_placement} w {castling} - 0 1"
    
    return final_fen

if __name__ == "__main__":
    # Create a dummy file for testing if it doesn't exist
    filename = "FenTest.txt"
    
    # Run the conversion
    data = clean_and_load_data(filename)
    fen_string = generate_fen(data)
    
    print("-" * 30)
    print("Generated FEN:")
    print(fen_string)
    print("-" * 30)