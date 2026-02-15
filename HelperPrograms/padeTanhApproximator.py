import math
import statistics

def pade_tanh(x):
    """
    Calculates the Pade approximant for tanh(x) using the provided formula:
    (x + 1/9 x^3 + 1/945 x^5) / (1 + 4/9 x^2 + 1/63 x^4)
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x3 * x2
    
    numerator = x + (1/9 * x3) + (1/945 * x5)
    denominator = 1 + (4/9 * x2) + (1/63 * x4)
    
    return numerator / denominator

def analyze_range(start, end, points):
    """
    Calculates errors for a specific range and returns the statistics.
    """
    step = (end - start) / (points - 1)
    tanh_errors = []
    deriv_errors = []
    
    for i in range(points):
        x = start + (i * step)
        
        # --- Tanh Value Comparison ---
        actual_tanh = math.tanh(x)
        approx_tanh = pade_tanh(x)
        tanh_errors.append(abs(actual_tanh - approx_tanh))
        
        # --- Derivative Comparison ---
        actual_deriv = 1.0 - (actual_tanh ** 2)
        approx_deriv = 1.0 - (approx_tanh ** 2)
        deriv_errors.append(abs(actual_deriv - approx_deriv))

    stats = {
        "tanh_avg": sum(tanh_errors) / points,
        "tanh_med": statistics.median(tanh_errors),
        "tanh_max": max(tanh_errors),
        "deriv_avg": sum(deriv_errors) / points,
        "deriv_med": statistics.median(deriv_errors),
        "deriv_max": max(deriv_errors)
    }
    return stats

def run_comparison():
    # Defining ranges as (start, end, total_points_proportional_to_width)
    ranges = [
        (-2.0, -1.0, 10000),
        (-1.0, -0.5, 10000),
        (-0.5, 0.0, 10000),
        (0.0, 0.5, 10000),
        (0.5, 1.0, 10000),
        (1.0, 2.0, 10000)
    ]
    
    print(f"{'Range':^15} | {'Type':^5} | {'Avg Error':^15} | {'Max Error':^15}")
    print("-" * 60)
    
    for start, end, pts in ranges:
        res = analyze_range(start, end, pts)
        range_str = f"[{start}, {end}]"
        
        # Print Tanh Stats
        print(f"{range_str:<15} | Tanh  | {res['tanh_avg']:.8e} | {res['tanh_max']:.8e}")
        # Print Derivative Stats
        print(f"{'':<15} | Deriv | {res['deriv_avg']:.8e} | {res['deriv_max']:.8e}")
        
        # Detailed block for the specific 3 values requested
        print(f"  -> Tanh  - Med: {res['tanh_med']:.8e}")
        print(f"  -> Deriv - Med: {res['deriv_med']:.8e}")
        print("-" * 60)

if __name__ == "__main__":
    run_comparison()