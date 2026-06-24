import cupy as cp
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.draw import line
from scipy.optimize import curve_fit
from scipy.stats import linregress


def compute_ca_wave_speed(
    Ca_map_history, mask_filename, pixel_size_um, dt
):
    print("\n===== Computing Ca²⁺ Wave Speed (Background Excluded) =====\n")

    # Load and apply the astrocyte mask (NumPy since it's small)
    mask = np.load(mask_filename)
    valid_mask = (mask == 1) | (mask == 2) | (mask == 3)  # Exclude background (0)

    # Apply mask to exclude background regions
    ca_filtered = Ca_map_history[:, valid_mask]  # Only soma, thick branches, thin processes

    # Compute mean Ca²⁺ and IP₃ over time (GPU-Accelerated)
    mean_ca_over_time = cp.mean(ca_filtered, axis=1)

    # Convert CuPy arrays to NumPy for fitting
    mean_ca_np = cp.asnumpy(mean_ca_over_time)
    time_range_np = np.arange(len(mean_ca_np)) * dt * 1000  # Convert to milliseconds

    # --- 1. Calculate Wave Speed ---
    threshold = 0.2 * cp.max(Ca_map_history)  # Set threshold for wavefront detection
    wavefront_map = cp.argmax(Ca_map_history > threshold, axis=0)  # Find wavefront time per pixel
    wavefront_positions = cp.mean(wavefront_map, axis=1)  # Mean wavefront position over time

    # Convert to NumPy
    wavefront_positions_np = cp.asnumpy(wavefront_positions)
    time_intervals_np = np.arange(len(wavefront_positions_np))

    # Convert to NumPy
    wavefront_positions_np = cp.asnumpy(wavefront_positions)
    time_intervals_np = np.arange(len(wavefront_positions_np))

    # Fit a linear model to estimate speed
    wave_speed_pixels_per_timestep, _ = np.polyfit(time_intervals_np, wavefront_positions_np, 1)
    wave_speed_um_per_s = (wave_speed_pixels_per_timestep * pixel_size_um) / dt
    
    # Print results for wave speed
    expected_wave_speed_range = "Expected: 7-27 µm/s"
    if 7 <= wave_speed_um_per_s <= 27:
        print(f"Wave Speed: {wave_speed_um_per_s:.2f} µm/s ({expected_wave_speed_range} )")
    else:
        print(f"Wave Speed: {wave_speed_um_per_s:.2f} µm/s ({expected_wave_speed_range} )")

    return wave_speed_um_per_s

def measure_wave_propagation(Ca_map_history, segmented_data, trigger_pixel, astro_mask, pixel_size_um=0.2, threshold=0.2):

    # Convert CuPy arrays to NumPy
    Ca_map_history_np = cp.asnumpy(Ca_map_history)  # Convert calcium history to NumPy
    astro_mask_np = cp.asnumpy(astro_mask)  # Convert astro_mask to NumPy

    # Get the final time step map to measure wave spread
    final_wave_map = Ca_map_history_np[-1]

    # Find all pixels where Ca²⁺ exceeded the threshold
    active_pixels = np.argwhere((final_wave_map > threshold) & astro_mask_np)

    # Compute distances from the trigger pixel
    distances = np.sqrt((active_pixels[:, 0] - trigger_pixel[0])**2 + (active_pixels[:, 1] - trigger_pixel[1])**2)

    # Get maximum distance in pixels and convert to micrometers
    max_distance_pixels = np.max(distances) if distances.size > 0 else 0
    max_distance_um = max_distance_pixels * pixel_size_um
    print(f"Calcium wave propagates up to: {max_distance_um:.2f} µm")
    
    return max_distance_um

def exp_decay(t, A, tau):
    """Single-exponential decay function."""
    return A * np.exp(-t / tau)

def compute_ip3_decay_constant(ip3_map_history, astro_mask, dt):

    # Ensure `ip3_map_history` has no NaN or Inf values
    ip3_map_history = cp.nan_to_num(ip3_map_history, nan=0.0, posinf=5.0, neginf=0.0)

    # Extract mean IP₃ over time
    avg_ip3_over_time = cp.mean(ip3_map_history[:, astro_mask], axis=1)
    avg_ip3_np = cp.asnumpy(avg_ip3_over_time)

    # Find peak and extract decay phase
    peak_index = np.argmax(avg_ip3_np)  
    decay_ip3 = avg_ip3_np[peak_index:]
    time_range = np.arange(len(decay_ip3)) * dt  

    # Ensure decay phase is not empty
    if len(decay_ip3) == 0:
        print("Warning: No valid decay phase detected!")
        return np.nan  # Prevents curve fitting failure

    # Remove `NaN` and `Inf` values before curve fitting
    valid_indices = np.isfinite(decay_ip3)
    time_range = time_range[valid_indices]
    decay_ip3 = decay_ip3[valid_indices]

    # Ensure enough valid points for fitting
    if len(decay_ip3) < 5:
        print("Warning: Not enough valid points for fitting!")
        return np.nan

    # Fit single-exponential decay
    try:
        popt, _ = curve_fit(exp_decay, time_range, decay_ip3, p0=[decay_ip3[0], 1.0])
        tau_ip3 = popt[1]  
    except RuntimeError:
        print("Warning: Fit failed.")
        return np.nan

    # Print results for IP₃ decay time constant
    expected_ip3_range = "Expected: 2 - 7 s"
    if 2 <= tau_ip3 <= 7:
        print(f"Computed IP₃ Decay Time Constant in range (τ): {tau_ip3:.3f} s ({expected_ip3_range} )")
    else:
        print(f"Error: Computed IP₃ Decay Time Constant out of range (τ): {tau_ip3:.3f} s ({expected_ip3_range} )")
        
    return tau_ip3

def compute_ca_time_decay_constant(Ca_map_history, mask_filename, dt):
    print("\n===== Computing Ca²⁺ Decay Analysis (Background Excluded) =====\n")

    # Load and apply the astrocyte mask
    mask = np.load(mask_filename)
    valid_mask = (mask == 0) | (mask == 1) | (mask == 2)  # Exclude background (0)

    # Apply mask to exclude background regions
    ca_filtered = Ca_map_history[:, valid_mask]  # Only soma, thick branches, thin processes

    # Compute mean Ca²⁺ over time
    mean_ca_over_time = cp.mean(ca_filtered, axis=1)
    mean_ca_np = cp.asnumpy(mean_ca_over_time)  # Convert for fitting

    # Generate time array (Convert to milliseconds)
    time_range_np = np.arange(len(mean_ca_np)) * dt * 1000  

    # Identify peak time index
    peak_idx = np.argmax(mean_ca_np)  # Find peak Ca²⁺ time index
    post_peak_time = time_range_np[peak_idx:]  # Use only decay phase
    post_peak_ca = mean_ca_np[peak_idx:]

    # Apply cutoff dynamically to exclude plateau
    cutoff_value = 0.1 * post_peak_ca[0]  # 10% of peak value
    cutoff_index = np.where(post_peak_ca < cutoff_value)[0]  # Find when decay falls below 10%

    if len(cutoff_index) > 0:
        post_peak_time = post_peak_time[:cutoff_index[0]]
        post_peak_ca = post_peak_ca[:cutoff_index[0]]

    # Ensure we have enough points for a meaningful fit
    if len(post_peak_time) < 5:  
        print("Not enough data points for fitting. Returning NaN.")
        return np.nan

    # Define exponential decay function
    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)

    # Fit the decay curve for Ca²⁺
    try:
        popt_ca, _ = curve_fit(exp_decay, post_peak_time, post_peak_ca, p0=[post_peak_ca[0], 2000])
        tau_ca = popt_ca[1]  # Ca²⁺ time constant (ms)
    except RuntimeError:
        tau_ca = np.nan  # If fitting fails, return NaN

    # Print results
    expected_ca_range = "Expected: 1500 - 2500 ms"
    if 1500 <= tau_ca <= 2500:
        print(f"Ca²⁺ Decay Time Constant in range (τ): {tau_ca:.2f} ms ({expected_ca_range} )")
    else:
        print(f"Error: Ca²⁺ Decay Time Constant out of range (τ): {tau_ca:.2f} ms ({expected_ca_range} )")

    return tau_ca

# Function to safely convert NumPy objects to JSON-compatible formats
def safe_list_conversion(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    return obj  # Keep other objects unchanged

# Function to convert NumPy objects to JSON-compatible formats
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    if isinstance(obj, np.integer):  # Convert NumPy int (e.g., int64) to Python int
        return int(obj)
    if isinstance(obj, np.floating):  # Convert NumPy float (e.g., float64) to Python float
        return float(obj)
    return obj  # Return object as-is if it's already serializable

def compute_wave_speed_through_soma(
    Ca_map_history, trigger_pixel, soma_centroid, astro_mask,
    segmented_data, pixel_size_um, dt, plot_debug=True
):

    # Convert everything to NumPy
    Ca_np = cp.asnumpy(Ca_map_history)
    mask_np = cp.asnumpy(astro_mask)
    seg_np = cp.asnumpy(segmented_data) if isinstance(segmented_data, cp.ndarray) else segmented_data

    r1, c1 = trigger_pixel
    r2, c2 = soma_centroid

    # Generate line from trigger to soma
    path_r, path_c = line(r1, c1, r2, c2)

    # Extend line beyond soma by 30% of the original length
    extend_len = int(0.3 * np.sqrt((r2 - r1)**2 + (c2 - c1)**2))
    dr, dc = r2 - r1, c2 - c1
    unit_r, unit_c = dr / np.hypot(dr, dc), dc / np.hypot(dr, dc)

    for i in range(1, extend_len + 1):
        r_ext = int(round(r2 + i * unit_r))
        c_ext = int(round(c2 + i * unit_c))
        if 0 <= r_ext < Ca_np.shape[1] and 0 <= c_ext < Ca_np.shape[2]:
            path_r = np.append(path_r, r_ext)
            path_c = np.append(path_c, c_ext)

    # Filter valid astrocyte pixels only
    valid_path = [(r, c) for r, c in zip(path_r, path_c) if mask_np[r, c]]
    if len(valid_path) < 2:
        raise ValueError("Too few valid astrocyte pixels in the selected path.")
    path_r, path_c = zip(*valid_path)
    path_coords = list(zip(path_r, path_c))

    # Extract calcium values and compute threshold crossings
    Ca_traces = Ca_np[:, path_r, path_c]  # shape: (T, path_len)
  
    # NEW: Time-to-peak method
    arrival_times = np.argmax(Ca_traces, axis=0) * dt  # axis=0 = time
    arrival_times = np.array(arrival_times)

    # Compute distances along path
    distances = np.sqrt((np.array(path_r) - r1)**2 + (np.array(path_c) - c1)**2) * pixel_size_um

    # Linear regression: time vs distance
    valid = ~np.isnan(arrival_times)
    distances = distances[valid]
    arrival_times = arrival_times[valid]

    if len(arrival_times) < 2:
        raise RuntimeError("Not enough points with valid arrival times for speed calculation.")

    slope, intercept, r_val, _, _ = linregress(arrival_times, distances)
    speed_um_per_s = slope

    print(f"Wave speed: {speed_um_per_s:.2f} µm/s | R² = {r_val**2:.3f}")

    # Plot the path and Ca²⁺ overlay (optional)
    if plot_debug:
        mid_time = np.argmax(np.mean(Ca_traces, axis=1))
        plt.figure(figsize=(7, 7))
        plt.imshow(seg_np, cmap='gray', alpha=0.4)
        plt.imshow(Ca_np[mid_time], cmap='Blues', alpha=0.6)
        plt.scatter([c1], [r1], color='red', label='Trigger', s=60)
        plt.scatter([c2], [r2], color='orange', label='Soma', s=60)
        plt.plot(path_c, path_r, color='lime', linewidth=2, label='Wave Path')
        plt.title("Ca²⁺ Wave Path Through Soma")
        plt.legend()
        plt.axis('off')
        plt.show()
        plt.figure()
        plt.plot(arrival_times, distances, 'o-')
        plt.xlabel("Arrival Time (s)")
        plt.ylabel("Distance from Trigger (µm)")
        plt.title("Distance vs. Ca²⁺ Arrival Time")
        plt.grid(True)
        plt.show()

    return speed_um_per_s

def compute_time_to_peak_map(Ca_map_history, dt, mask=None):
    peak_indices = cp.argmax(Ca_map_history, axis=0)  # [H, W] = index of peak
    time_to_peak = peak_indices * dt
    if mask is not None:
        time_to_peak = cp.where(mask, time_to_peak, cp.nan)
    return cp.asnumpy(time_to_peak)

def compute_wave_speed_from_morphology_path_fallback(
    t_peak_map, trigger_pixel, mask, pixel_size_um, min_valid_points=5, plot_debug=True
):
    if isinstance(mask, cp.ndarray):
        mask = cp.asnumpy(mask)

    # Get valid astrocyte pixels
    indices = np.argwhere(mask & ~np.isnan(t_peak_map))
    arrival_times = []
    distances = []

    for r, c in indices:
        time = t_peak_map[r, c]
        if np.isnan(time):
            continue
        dy, dx = r - trigger_pixel[0], c - trigger_pixel[1]
        dist_um = np.sqrt(dx**2 + dy**2) * pixel_size_um
        arrival_times.append(time)
        distances.append(dist_um)

    arrival_times = np.array(arrival_times)
    distances = np.array(distances)

    if len(arrival_times) < min_valid_points:
        print("Not enough valid points to fit wave speed.")
        return np.nan

    # Linear fit
    slope, intercept, r_val, _, _ = linregress(arrival_times, distances)
    wave_speed = slope  # µm/s

    if plot_debug:
        plt.figure(figsize=(6, 5))
        plt.scatter(arrival_times, distances, c=distances, cmap='viridis', s=30)
        plt.plot(arrival_times, slope * arrival_times + intercept, 'k--', label=f"Speed: {wave_speed:.2f} µm/s")
        plt.xlabel("Arrival Time (s)")
        plt.ylabel("Distance from Trigger (µm)")
        plt.title("Wave Speed via Morphology Path Fallback")
        plt.legend()
        plt.grid(True)
        plt.colorbar(label="Distance")
        plt.tight_layout()
        plt.show()

    return wave_speed
