import random
import os
import cupy as cp
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def load_astrocyte_data(segmented_data=None):
    # If segmented_data is None, prompt the user to select a file
    if segmented_data is None:
        files = [f for f in os.listdir() if f.endswith(".npy")]
        if not files:
            raise FileNotFoundError("No .npy files found in the directory!")
        
        print("Available Astrocyte Files:")
        for i, file in enumerate(files):
            print(f"{i + 1}: {file}")
        
        choice = int(input("Select a file by entering the corresponding number: ")) - 1
        segmented_data = np.load(files[choice])  # Keep NumPy for file loading

    # Ensure that segmented_data is a NumPy array before conversion
    if not isinstance(segmented_data, np.ndarray):
        raise TypeError("segmented_data must be a NumPy array.")

    # Convert segmented_data to CuPy for GPU acceleration
    data = cp.array(segmented_data)

    # Get unique labels and their counts using CuPy
    unique_values, counts = cp.unique(data, return_counts=True)
    unique_values_np = unique_values.get()  # Convert back to NumPy to sort
    counts_np = counts.get()

    # Sort labels by pixel count in ascending order
    #sorted_labels = [label for _, label in sorted(zip(counts_np, unique_values_np))]
    # Assign labels dynamically based on pixel count
    soma_label = 0
    thick_branch_label = 1
    thin_process_label = 2
    background_label = 3
        
    # Create masks for each region (fully vectorized using CuPy)
    soma_mask = data == soma_label
    thick_branch_mask = data == thick_branch_label
    thin_process_mask = data == thin_process_label
    background_mask = data == background_label

    # Compute pixel counts using CuPy
    soma_pixels = cp.sum(soma_mask).item()
    thick_pixels = cp.sum(thick_branch_mask).item()
    thin_pixels = cp.sum(thin_process_mask).item()
    background_pixels = cp.sum(background_mask).item()

    # Print pixel counts and assigned labels
    print(f"\n Initial Pixel Counts:")
    print(f"  Soma: {soma_pixels} pixels (Label: {soma_label})")
    print(f"  Thick Branches: {thick_pixels} pixels (Label: {thick_branch_label})")
    print(f"  Thin Processes: {thin_pixels} pixels (Label: {thin_process_label})")
    print(f"  Background: {background_pixels} pixels (Label: {background_label})")

    return data, soma_mask, thick_branch_mask, thin_process_mask, background_mask, soma_label, thick_branch_label, thin_process_label, background_label

def find_soma_centroid_old(data, soma_mask):

    # Convert soma_mask from CuPy (cp.array) to NumPy (np.array)
    soma_mask_np = soma_mask.get().astype(np.uint8)  # 

    props = regionprops(soma_mask_np)
    if len(props) == 0:
        raise ValueError("No soma region found in the input data!")

    return tuple(map(int, props[0].centroid))  # 

def compute_distance_map(data, centroid_row, centroid_col, background_label, pixel_size_um=0.5):
    rows, cols = data.shape

    # Generate indices on the GPU (CuPy)
    rr, cc = cp.indices((rows, cols))

    # Compute Euclidean distance map (fully vectorized on GPU)
    dist_map = cp.sqrt((rr - centroid_row) ** 2 + (cc - centroid_col) ** 2)

    # Convert to micrometers (scaling)
    dist_map_um = dist_map * pixel_size_um

    # Ignore background pixels (set to NaN)
    dist_map_um[data == background_label] = cp.nan  

    return dist_map_um

def find_trigger_pixel_old(region="soma", data=None, soma_mask=None, thick_branch_mask=None, thin_process_mask=None):
    if region == "soma":
        pixel_indices = cp.column_stack(cp.where(soma_mask))
    elif region == "thick_branch":
        pixel_indices = cp.column_stack(cp.where(thick_branch_mask))
    elif region == "thin_process":
        pixel_indices = cp.column_stack(cp.where(thin_process_mask))
    else:
        raise ValueError("Invalid region specified. Choose from 'soma', 'thick_branch', or 'thin_process'.")

    if pixel_indices.shape[0] == 0:
        raise ValueError(f"No valid pixels found in the selected region: {region}")

    # Select a random pixel within the chosen region (Convert to NumPy for `random.randint()`)
    pixel_indices_np = pixel_indices.get()  # Convert CuPy array to NumPy array
    selected_pixel = tuple(pixel_indices_np[random.randint(0, pixel_indices_np.shape[0] - 1)])

    print(f"Trigger pixel selected in {region}: {selected_pixel}")
    return selected_pixel

def show_trigger_pixel_old(region, segmented_data, soma_mask, thick_branch_mask, thin_process_mask):
    # Find the trigger pixel
    trigger_pixel = find_trigger_pixel(region=region, 
                                       data=segmented_data, 
                                       soma_mask=soma_mask, 
                                       thick_branch_mask=thick_branch_mask, 
                                       thin_process_mask=thin_process_mask)

    if trigger_pixel is None:
        print("Warning: No trigger pixel found. Using fallback coordinates (10, 10).")
        trigger_pixel = (10, 10)  # Placeholder coordinates
    
    # Extract row and col from the trigger pixel
    trigger_row, trigger_col = trigger_pixel

    # Convert segmented data from CuPy to NumPy for plotting
    segmented_data_np = cp.asnumpy(segmented_data)

    # Plot the segmented astrocyte data
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_data_np, cmap="gray")  # Display image in grayscale

    # Overlay the trigger pixel as a red dot
    plt.scatter(trigger_col, trigger_row, color='red', marker='o', s=100, label="Trigger Pixel")

    # Add labels and title
    plt.title(f"Trigger Pixel in {region}")
    plt.legend()
    plt.show()

def find_soma_centroid(soma_mask):
    pixel_indices = cp.column_stack(cp.where(soma_mask))
    if pixel_indices.shape[0] == 0:
        raise ValueError("No valid soma pixels found.")
    
    # Compute centroid using mean of x and y coordinates
    centroid = cp.mean(pixel_indices, axis=0)
    centroid_pixel = tuple(cp.round(centroid).astype(int).tolist())
    return centroid_pixel

def find_region_pixel(region="soma", soma_mask=None, thick_branch_mask=None, thin_process_mask=None, label="", debug_mode=False):
    if region == "soma":
        selected_pixel = find_soma_centroid(soma_mask)

    elif region == "thick_branch":
        # Find thick branches **only** connected to soma
        labeled_thick, num_features = ndi.label(cp.asnumpy(thick_branch_mask))
        soma_labels = np.unique(labeled_thick[cp.asnumpy(soma_mask)])

        # Extract only thick branches that **share a label** with the soma
        valid_thick_pixels = np.column_stack(np.where(np.isin(labeled_thick, soma_labels)))
        if valid_thick_pixels.shape[0] == 0:
            raise ValueError("No valid thick branch pixels connected to the soma.")

        # Select a **random** pixel within connected thick branches
        selected_pixel = tuple(valid_thick_pixels[np.random.randint(0, valid_thick_pixels.shape[0])])

    elif region == "thin_process":
        pixel_indices = cp.column_stack(cp.where(thin_process_mask))
        if pixel_indices.shape[0] == 0:
            raise ValueError("No valid thin process pixels found.")

        # Select a **random** pixel within the thin process region
        pixel_indices_np = pixel_indices.get()  # Convert CuPy array to NumPy
        selected_pixel = tuple(pixel_indices_np[np.random.randint(0, pixel_indices_np.shape[0])])

    else:
        raise ValueError("Invalid region specified. Choose from 'soma', 'thick_branch', or 'thin_process'.")
    if debug_mode:
        print(f"{label} selected in {region}: {selected_pixel}")
    return selected_pixel

def show_trigger_pixel(segmented_data, trigger_pixel):
    if trigger_pixel is None:
        print("Warning: No trigger pixel found. Using fallback coordinates (10, 10).")
        trigger_pixel = (10, 10)  # Placeholder coordinates

    # Convert segmented data from CuPy to NumPy for plotting
    segmented_data_np = cp.asnumpy(segmented_data)

    # Extract row and col from the trigger pixel
    trigger_row, trigger_col = trigger_pixel

    # Plot the segmented astrocyte data
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_data_np, cmap="gray")  # Display image in grayscale

    # Overlay the trigger pixel as a red dot
    plt.scatter(trigger_col, trigger_row, color='red', marker='o', s=100, label="Trigger Pixel")

    # Add labels and title
    plt.title(f"Trigger Pixel at {trigger_pixel}")
    plt.legend()
    plt.show()
