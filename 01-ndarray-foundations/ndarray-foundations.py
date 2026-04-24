import numpy as np

# --- 1. System Config & Initialization ---
# Create base 3D RGB array (512px square)
ocean = np.zeros((512, 512, 3), dtype='uint8')

# c# Generate constant value block via broadcasting
land_mass = np.ones((100, 100, 3), dtype='uint8') * 150 

# Simulated complex signal data (Real + Imaginary)
radar_interference = np.empty((512, 512), dtype=np.complex128)
radar_interference.real = np.random.random((512, 512))
radar_interference.imag = np.random.random((512, 512))

# --- 2. Data Audit Utility ---
def check_satellite_data(name, data):
    print(f"\n--- Report for: {name} ---")
    print(f"Resolution: {data.shape[0]}x{data.shape[1]}")
    if data.ndim == 3:
        print(f"Color Channels: {data.shape[2]}")
    print(f"Total Pixels: {data.size // (data.shape[2] if data.ndim==3 else 1)}") 
    print(f"Memory Layout: {data.ndim}D Array")
    print(f"Precision: {data.dtype}")

# --- 3. Image Processing Pipeline ---
canvas = ocean.copy()

# Slice-based painting (Top-left Red / Bottom-half Green)
canvas[0:100, 0:100, 0] = 255 
canvas[256:, :, 1] = 255 
# Vectorized brightness scaling (float conversion to prevent overflow)
bright_canvas = canvas.astype(np.float32) * 1.2
bright_canvas = np.clip(bright_canvas, 0, 255)

# Boolean masking: Find and re-color unassigned pixels (Black -> Grey)
is_black = np.all(canvas == 0, axis=2)
num_black = np.sum(is_black)
canvas[is_black] = 50

# --- 4. Validation Output ---
check_satellite_data("Final Canvas", canvas)
check_satellite_data("Radar System", radar_interference)

print("\n--- Summary of Operations ---")
print(f"Red square check (50,50): {canvas[50, 50]}")
print(f"Green half check (400,400): {canvas[400, 400]}")
print(f"Brightened red pixel sample: {bright_canvas[50, 50, 0]}")
print(f"Remaining black pixels (before grey-fill): {num_black}")
