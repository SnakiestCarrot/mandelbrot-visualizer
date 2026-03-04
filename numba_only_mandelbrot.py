import time
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# --- NEW IMPORT ---
from numba import jit, prange

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
MAX_ITER = 256

CENTER_REAL = -0.7436438870371587
CENTER_IMAG = 0.13182590420531197
START_ZOOM = 1.0
END_ZOOM = 100.0
FRAMES = 5

# --- OPTIMIZATION 1: JIT COMPILATION ---
# @jit: Tells Numba to compile this function to Machine Code (like C++).
# nopython=True: Forces it to fail if it can't optimize everything (ensures speed).
# fastmath=True: Allows slight precision loss for speed (optional but recommended).
@jit(nopython=True)
def mandelbrot_kernel(c, max_iter):
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# --- OPTIMIZATION 1: PARALLEL LOOP ---
# parallel=True: Enables multi-core CPU usage automatically.
@jit(nopython=True, parallel=True)
def generate_frame_optimized(center_r, center_i, zoom, width, height, max_iter):
    view_width = 4.0 / zoom
    view_height = 4.0 / zoom
    
    r_min = center_r - (view_width / 2.0)
    i_min = center_i - (view_height / 2.0)
    
    dx = view_width / width
    dy = view_height / height

    # We use a NumPy array directly now (Numba loves NumPy)
    result = np.zeros((height, width), dtype=np.int32)

    # prange: "Parallel Range".
    # This splits the loop chunks across all your CPU cores.
    for y in prange(height):
        imag = i_min + y * dy
        for x in range(width):
            real = r_min + x * dx
            c = complex(real, imag)
            
            # Call the compiled kernel
            result[y, x] = mandelbrot_kernel(c, max_iter)
            
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-output', action='store_true', help='Skip saving output files (for benchmarking)')
    parser.add_argument('--width', type=int, default=WIDTH, help='Image width in pixels')
    parser.add_argument('--height', type=int, default=HEIGHT, help='Image height in pixels')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER, help='Maximum iterations per pixel')
    parser.add_argument('--frames', type=int, default=FRAMES, help='Number of zoom frames')
    parser.add_argument('--start-zoom', type=float, default=START_ZOOM, help='Starting zoom level')
    parser.add_argument('--end-zoom', type=float, default=END_ZOOM, help='Ending zoom level')
    args = parser.parse_args()

    width = args.width
    height = args.height
    max_iter = args.max_iter
    frames = args.frames

    print(f"Starting Numba Optimization: {frames} frames.")
    print(f"Center: {CENTER_REAL} + {CENTER_IMAG}i")

    # --- WARMUP COMPILATION ---
    # Numba compiles the function the FIRST time it runs.
    # We run a dummy call so the compilation time doesn't mess up our measurements.
    print("Compiling JIT functions (Warmup)...")
    _ = generate_frame_optimized(CENTER_REAL, CENTER_IMAG, 1.0, 100, 100, 100)
    print("Compilation Complete. Starting Benchmark.")
    print("-" * 40)

    total_time = 0
    if not args.no_output:
        if not os.path.exists('frames_numba'):
            os.makedirs('frames_numba')

    zooms = np.geomspace(args.start_zoom, args.end_zoom, frames)

    for i, current_zoom in enumerate(zooms):
        start_time = time.time()

        # Run optimized generation
        data = generate_frame_optimized(
            CENTER_REAL, CENTER_IMAG, current_zoom, width, height, max_iter
        )

        duration = time.time() - start_time
        total_time += duration

        print(f"Frame {i+1}/{frames} | Zoom: {current_zoom:.2f}x | Time: {duration:.4f}s")

        if not args.no_output:
            plt.imsave(f'frames_numba/frame_{i:03d}.png', data, cmap='magma')

    print("-" * 40)
    print(f"Total Simulation Time: {total_time:.4f}s")

if __name__ == "__main__":
    main()