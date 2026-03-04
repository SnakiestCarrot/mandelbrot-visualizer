import time
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# configuration
WIDTH, HEIGHT = 1000, 1000 
MAX_ITER = 256

# coordinates to zoom into 
CENTER_REAL = -0.7436438870371587
CENTER_IMAG = 0.13182590420531197
START_ZOOM = 1.0
END_ZOOM = 100.0
FRAMES = 5

def mandelbrot_kernel(c, max_iter):
    """
    Calculates stability of a single point c.
    Returns the number of iterations before escape.
    """
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_frame(center_r, center_i, zoom, width, height, max_iter):
    """
    Generates a single frame at a specific zoom level.
    """
    # Calculate the viewport size based on zoom
    # At zoom 1, the view is roughly 4 units wide.
    # At zoom 1000, it is 0.004 units wide.
    view_width = 4.0 / zoom
    view_height = 4.0 / zoom
    
    # Calculate boundaries
    r_min = center_r - (view_width / 2.0)
    i_min = center_i - (view_height / 2.0)
    
    dx = view_width / width
    dy = view_height / height

    # Use a flat list (Baseline style - inefficient)
    # We use 'uint16' conceptually, but Python lists store full objects (overhead)
    result = [0] * (width * height)

    start_time = time.time()
    
    # Nested loops (The Baseline Bottleneck)
    for y in range(height):
        imag = i_min + y * dy
        for x in range(width):
            real = r_min + x * dx
            c = complex(real, imag)
            
            color = mandelbrot_kernel(c, max_iter)
            result[y * width + x] = color
            
    end_time = time.time()
    return result, end_time - start_time

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

    print(f"Starting Zoom Simulation: {frames} frames.")
    print(f"Center: {CENTER_REAL} + {CENTER_IMAG}i")
    print("-" * 40)

    total_time = 0

    if not args.no_output:
        if not os.path.exists('frames'):
            os.makedirs('frames')

    # Zoom Logic: Logarithmic scale looks smoother than linear
    zooms = np.geomspace(args.start_zoom, args.end_zoom, frames)

    for i, current_zoom in enumerate(zooms):
        print(f"Frame {i+1}/{frames} | Zoom: {current_zoom:.2f}x ...", end=" ", flush=True)

        # Run generation
        data, duration = generate_frame(
            CENTER_REAL, CENTER_IMAG, current_zoom, width, height, max_iter
        )

        total_time += duration
        print(f"Done in {duration:.4f}s")

        if not args.no_output:
            # Reshape for matplotlib
            matrix = np.array(data).reshape((height, width))
            plt.imsave(f'frames/frame_{i:03d}.png', matrix, cmap='magma')

    print("-" * 40)
    print(f"Total Simulation Time: {total_time:.4f}s")

if __name__ == "__main__":
    main()