import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import multiprocessing
from numba import jit

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000  # High res to make the CPU work hard
MAX_ITER = 13000
FRAMES = 60                 # More frames = better parallel demonstration
START_ZOOM = 1.0
END_ZOOM = 1000000000.0       # Deep zoom
CENTER_REAL = -0.7436438870371587
CENTER_IMAG = 0.13182590420531197
SHOW_BLOCKS = False         # Turn off for production run to save disk I/O time

# --- 1. THE KERNEL (COMPILED) ---
@jit(nopython=True, fastmath=True)
def mandelbrot_kernel(c, max_iter):
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# --- 2. THE QUADTREE LOGIC (Rectangle-Aware) ---
@jit(nopython=True)
def check_border_rect(x_start, y_start, w, h, r_min, i_min, dx, dy, max_iter):
    x_end = x_start + w
    y_end = y_start + h
    c_ref = complex(r_min + x_start * dx, i_min + y_start * dy)
    ref_color = mandelbrot_kernel(c_ref, max_iter)
    
    # Top/Bottom
    for x in range(x_start, x_end):
        c_top = complex(r_min + x * dx, i_min + y_start * dy)
        if mandelbrot_kernel(c_top, max_iter) != ref_color: return False, -1
        c_bot = complex(r_min + x * dx, i_min + (y_end - 1) * dy)
        if mandelbrot_kernel(c_bot, max_iter) != ref_color: return False, -1

    # Left/Right
    for y in range(y_start + 1, y_end - 1):
        c_left = complex(r_min + x_start * dx, i_min + y * dy)
        if mandelbrot_kernel(c_left, max_iter) != ref_color: return False, -1
        c_right = complex(r_min + (x_end - 1) * dx, i_min + y * dy)
        if mandelbrot_kernel(c_right, max_iter) != ref_color: return False, -1
            
    return True, ref_color

def mariani_silver(x, y, w, h, img_buffer, block_list, 
                   r_min, i_min, dx, dy, max_iter):
    if w <= 4 or h <= 4:
        for iy in range(y, y + h):
            for ix in range(x, x + w):
                if ix < WIDTH and iy < HEIGHT:
                    c = complex(r_min + ix * dx, i_min + iy * dy)
                    img_buffer[iy, ix] = mandelbrot_kernel(c, max_iter)
        return

    is_solid, color = check_border_rect(x, y, w, h, r_min, i_min, dx, dy, max_iter)

    if is_solid:
        img_buffer[y:y+h, x:x+w] = color
        if SHOW_BLOCKS:
            block_list.append((x, y, w, h))
    else:
        half_w = w // 2
        rem_w = w - half_w
        half_h = h // 2
        rem_h = h - half_h
        
        mariani_silver(x, y, half_w, half_h, img_buffer, block_list, r_min, i_min, dx, dy, max_iter)
        mariani_silver(x + half_w, y, rem_w, half_h, img_buffer, block_list, r_min, i_min, dx, dy, max_iter)
        mariani_silver(x, y + half_h, half_w, rem_h, img_buffer, block_list, r_min, i_min, dx, dy, max_iter)
        mariani_silver(x + half_w, y + half_h, rem_w, rem_h, img_buffer, block_list, r_min, i_min, dx, dy, max_iter)

# --- 3. THE WORKER FUNCTION ---
# This runs on a separate CPU core. It needs all data required to make ONE frame.
def process_frame(args):
    """
    args is a tuple: (frame_index, zoom, r_min, i_min, dx, dy)
    """
    idx, zoom, r_min, i_min, dx, dy = args
    
    # Each process needs its own buffer (processes don't share memory by default)
    local_buffer = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    local_blocks = []
    
    # Run the Quadtree Algorithm
    start_t = time.time()
    mariani_silver(0, 0, WIDTH, HEIGHT, local_buffer, local_blocks, r_min, i_min, dx, dy, MAX_ITER)
    calc_time = time.time() - start_t
    
    # Save the image immediately (Distributed I/O)
    # This prevents the Main process from getting bottled up saving 50 images at the end.
    my_dpi = 100
    fig_w, fig_h = WIDTH / my_dpi, HEIGHT / my_dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=my_dpi)
    ax.imshow(local_buffer, cmap='magma', origin='upper', interpolation='nearest')
    ax.axis('off')
    plt.subplots_adjust(0,0,1,1,0,0)
    ax.margins(0,0)
    
    if SHOW_BLOCKS:
        for (bx, by, bw, bh) in local_blocks:
            if bw > 4 or bh > 4:
                rect = patches.Rectangle((bx, by), bw, bh, linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.5)
                ax.add_patch(rect)
                
    filename = f"frames_multi/frame_{idx:03d}.png"
    plt.savefig(filename, dpi=my_dpi, pad_inches=0)
    plt.close(fig)
    
    return idx, calc_time

# --- 4. MAIN CONTROLLER ---
def main():
    print(f"--- STARTING MULTIPROCESSING RENDER ---")
    print(f"Cores Available: {multiprocessing.cpu_count()}")
    print(f"Task: Render {FRAMES} frames from Zoom {START_ZOOM}x to {END_ZOOM}x")
    
    if not os.path.exists('frames_multi'):
        os.makedirs('frames_multi')

    # Prepare the Job Queue
    tasks = []
    zooms = np.geomspace(START_ZOOM, END_ZOOM, FRAMES)
    
    for i, z in enumerate(zooms):
        view_w, view_h = 4.0/z, 4.0/z
        r_min = CENTER_REAL - view_w/2.0
        i_min = CENTER_IMAG - view_h/2.0
        dx, dy = view_w/WIDTH, view_h/HEIGHT
        
        # Pack arguments
        tasks.append((i, z, r_min, i_min, dx, dy))

    global_start = time.time()

    # Create the Pool
    # We leave 1 core free for the OS if you want, or use all.
    # processes=None defaults to os.cpu_count()
    with multiprocessing.Pool() as pool:
        # imap_unordered is BEST for dynamic load balancing.
        # It yields results as soon as they finish, regardless of order.
        results = pool.imap_unordered(process_frame, tasks)
        
        # Monitor progress
        for i, (frame_idx, duration) in enumerate(results):
            print(f"[{i+1}/{FRAMES}] Finished Frame {frame_idx:03d} in {duration:.2f}s")

    global_end = time.time()
    total_time = global_end - global_start
    
    print("-" * 40)
    print(f"Total Batch Time: {total_time:.2f}s")
    print(f"Average Time per Frame: {total_time/FRAMES:.2f}s")
    print(f"Theoretical Speedup: Check against your Serial Baseline!")

if __name__ == "__main__":
    main()