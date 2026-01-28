import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
from numba import jit

# --- CONFIGURATION ---
WIDTH, HEIGHT = 700, 700
MAX_ITER = 10000
FRAMES = 30
START_ZOOM = 1.0
END_ZOOM = 100000000.0
CENTER_REAL = -0.13856524454488
CENTER_IMAG = -0.64935990748190

# Set this to True to overlay the red optimization boxes on the output images
# This proves your "Data Layout" optimization visually.
SHOW_BLOCKS = True

# --- KERNEL (Numba Accelerated) ---
# We keep the math fast so the recursion is the main logic bottleneck we test
@jit(nopython=True, fastmath=True)
def mandelbrot_kernel(c, max_iter):
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# --- QUADTREE LOGIC ---
def check_border_rect(x_start, y_start, w, h, r_min, i_min, dx, dy, max_iter):
    """
    Checks the perimeter of a RECTANGLE (w, h).
    """
    x_end = x_start + w
    y_end = y_start + h
    
    # Reference color (Top-Left)
    c_ref = complex(r_min + x_start * dx, i_min + y_start * dy)
    ref_color = mandelbrot_kernel(c_ref, max_iter)
    
    # 1. Top and Bottom Rows (Iterate Width)
    for x in range(x_start, x_end):
        # Top Edge
        c_top = complex(r_min + x * dx, i_min + y_start * dy)
        if mandelbrot_kernel(c_top, max_iter) != ref_color: return False, -1
        
        # Bottom Edge (y_end - 1)
        c_bot = complex(r_min + x * dx, i_min + (y_end - 1) * dy)
        if mandelbrot_kernel(c_bot, max_iter) != ref_color: return False, -1

    # 2. Left and Right Columns (Iterate Height)
    # Start at y+1 and end at y-1 to avoid re-checking corners
    for y in range(y_start + 1, y_end - 1):
        # Left Edge
        c_left = complex(r_min + x_start * dx, i_min + y * dy)
        if mandelbrot_kernel(c_left, max_iter) != ref_color: return False, -1
        
        # Right Edge (x_end - 1)
        c_right = complex(r_min + (x_end - 1) * dx, i_min + y * dy)
        if mandelbrot_kernel(c_right, max_iter) != ref_color: return False, -1
            
    return True, ref_color

def mariani_silver(x, y, w, h, img_buffer, block_list, 
                   r_min, i_min, dx, dy, max_iter):
    """
    Recursive function using WIDTH and HEIGHT separately.
    """
    # Base case: small enough?
    if w <= 4 or h <= 4:
        for iy in range(y, y + h):
            for ix in range(x, x + w):
                if ix < WIDTH and iy < HEIGHT:
                    c = complex(r_min + ix * dx, i_min + iy * dy)
                    img_buffer[iy, ix] = mandelbrot_kernel(c, max_iter)
        return

    # Check optimization
    is_solid, color = check_border_rect(x, y, w, h, r_min, i_min, dx, dy, max_iter)

    if is_solid:
        # Fill optimization
        img_buffer[y:y+h, x:x+w] = color
        if SHOW_BLOCKS:
            # Save tuple as (x, y, width, height)
            block_list.append((x, y, w, h))
    else:
        # --- RECTANGULAR SPLIT LOGIC ---
        half_w = w // 2
        rem_w = w - half_w  # Remainder width
        
        half_h = h // 2
        rem_h = h - half_h  # Remainder height
        
        # We spawn 4 children with their specific dimensions
        
        # Top-Left (half_w, half_h)
        mariani_silver(x, y, half_w, half_h, 
                       img_buffer, block_list, r_min, i_min, dx, dy, max_iter)
        
        # Top-Right (rem_w, half_h)
        mariani_silver(x + half_w, y, rem_w, half_h, 
                       img_buffer, block_list, r_min, i_min, dx, dy, max_iter)
        
        # Bottom-Left (half_w, rem_h)
        mariani_silver(x, y + half_h, half_w, rem_h, 
                       img_buffer, block_list, r_min, i_min, dx, dy, max_iter)
        
        # Bottom-Right (rem_w, rem_h)
        mariani_silver(x + half_w, y + half_h, rem_w, rem_h, 
                       img_buffer, block_list, r_min, i_min, dx, dy, max_iter)

def generate_quadtree_frame(zoom, r_min, i_min, dx, dy):
    img_buffer = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    block_list = []
    
    start = time.time()
    # Start recursion from full image size
    mariani_silver(0, 0, WIDTH, HEIGHT, img_buffer, block_list, r_min, i_min, dx, dy, MAX_ITER)
    duration = time.time() - start
    
    return img_buffer, block_list, duration

def main():
    print(f"Starting Quadtree Zoom Sequence...")
    if not os.path.exists('frames_quadtree'):
        os.makedirs('frames_quadtree')

    zooms = np.geomspace(START_ZOOM, END_ZOOM, FRAMES)
    total_time = 0

    for i, current_zoom in enumerate(zooms):
        # Calculate viewport
        view_w = 4.0 / current_zoom
        view_h = 4.0 / current_zoom
        r_min = CENTER_REAL - (view_w / 2.0)
        i_min = CENTER_IMAG - (view_h / 2.0)
        dx = view_w / WIDTH
        dy = view_h / HEIGHT

        print(f"Frame {i+1}/{FRAMES} | Zoom: {current_zoom:.1f}x ...", end=" ", flush=True)
        
        # Run Algorithm
        data, blocks, duration = generate_quadtree_frame(current_zoom, r_min, i_min, dx, dy)
        total_time += duration
        print(f"Time: {duration:.4f}s | Optimization Blocks: {len(blocks)}")

        # Save Image with Matplotlib (to draw the boxes)
        my_dpi = 100
        fig_width = WIDTH / my_dpi
        fig_height = HEIGHT / my_dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=my_dpi)
        
        # 'nearest' interpolation prevents blurring edges
        ax.imshow(data, cmap='magma', origin='upper', interpolation='nearest')
        
        # Remove all axes, borders, and whitespace
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.margins(0,0)

        if SHOW_BLOCKS:
            # UPDATED: Unpack 4 values now (x, y, width, height)
            for (bx, by, bw, bh) in blocks:
                # Check dimensions independently
                if bw > 4 or bh > 4: 
                    # UPDATED: Pass width (bw) and height (bh) to the Rectangle
                    rect = patches.Rectangle((bx, by), bw, bh, 
                                           linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.6)
                    ax.add_patch(rect)
        
        plt.savefig(f'frames_quadtree/frame_{i:03d}.png', dpi=my_dpi, pad_inches=0)
        plt.close(fig)

    print("-" * 40)
    print(f"Total Time: {total_time:.4f}s")
    print("Check the /frames_quadtree folder for the red-boxed images.")

if __name__ == "__main__":
    main()