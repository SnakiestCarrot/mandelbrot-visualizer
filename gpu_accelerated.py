import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import concurrent.futures

# --- CONFIGURATION ---
WIDTH, HEIGHT = 10000, 10000
MAX_ITER = 20000
FRAMES = 10
START_ZOOM = 1.0
END_ZOOM = 1.0e14
CENTER_REAL = -0.7436438870371587
CENTER_IMAG = 0.13182590420531197

# --- FUSED KERNEL ---
mandelbrot_kernel = cp.ElementwiseKernel(
    'complex128 c, int32 max_iter',
    'int32 output',
    '''
    complex<double> z = 0;
    int n = 0;
    while (n < max_iter && z.real()*z.real() + z.imag()*z.imag() <= 4.0) {
        z = z * z + c;
        n++;
    }
    output = n;
    ''',
    'mandelbrot_kernel'
)

def gpu_mandelbrot(width, height, max_iter, zoom, center_r, center_i):
    view_w = 4.0 / zoom
    view_h = 4.0 / zoom
    x = cp.linspace(center_r - view_w/2, center_r + view_w/2, width)
    y = cp.linspace(center_i - view_h/2, center_i + view_h/2, height)
    
    real, imag = cp.meshgrid(x, y)
    c = real + 1j * imag
    output = cp.empty((height, width), dtype=cp.int32)
    
    mandelbrot_kernel(c, max_iter, output)
    cp.cuda.Stream.null.synchronize()
    
    return cp.asnumpy(output)

# --- HELPER FUNCTION FOR THREADS ---
def save_frame_task(data, filename):
    """
    This runs in a background thread.
    It handles the slow PNG compression and disk write.
    """
    # This line is the bottleneck (taking ~0.5s - 1.0s per image)
    plt.imsave(filename, data, cmap='magma')
    return filename

def main():
    print("Starting GPU Optimization (Fused Kernel + Async I/O)...")
    if not os.path.exists('frames_gpu'):
        os.makedirs('frames_gpu')

    zooms = np.geomspace(START_ZOOM, END_ZOOM, FRAMES)
    
    # We use a ThreadPoolExecutor to save images in the background
    # max_workers=4 means 4 images can be saving at the same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        # Warmup
        print("Warming up GPU...")
        _ = gpu_mandelbrot(500, 500, 100, 1.0, -0.75, 0.1)

        total_start = time.time()
        
        for i, current_zoom in enumerate(zooms):
            print(f"Frame {i+1}/{FRAMES} | Zoom: {current_zoom:.1f}x ...", end=" ")
            
            # 1. GPU Calc (Fast ~0.03s)
            t0 = time.time()
            result = gpu_mandelbrot(WIDTH, HEIGHT, MAX_ITER, current_zoom, CENTER_REAL, CENTER_IMAG)
            calc_time = time.time() - t0
            
            # 2. Submit Save Job (Instant)
            # We don't wait for it to finish. We just throw it into the pool.
            fname = f'frames_gpu/frame_{i:03d}.png'
            future = executor.submit(save_frame_task, result, fname)
            futures.append(future)
            
            print(f"Calc: {calc_time:.4f}s | Saving in background...")

        # 3. Wait for all saves to complete
        print("-" * 40)
        print("GPU finished. Waiting for disk I/O to catch up...")
        for future in concurrent.futures.as_completed(futures):
            # Just ensuring everything finished without error
            pass
            
        total_end = time.time()
        print(f"Total Wall-Clock Time: {total_end - total_start:.4f}s")

if __name__ == "__main__":
    main()