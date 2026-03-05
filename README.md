# Mandelbrot Visualizer

A performance study of five progressively optimised implementations for rendering the Mandelbrot set as an animated zoom sequence. The project benchmarks pure Python, JIT compilation, quadtree subdivision, multiprocessing, and GPU acceleration.

## Implementations

| File | Description |
|---|---|
| `baseline_mandelbrot.py` | Pure Python nested loops — reference implementation |
| `numba_only_mandelbrot.py` | Numba JIT compilation with `prange` CPU parallelism |
| `quad_and_numba_mandelbrot.py` | Mariani–Silver quadtree with Numba-compiled border checks |
| `multi_numba_quad_mandelbrot.py` | Mariani–Silver quadtree distributed across CPU cores via `multiprocessing` |
| `gpu_accelerated.py` | CuPy `ElementwiseKernel` CUDA kernel with async I/O |

## Benchmark Results

| Implementation | Time (s) | Speedup |
|---|---|---|
| Baseline (Pure Python) | 69.92 | 1.0× |
| Quadtree + Numba (Mariani–Silver) | 1.99 | 35.1× |
| Numba JIT (Parallel) | 0.86 | 81.3× |
| Multiprocessing + Quadtree | 0.88 | 79.5× |
| GPU (CuPy CUDA Kernel) | 0.05 | 1,533× |

*Standard config: 1000×1000, 256 max iterations, 5 frames, zoom 1×–100×, file I/O excluded.*

## Requirements

- Python 3.12+
- `numpy`
- `matplotlib`
- `numba` (for JIT, quadtree, and multiprocessing implementations)
- `cupy` + CUDA toolkit (for GPU implementation)

Install CPU dependencies:
```bash
pip install numpy matplotlib numba
```

Install GPU dependency (match your CUDA version):
```bash
pip install cupy-cuda12x
```

## Usage

All scripts share the same CLI arguments:

```
--width       Image width in pixels  (default: 1000)
--height      Image height in pixels (default: 1000)
--max-iter    Maximum iterations per pixel (default: 256)
--frames      Number of zoom frames (default: 5)
--start-zoom  Starting zoom level (default: 1.0)
--end-zoom    Ending zoom level (default: 100.0)
--no-output   Skip saving frames (useful for benchmarking)
```

### Examples

```bash
# Run the baseline
python baseline_mandelbrot.py

# Benchmark Numba JIT without saving output
python numba_only_mandelbrot.py --no-output

# Run GPU implementation with custom resolution
python gpu_accelerated.py --width 2000 --height 2000 --frames 10

# Run multiprocessing quadtree for a deep zoom
python multi_numba_quad_mandelbrot.py --max-iter 1000 --frames 10 --end-zoom 1000
```

Output frames are saved to a per-implementation directory (`frames/`, `frames_numba/`, `frames_quadtree/`, `frames_multi/`, `frames_gpu/`).

## Zoom Target

All implementations zoom into the coordinate $c = -0.7436438870371587 + 0.1318259042i$, a region of high boundary complexity on the edge of the Mandelbrot set.
