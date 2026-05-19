import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time

@jit(nopython=True, parallel=True, fastmath=True)
def generate_mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
    img = np.full((height, width), max_iter, dtype=np.int32)
    dx = (xmax - xmin) / (width - 1)
    dy = (ymax - ymin) / (height - 1)
    
    for i in prange(height):
        cy = ymin + i * dy
        for j in prange(width):
            cx = xmin + j * dx
            zx, zy = 0.0, 0.0
            iteration = 0
            
            while zx*zx + zy*zy <= 4.0 and iteration < max_iter:
                xtemp = zx*zx - zy*zy + cx
                zy = 2.0 * zx * zy + cy
                zx = xtemp
                iteration += 1
                
            if iteration < max_iter:
                img[i, j] = iteration
                
    return img

@jit(nopython=True, parallel=True, fastmath=True)
def generate_julia_numba(xmin, xmax, ymin, ymax, width, height, max_iter, cx, cy):
    img = np.full((height, width), max_iter, dtype=np.int32)
    dx = (xmax - xmin) / (width - 1)
    dy = (ymax - ymin) / (height - 1)
    
    for i in prange(height):
        y0 = ymin + i * dy
        for j in prange(width):
            x0 = xmin + j * dx
            zx = x0
            zy = y0
            iteration = 0
            
            while zx*zx + zy*zy <= 4.0 and iteration < max_iter:
                xtemp = zx*zx - zy*zy + cx
                zy = 2.0 * zx * zy + cy
                zx = xtemp
                iteration += 1
                
            if iteration < max_iter:
                img[i, j] = iteration
                
    return img

def main():
    WIDTH, HEIGHT = 1920, 1080
    MAX_ITER = 500
    cx, cy = -0.7, 0.27015
    
    print("[*] Computing Mandelbrot...")
    mandel_img = generate_mandelbrot_numba(-2.0, 0.5, -1.25, 1.25, WIDTH, HEIGHT, MAX_ITER)
    
    print("[*] Computing Julia...")
    julia_img = generate_julia_numba(-1.5, 1.5, -1.0, 1.0, WIDTH, HEIGHT, MAX_ITER, cx, cy)

    # Отрисовка
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), facecolor='#0d0d11')
    
    ax[0].imshow(mandel_img, cmap='twilight_shifted', extent=[-2.0, 0.5, -1.25, 1.25])
    ax[0].set_title("Mandelbrot Set (Numba JIT)", color='white', fontsize=16, pad=15)
    ax[0].axis('off')
    
    ax[1].imshow(julia_img, cmap='magma', extent=[-1.5, 1.5, -1.0, 1.0])
    ax[1].set_title(f"Julia Set (c = {cx} + {cy}i)", color='white', fontsize=16, pad=15)
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
