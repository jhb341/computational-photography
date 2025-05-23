import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(img, title):
    if img.ndim == 3:
        height, width, _ = img.shape
        # BGR -> RGB 변환 (OpenCV는 기본 BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.xticks([0, width-1])
        plt.yticks([0, height-1])
        plt.title(title)
        plt.show()
    else:
        height, width = img.shape
        plt.imshow(img, cmap='gray')
        plt.xticks([0, width-1])
        plt.yticks([0, height-1])
        plt.title(title)
        plt.show()

def gaussian_lpf(shape, co):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    u_c, v_c = P//2, Q//2
    d_square = (U - u_c)**2 + (V - v_c)**2
    
    g_lpf = np.exp(-d_square / (2 * (co**2)))
    return g_lpf