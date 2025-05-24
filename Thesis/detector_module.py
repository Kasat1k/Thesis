import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, convolve
from scipy.special import gamma, factorial

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.asarray(img) / 255.0
    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    Image.fromarray((img_gray * 255).astype(np.uint8)).save('static/uploaded.png')
    return img_gray

def apply_gaussian_filter(img, sigma):
    return gaussian_filter(img, sigma=sigma)

def fractional_derivative_rl(img, alpha, axis, kernel_size):
    half = kernel_size // 2
    kernel = np.zeros(kernel_size)
    for k in range(-half, half + 1):
        coeff = gamma(alpha + 1) / (factorial(abs(k)) * gamma(alpha - abs(k) + 1))
        kernel[k + half] = ((-1) ** k) * coeff
    kernel = kernel.reshape(1, -1) if axis == 0 else kernel.reshape(-1, 1)
    return convolve(img, kernel, mode='nearest')


def caputo_fabrizio_derivative(img, alpha, axis):
    h = 1  
    M = 1  

    f_current = img
    f_prev = np.roll(img, 1, axis=axis)

    a1 = (1 - alpha) / M
    a2 = (3 * alpha * h) / (2 * M)
    a3 = (1 - alpha) / M
    a4 = (alpha * h) / (2 * M)

    return a1 * (np.roll(f_current, -1, axis=axis) - np.roll(f_current, 1, axis=axis)) + \
           a2 * (np.roll(f_current, -1, axis=axis) - np.roll(f_prev, 1, axis=axis)) + \
           a3 * (f_current - f_prev) + a4 * (np.roll(f_current, 1, axis=axis) - np.roll(f_prev, 2, axis=axis))

def compute_gradients_rl(img, alpha, kernel_size):
    fx = fractional_derivative_rl(img, alpha, 0, kernel_size)
    fy = fractional_derivative_rl(img, alpha, 1, kernel_size)
    return np.sqrt(fx**2 + fy**2), np.arctan2(fy, fx)

def compute_gradients_cf(img, alpha):
    fx = caputo_fabrizio_derivative(img, alpha, 0)
    fy = caputo_fabrizio_derivative(img, alpha, 1)
    return np.sqrt(fx**2 + fy**2), np.arctan2(fy, fx)

def non_maximal_suppression(G, theta):
    M, N = G.shape
    suppressed = np.zeros((M, N))
    angle = np.rad2deg(theta) % 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            a = angle[i, j]
            q = r = 0
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif 22.5 <= a < 67.5:
                q, r = G[i+1, j-1], G[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q, r = G[i+1, j], G[i-1, j]
            else:
                q, r = G[i-1, j-1], G[i+1, j+1]

            if G[i, j] >= q and G[i, j] >= r:
                suppressed[i, j] = G[i, j]
    return suppressed

def auto_threshold_percentile(G, low_p=30, high_p=97):
    high = np.percentile(G, high_p)
    low = np.percentile(G, low_p)
    return max(min(low, high * 0.5), 0.001), min(high, 0.8)

def hysteresis_thresholding(G, low, high):
    M, N = G.shape
    out = np.zeros((M, N), dtype=np.uint8)
    strong = G >= high
    weak = (G >= low) & (G < high)
    out[strong] = 255
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if weak[i, j] and np.any(strong[i-1:i+2, j-1:j+2]):
                out[i, j] = 255
    return out

def auto_select_parameters(img_path, kernel_options=[5, 7, 9, 11]):
    img = load_image(img_path)
    best_alpha, best_sigma, best_kernel, best_score = 0.5, 1.0, 5, -1
    for k in kernel_options:
        for alpha in [0.3, 0.5, 0.7]:
            for sigma in [0.5, 1.0, 2.0]:
                smoothed = apply_gaussian_filter(img, sigma)
                G, _ = compute_gradients_rl(smoothed, alpha, k)
                score = np.std(G) / (np.mean(G) + 1e-5)
                if score > best_score:
                    best_alpha, best_sigma, best_kernel, best_score = alpha, sigma, k, score
    return best_alpha, best_sigma, best_kernel


def run_edge_detection(img_path, alpha, sigma):
    img = load_image(img_path)
    _, _, kernel_size = auto_select_parameters(img_path)
    filtered = apply_gaussian_filter(img, sigma)

    G_rl, theta_rl = compute_gradients_rl(filtered, alpha, kernel_size)
    G_cf, theta_cf = compute_gradients_cf(filtered, alpha)

    G_rl = non_maximal_suppression(G_rl, theta_rl)
    G_cf = non_maximal_suppression(G_cf, theta_cf)

    low_rl, high_rl = auto_threshold_percentile(G_rl)
    low_cf, high_cf = auto_threshold_percentile(G_cf)

    edges_rl = hysteresis_thresholding(G_rl, low_rl, high_rl)
    edges_cf = hysteresis_thresholding(G_cf, low_cf, high_cf)
    edges_cv = cv2.Canny((filtered * 255).astype(np.uint8), 50, 150)

    Image.fromarray(edges_rl).save('static/result_rl.png')
    Image.fromarray(edges_cf).save('static/result_cf.png')
    Image.fromarray(edges_cv).save('static/result_cv.png')
