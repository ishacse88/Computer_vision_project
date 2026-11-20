import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

def apply_brightness(pil_img, factor):
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)

def apply_contrast(pil_img, factor):
    enhancer = ImageEnhance.Contrast(pil_img)
    return enhancer.enhance(factor)

def apply_blur(pil_img, radius):
    if radius <= 0:
        return pil_img
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_noise(pil_img, noise_level):
    if noise_level <= 0:
        return pil_img
    img_array = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, noise_level * 25, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_crop_zoom(pil_img, crop_percent):
    if crop_percent <= 0:
        return pil_img
    w, h = pil_img.size
    crop_pixels_w = int((w * crop_percent) / 200.0)
    crop_pixels_h = int((h * crop_percent) / 200.0)
    
    left = crop_pixels_w
    top = crop_pixels_h
    right = w - crop_pixels_w
    bottom = h - crop_pixels_h
    
    cropped = pil_img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)