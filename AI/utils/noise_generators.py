import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance

# 노이즈 추가 함수들
def add_lighting_variation(image, scale_range=(0.5, 1.5)):
    noisy = np.copy(image)
    scale_factor = np.random.uniform(*scale_range)
    noisy = image * scale_factor
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_dust_and_particles(image, dust_prob=0.1, particle_prob=0.05):
    noisy = np.copy(image)
    dust_mask = np.random.rand(*noisy.shape[:2]) < dust_prob
    dusty_pixels = np.random.randint(150, 200, noisy.shape[2])  
    noisy[dust_mask] = dusty_pixels

    particle_mask = np.random.rand(*noisy.shape[:2]) < particle_prob
    particle_pixels = np.random.randint(50, 100, noisy.shape[2])  
    noisy[particle_mask] = particle_pixels

    print('add dust and particlae', type(noisy))
    return noisy.astype(np.uint8)

def add_random_rotation(image, angle_range=(-10, 10)):
    angle = np.random.uniform(*angle_range)
    rotated = ndimage.rotate(image, angle, reshape=False)
    return rotated.astype(np.uint8)

def add_blur(image, blur_prob=0.2):
    if np.random.rand() < blur_prob:
        kernel_size = np.random.choice([3, 5, 7])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred.astype(np.uint8)
    else:
        return image

def add_color_jitter(image, brightness_range=(-20, 20), contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2)):
    noisy = image.copy()
    noisy = noisy.squeeze()  # Remove singleton dimensions
    noisy = noisy.transpose(1, 2, 0)  # Reshape to (224, 224, 1)
    
    # Scale the pixel values to the range [0, 255] if needed
    # You may need to adjust this based on the actual range of your data
    noisy = (noisy * 255).astype(np.uint8)
    
    noisy = Image.fromarray(noisy)
    enhancer = ImageEnhance.Brightness(noisy)
    noisy = enhancer.enhance(np.random.uniform(*brightness_range))

    enhancer = ImageEnhance.Contrast(noisy)
    noisy = enhancer.enhance(np.random.uniform(*contrast_range))

    enhancer = ImageEnhance.Color(noisy)
    noisy = enhancer.enhance(np.random.uniform(*saturation_range))

    # PIL to numpy
    noisy = np.array(noisy)
    noisy = noisy.transpose(2, 0, 1)  # Reshape to (1, 224, 224)
    noisy = noisy.unsqueeze(0)  # Reshape to (1, 1, 224, 224)
    
    return noisy

if __name__ == "__main__":
    # Example usage
    image_path = 'utils/curby.jpg'
    max_shift = 50

    # shake_image(image_path, max_shift)
