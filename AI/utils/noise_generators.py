import cv2
import numpy as np

def shake_image(image_path, max_shift=10):
    # Load the image
    image = cv2.imread(image_path)
    
    # Generate random shift values
    shift_x = np.random.randint(-max_shift, max_shift)
    shift_y = np.random.randint(-max_shift, max_shift)
    
    # Define the transformation matrix for shifting
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Apply the transformation
    shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return shifted_image

if __name__ == "__main__":
    # Example usage
    image_path = 'utils/curby.jpg'
    max_shift = 50

    shake_image(image_path, max_shift)
