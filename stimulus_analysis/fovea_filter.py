import numpy as np
import cv2

def retina_transform(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Perform Fourier Transform
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Create a mask to mimic the retina's frequency capture
    mask = np.ones((h, w))
    center_x, center_y = h // 2, w // 2
    
    # Define the scaling factor for the visual field
    scaling_factor = 30  # entire image spans 30 degrees of the visual field
    
    for i in range(h):
        for j in range(w):
            distance_from_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            angle = np.arctan2(i - center_x, j - center_y)
            
            # Convert distance to degrees
            distance_in_degrees = (distance_from_center / np.sqrt(center_x**2 + center_y**2)) * scaling_factor
            
            # Reduce high-frequency components based on distance from the center
            if distance_in_degrees > 1:
                attenuation_factor = np.exp(-distance_in_degrees)
                mask[i, j] *= attenuation_factor
    
    # Apply the mask
    f_transform_shifted *= mask
    
    # Inverse Fourier Transform
    f_transform_unshifted = np.fft.ifftshift(f_transform_shifted)
    img_transformed = np.fft.ifft2(f_transform_unshifted)
    img_transformed = np.abs(img_transformed).astype(np.uint8)
    
    return img_transformed

# Test the function
transformed_img = retina_transform('path/to/your/image.jpg')
cv2.imwrite('transformed_image.jpg', transformed_img)
