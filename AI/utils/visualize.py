import cv2

def overlay_attention_map(original_image, attention_map, output_path, alpha=0.5):
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) * 255
    
    # Resize attention map to match the original image size
    attention_map = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))

    # Apply colormap to attention map for better visualization (optional)
    attention_colormap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

    # Blend attention map with the original image
    overlaid_image = cv2.addWeighted(original_image, 1-alpha, attention_colormap, alpha, 0)

    # Save the overlaid image
    cv2.imwrite(output_path, overlaid_image)
