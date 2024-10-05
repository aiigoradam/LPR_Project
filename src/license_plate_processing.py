import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import numpy as np

# =====================================
# License Plate Generation
# =====================================

import random
from PIL import Image, ImageDraw, ImageFont

def generate_plate_number():
    """
    Generates a random plate number.
    
    Returns:
        str: The generated plate number in the format "XX-XXX-XX".
    """
    return f"{random.randint(10, 99)}-{random.randint(100, 999)}-{random.randint(10, 99)}"

def create_license_plate():
    """
   Creates a license plate image with a random number.
    
    Returns:
        tuple: A tuple containing the license plate image and a list of corner coordinates.
            - image (PIL.Image): The generated license plate image.
            - corners (list of tuples): The corner coordinates of the license plate [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """
    plate_number = generate_plate_number()
    width, height = 400, 100
    background_color = (255, 203, 9)

    text_color = (0, 0, 0)
    text_size = 82

    # Create the image
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Load a font
    try:
        font = ImageFont.truetype("bahnschrift.ttf", text_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text bounding box
    text_bbox = draw.textbbox((0, 0), plate_number, font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2 - text_bbox[0]
    text_y = (height - text_height) // 2 - text_bbox[1]

    # Draw the text
    draw.text((text_x, text_y), plate_number, fill=text_color, font=font)

    # Create a new image with black background
    new_width, new_height = int(width * 1.5), int(height * 2)
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Calculate the corners of the original image in the new image
    x1 = (new_width - width) // 2
    y1 = (new_height - height) // 2
    x2 = x1 + width
    y2 = y1 + height

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # Paste the original image in the center
    new_image.paste(image, (x1, y1))

    return new_image, corners

# =====================================
# Image Warping and Dewarping
# =====================================

import cv2
import numpy as np

def calculate_center(src_points):
    '''
     Calculate the center of the source points.

    Args:
        src_points (numpy.ndarray): Coordinates of the corners of the input image (in clockwise order).
                    numpy array of shape (4, 2): [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]   

    Returns:
        numpy.ndarray: The center coordinates [center_x, center_y].
    '''
    center_x = np.mean(src_points[:, 0])
    center_y = np.mean(src_points[:, 1])
    return np.array([center_x, center_y])

def warp_image(image, src_points, alpha, beta, f = 400):
    '''
     Warp the input image using a perspective transformation based on the rotation angles alpha and beta.

    Args:
        image (numpy.ndarray): The input image to be warped.
        src_points (numpy.ndarray): Coordinates of the corners of the input image (in clockwise order).
                    numpy array of shape (4, 2): [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        alpha (float): Rotation angle around the y-axis (horizontal rotation) in degrees.
        beta (float): Rotation angle around the x-axis (vertical rotation) in degrees.
        f (int, optional): Focal length (distance from the camera to the image plane). Defaults to 400.

    Returns:
        tuple: A tuple containing the warped image and the destination points.
            - warped_image (numpy.ndarray): The warped image.
            - dst_points (numpy.ndarray): The new coordinates of the corners after warping. Same format as src_points.
    '''
    # Convert degrees to radians
    alpha_rad = np.deg2rad(alpha)  # Rotation angle around the y-axis
    beta_rad = np.deg2rad(beta)    # Rotation angle around the x-axis

    # Rotation matrices around the x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(beta_rad), -np.sin(beta_rad)],
        [0, np.sin(beta_rad), np.cos(beta_rad)]
    ])

    # Rotation matrices around the y-axis
    R_y = np.array([
        [np.cos(alpha_rad), 0, np.sin(alpha_rad)],
        [0, 1, 0],
        [-np.sin(alpha_rad), 0, np.cos(alpha_rad)]
    ])

    # Combined rotation
    R = np.dot(R_y, R_x)

    # Calculate the center of the source points
    center = calculate_center(src_points)

    # Calculate new positions after applying the rotation
    dst_points = [] 
 
    for point in src_points:
        # Convert to homogeneous coordinates (3D)
        x, y = point - center
        z = 0
        vec = np.dot(R, np.array([x, y, z]))

        # Project back to 2D
        x_proj = center[0] + f * (vec[0] / (f + vec[2]))
        y_proj = center[1] + f * (vec[1] / (f + vec[2]))

        dst_points.append([x_proj, y_proj])

    # Ensure src_points and dst_points are float32
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    # Get the perspective transformation matrix and apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)

    return warped_image, dst_points

def dewarp_image(image, src_points, dst_points):
    '''
    Dewarp the input image using a perspective transformation based on the source and destination points.

    Args:
        image (numpy.ndarray): The input image to be dewarped.
        src_points (numpy.ndarray): Coordinates of the corners of the input image (in clockwise order).
        dst_points (numpy.ndarray): Coordinates of the corners of the warped image (in clockwise order).
                    numpy array of shape (4, 2): [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    
    Returns:
        image (numpy.ndarray): The dewarped image.
    '''

    # Ensure src_points and dst_points are float32
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    # Get the inverse perspective transformation matrix and apply it
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    dewarped_image = cv2.warpPerspective(image, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)

    return dewarped_image

def add_gaussian_noise(image, dst_points, mean=0, stddev=30):
    """
    Add Gaussian noise to the L channel of the HSL color space of the license plate region in the image.

    Args:
        image (numpy.ndarray): The input image in RGB format.
        dst_points (numpy.ndarray): The destination points (warped coordinates) of the license plate.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.
        stddev (float, optional): Standard deviation of the Gaussian noise. Defaults to 30.

    Returns:
        numpy.ndarray: The image with added Gaussian noise in the L channel of the license plate region.
    """
    # Create a mask based on the destination points
    mask = np.zeros(image.shape[:2], dtype=np.uint8) 
    cv2.fillPoly(mask, [np.int32(dst_points)], 255) 

    # Convert the image to HLS color space
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H, L, S = cv2.split(hls_image)

    # Apply noise only to the L channel within the mask
    noise = np.random.normal(mean, stddev, L.shape).astype(np.float32)
    L_noisy = np.where(mask == 255, L.astype(np.float32) + noise, L.astype(np.float32))
    L_noisy = np.clip(L_noisy, 0, 255).astype(np.uint8)

    # Merge the channels back
    noisy_hls_image = cv2.merge([H, L_noisy, S])
    
    # Convert back to RGB color space
    noisy_rgb_image = cv2.cvtColor(noisy_hls_image, cv2.COLOR_HLS2RGB)
    
    return noisy_rgb_image

# =====================================
# Main Function for Testing
# =====================================

def main():
    # Generate the license plate image and get the source points
    license_plate_image, src_points = create_license_plate()
    license_plate_image = cv2.cvtColor(np.array(license_plate_image), cv2.COLOR_RGB2BGR)
    
    # Define the parameters
    alpha = 65  # Rotation angle around the y-axis
    beta = 65   # Rotation angle around the x-axis
    f = 400  # Assumed focal length

    # Warp the image
    warped_image, dst_points = warp_image(license_plate_image, np.array(src_points), alpha, beta, f)

    # Convert the warped image from BGR to RGB for displaying with matplotlib
    warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    # Add Gaussian noise to the L channel in HSL color space of the license plate region
    noisy_warped_image_rgb = add_gaussian_noise(warped_image_rgb, dst_points)

    # Convert the noisy image back to BGR for dewarping with OpenCV
    noisy_warped_image_bgr = cv2.cvtColor(noisy_warped_image_rgb, cv2.COLOR_RGB2BGR)

    # Dewarp the noisy image
    dewarped_image = dewarp_image(noisy_warped_image_bgr, np.array(src_points), dst_points)
    dewarped_image_rgb = cv2.cvtColor(dewarped_image, cv2.COLOR_BGR2RGB)

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    # Display the original image
    original_image_rgb = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_image_rgb)
    axes[0].axis('on')
    axes[0].set_title('Original Image')

    # Display the warped image with noise
    axes[1].imshow(noisy_warped_image_rgb)
    axes[1].axis('on')
    axes[1].set_title(f'Warped Image with Noise (alpha={alpha}, beta={beta})')

    # Display the dewarped image
    axes[2].imshow(dewarped_image_rgb)
    axes[2].axis('on')
    axes[2].set_title(f'Dewarped Image (alpha={alpha}, beta={beta})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()