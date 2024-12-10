import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm 

# =====================================
# License Plate Generation
# =====================================

def create_license_plate(width, height, text_size):
    """
    Creates a license plate image with a random number.

    Args:
        width (int): The width of the license plate image.
        height (int): The height of the license plate image.
        text_size (int): The font size of the license plate number.

    Returns:
        tuple: A tuple containing:
            - image (PIL.Image): The generated license plate image.
            - corners (list of tuples): The corner coordinates of the license plate [(x, y), (x + w, y), (x + w, y + h), (x, y + h)].
            - plate_number (str): The generated plate number.
            - bboxes (list of tuples): List of bounding boxes for each digit [(x, y, w, h), ...].
    """
    plate_number = " ".join([str(random.randint(0, 9)) for _ in range(6)])  # Generate a random plate number

    background_color = (255, 203, 9)
    text_color = (0, 0, 0)

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

    # Calculate the position and size
    x = (new_width - width) // 2
    y = (new_height - height) // 2
    w = width
    h = height

    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    # Paste the original image in the center
    new_image.paste(image, (x, y))

    plate_number = plate_number.replace(" ", "")  # Plate number without spaces for metadata

    # Convert PIL Image to numpy array
    cv_image = np.array(new_image)
    # Convert RGB to BGR for OpenCV
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Crop the grayscale image to the license plate area
    gray_plate = gray[y:y + h, x:x + w]

    # Threshold the image to get binary image
    _, thresh_plate = cv2.threshold(gray_plate, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    offset = 1
    bboxes = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        cx, cy, cw, ch = cx - offset, cy - offset, cw + offset, ch + offset
        bboxes.append((cx, cy, cw, ch))

    # Sort bounding boxes from left to right
    bboxes = sorted(bboxes, key=lambda bbox: bbox[0])

    return new_image, corners, plate_number, bboxes

# =====================================
# Image Warping and Dewarping
# =====================================

def warp_image(image, src_points, alpha, beta, f):
    """
    Applies a perspective warp to the input image based on specified rotation angles.

    Args:
        image (numpy.ndarray): The input image to be warped.
        src_points (numpy.ndarray): Coordinates of the four corners of the input image, in clockwise order.
        alpha (float): Rotation angle around the y-axis (horizontal rotation), in degrees.
        beta (float): Rotation angle around the x-axis (vertical rotation), in degrees.
        f (int, optional): Focal length, representing the distance from the camera to the image plane. 

    Returns:
        tuple: A tuple containing:
            - warped_image (numpy.ndarray): The resulting warped image.
            - dst_points (numpy.ndarray): The new coordinates of the four corners after warping, in the same format as src_points.
    """

    # Convert RGB to BGR for OpenCV operations
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
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
    center_x = np.mean(src_points[:, 0])
    center_y = np.mean(src_points[:, 1])
    center = np.array([center_x, center_y])

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
    
    # Convert BGR back to RGB
    warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    return warped_image_rgb, dst_points

def dewarp_image(image, src_points, dst_points):
    """
    Dewarp the input image using a perspective transformation based on the source and destination points.

    Args:
        image (numpy.ndarray): The input image to be dewarped.
        src_points (numpy.ndarray): Coordinates of the corners of the original image (in clockwise order).
        dst_points (numpy.ndarray): Coordinates of the corners of the warped image (in clockwise order).
                  
    Returns:
        numpy.ndarray: The dewarped image.
    """
    # Convert RGB to BGR for OpenCV operations
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure src_points and dst_points are float32
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    # Get the inverse perspective transformation matrix and apply it
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    dewarped_image = cv2.warpPerspective(image, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    
    # Convert BGR back to RGB
    dewarped_image_rgb = cv2.cvtColor(dewarped_image, cv2.COLOR_BGR2RGB)

    return dewarped_image_rgb

def crop_to_original_size(image, original_width, original_height):
    """
    Crops the image back to the original license plate size.

    Args:
        image (np.ndarray): The image to be cropped.
        original_width (int): The original width of the license plate.
        original_height (int): The original height of the license plate.

    Returns:
        np.ndarray: The cropped image.
    """
    height, width = image.shape[:2]

    # Calculate cropping coordinates to get the central region
    left = (width - original_width) // 2
    top = (height - original_height) // 2
    right = left + original_width
    bottom = top + original_height

    # Crop and return the image
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def simulate_luminance_chroma_noise_with_artifacts(image, lum_std=20, chroma_std=10):
    """
    Simulate Gaussian blur, vertical double edges, luminance/chroma noise, and mosaic noise applied 
    at the end with a random chance per tile.

    Args:
        image (ndarray): Input image in RGB format.
        lum_std (int): Standard deviation for luminance noise.
        chroma_std (int): Standard deviation for chroma noise.

    Returns:
        ndarray: Processed image in RGB format.
    """
    # Step 1: Convert RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Step 2: Apply Gaussian blur
    blur_kernel_size = (3, 3)  # Kernel size for blur
    blurred_image = cv2.GaussianBlur(image_bgr, blur_kernel_size, sigmaX=1, sigmaY=1)

    # Step 3: Apply vertical double-edge effect
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    double_edges = cv2.filter2D(blurred_image, -1, kernel)
    blurred_image = cv2.addWeighted(blurred_image, 1.0, double_edges, 0.2, 0)

    # Step 4: Add luminance and chroma noise in YCrCb space
    ycrcb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Add Gaussian noise to luminance (Y) and chroma (Cr, Cb) channels
    lum_noise = np.random.normal(0, lum_std, y.shape).astype(np.float32)
    cr_noise = np.random.normal(0, chroma_std, cr.shape).astype(np.float32)
    cb_noise = np.random.normal(0, chroma_std, cb.shape).astype(np.float32)

    y_noisy = np.clip(y.astype(np.float32) + lum_noise, 0, 255).astype(np.uint8)
    cr_noisy = np.clip(cr.astype(np.float32) + cr_noise, 0, 255).astype(np.uint8)
    cb_noisy = np.clip(cb.astype(np.float32) + cb_noise, 0, 255).astype(np.uint8)

    # Merge noisy channels and convert back to BGR
    ycrcb_noisy = cv2.merge([y_noisy, cr_noisy, cb_noisy])
    noisy_bgr = cv2.cvtColor(ycrcb_noisy, cv2.COLOR_YCrCb2BGR)

    # Step 5: Apply mosaic noise with 35% probability for tiles of size 4
    h, w = noisy_bgr.shape[:2]
    tile_size = 4

    for yy in range(0, h, tile_size):
        for xx in range(0, w, tile_size):
            if random.random() < 0.35:  # 35% chance to apply mosaic noise
                tile_h = min(tile_size, h - yy)
                tile_w = min(tile_size, w - xx)

                # Apply mosaic effect with block size 2
                tile = noisy_bgr[yy:yy + tile_h, xx:xx + tile_w]
                small_h, small_w = max(1, tile_h // 2), max(1, tile_w // 2)
                mosaic_tile = cv2.resize(tile, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                mosaic_tile = cv2.resize(mosaic_tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

                noisy_bgr[yy:yy + tile_h, xx:xx + tile_w] = mosaic_tile

    # Step 6: Convert BGR back to RGB before returning
    noisy_rgb = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2RGB)
    return noisy_rgb

# =====================================
# Manage Existing Data
# =====================================

def manage_existing_data(output_dir, num_samples):
    """
    Ensures that only the specified number of samples are present in the output directory by deleting extra files.
    
    Args:
        output_dir (str): Directory where images and metadata are stored.
        num_samples (int): Number of new samples to generate.
    """
    # Delete all existing files if they exceed the desired count
    for filename in os.listdir(output_dir):
        # Check if file is an original image, distorted image, or metadata file with an index beyond the current range
        if (filename.startswith("original_") or filename.startswith("distorted_") or filename.startswith("metadata_")):
            try:
                # Extract the index from the filename
                index = int(filename.split("_")[1].split(".")[0])
                # Delete files with an index equal to or beyond `num_samples`
                if index >= num_samples:
                    os.remove(os.path.join(output_dir, filename))
            except (IndexError, ValueError):
                # If parsing fails, ignore the file
                continue


# =====================================
# Dataset Generation with Cropping
# =====================================

def generate_dataset(num_samples, output_dir, original_width, original_height, text_size, seed=None):
    """
    Generates and saves the dataset, including cropping of distorted images to original plate size.
    
    Args:
        num_samples (int): Number of image pairs to generate.
        output_dir (str): Directory to save generated images and metadata.
        noise_level_range (tuple): Range of noise levels (stddev) for luminance/chroma noise.
        original_width (int): Original width of the license plate.
        original_height (int): Original height of the license plate.
        text_size (int): Font size of the license plate number.
        seed (int): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Manage existing data before generating new data
    manage_existing_data(output_dir, num_samples)
    
    for idx in tqdm(range(num_samples), desc="Generating samples"):
        # Generate a clean license plate
        original_image_pil, src_points, plate_number, digit_bboxes = create_license_plate(original_width, original_height, text_size)
        original_image_rgb = np.array(original_image_pil)

        # Sample angles freely
        alpha = random.uniform(80, 90)  # First angle from 80 to 89 inclusive
        beta = random.uniform(0, 90)    # Second angle from 0 to 89 inclusive

        # Randomly flip signs to include negative angles
        if random.random() < 0.5:
            alpha = -alpha
        if random.random() < 0.5:
            beta = -beta

        # Random noise level in the range 10 to 30
        noise_level = random.uniform(10, 30)

        # Warp and add noise to the image
        warped_image, dst_points = warp_image(original_image_rgb, np.array(src_points), alpha, beta, f=original_width)
        noisy_image = simulate_luminance_chroma_noise_with_artifacts(
            warped_image, lum_std=noise_level, chroma_std=noise_level / 4
        )
        distorted_image = dewarp_image(noisy_image, src_points, dst_points)

        # Crop both the original and distorted images back to the original license plate size
        cropped_original_image = crop_to_original_size(original_image_rgb, original_width, original_height)
        cropped_distorted_image = crop_to_original_size(distorted_image, original_width, original_height)

        # Save images
        original_path = os.path.join(output_dir, f"original_{idx}.png")
        distorted_path = os.path.join(output_dir, f"distorted_{idx}.png")
        Image.fromarray(cropped_original_image).save(original_path)
        Image.fromarray(cropped_distorted_image).save(distorted_path)

        # Save metadata
        metadata = {
            "idx": idx,
            "plate_number": plate_number,
            "alpha": alpha,
            "beta": beta,
            "noise_level": noise_level,
            "digit_bboxes": digit_bboxes
        }
        metadata_path = os.path.join(output_dir, f"metadata_{idx}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)


def generate_test_dataset(output_dir, original_width, original_height, text_size, seed=None):
    """
    Generates a test dataset covering the full range of angles:
    - alpha in [0..89]
    - beta in [0..89]

    For each (alpha, beta) pair, one image is created:
        - Noise: stddev in [10..30]

    This ensures a comprehensive coverage of the angle space.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Define full angle ranges
    alpha_range = range(0, 90)  # 0 to 89 inclusive
    beta_range = range(0, 90)   # 0 to 89 inclusive

    parameter_combinations = [(a, b) for a in alpha_range for b in beta_range]

    idx = 0
    for alpha, beta in tqdm(parameter_combinations, desc="Generating full-range test dataset"):
        # Create the license plate
        original_image_pil, src_points, plate_number, digit_bboxes = create_license_plate(original_width, original_height, text_size)
        original_image_rgb = np.array(original_image_pil)

        # Warp the image
        warped_image, dst_points = warp_image(original_image_rgb, np.array(src_points), alpha, beta, f=original_width)

        # Generate noise in the range [10, 30]
        noise_level = random.uniform(10, 30)
        noisy_image = simulate_luminance_chroma_noise_with_artifacts(
            warped_image, lum_std=noise_level, chroma_std=noise_level / 4
        )
        distorted_image = dewarp_image(noisy_image, src_points, dst_points)
        cropped_original_image = crop_to_original_size(original_image_rgb, original_width, original_height)
        cropped_distorted_image = crop_to_original_size(distorted_image, original_width, original_height)

        # Save images and metadata
        original_path = os.path.join(output_dir, f"original_{idx}.png")
        distorted_path = os.path.join(output_dir, f"distorted_{idx}.png")
        Image.fromarray(cropped_original_image).save(original_path)
        Image.fromarray(cropped_distorted_image).save(distorted_path)
        
        metadata = {
            "idx": idx,
            "plate_number": plate_number,
            "alpha": alpha,
            "beta": beta,
            "noise_level": noise_level,
            "digit_bboxes": digit_bboxes
        }
        metadata_path = os.path.join(output_dir, f"metadata_{idx}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        idx += 1

    print("Full-range test dataset generation completed.")
               
# =====================================
# Main Function 
# =====================================

def main():
    num_samples = 10240  # Number of training images to generate for the dataset
    unique_samples = 32  # Number of unique images to generate for experiments
    output_dir = "data"  # Directory to save the dataset images
    test_output_dir = "data_test"  # Directory to save the test dataset images
    unique_output_dir = "data_unique"  # Directory to save the unique experimental images
    seed = 42  # Random seed for reproducibility

    factor = 2  # Scaling factor for image size (0: 128x32, 1: 256x64, 2: 512x128, 3: 1024x256)
    scale = 2 ** factor # Calculate the scaling multiplier

    # Scale the height, width, and text size
    f = int(128 * scale)         # Scaled width (focal length)
    h = int(32 * scale)          # Scaled height
    text_size = int(25 * scale)  # Scaled text size

    # Generate the training dataset
    generate_dataset(num_samples, output_dir, f, h, text_size, seed=seed)
    
    # Keep the unique dataset for experiments
    generate_dataset(unique_samples, unique_output_dir, f, h, text_size, seed=seed+28)
    
    # Generate the test dataset using the new function
    generate_test_dataset(output_dir=test_output_dir, original_width=f, original_height=h, text_size=text_size, seed=seed+73)
    
if __name__ == "__main__":
    main()

