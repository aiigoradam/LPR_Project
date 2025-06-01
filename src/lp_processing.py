# src/lp_processing.py

import os
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import qmc
from tqdm import tqdm

# =====================================
# License Plate Generation
# =====================================


def create_license_plate(plate_width=256, plate_height=64, font_size=100, num_chars=6, fill_density=0.9):
    plate_number = "".join(str(random.randint(0, 9)) for _ in range(num_chars))

    base_plate = Image.new("RGB", (plate_width, plate_height), (255, 203, 9))
    drawer = ImageDraw.Draw(base_plate)
    try:
        font = ImageFont.truetype("bahnschrift.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    usable_width = plate_width * fill_density
    cell_width = int(usable_width / num_chars)
    margin_x = int((plate_width - cell_width * num_chars) / 2)
    margin_y = 10
    center_y = plate_height / 2

    for i, char in enumerate(plate_number):
        left, top, right, bottom = drawer.textbbox((0, 0), char, font=font)
        char_w, char_h = right - left, bottom - top

        cell_left = margin_x + i * cell_width
        center_x = cell_left + cell_width / 2

        x = center_x - char_w / 2 - left
        y = center_y - char_h / 2 - top
        drawer.text((x, y), char, fill=(0, 0, 0), font=font)

    digit_bboxes = [
        (margin_x + i * cell_width, margin_y, cell_width, plate_height - 2 * margin_y) for i in range(num_chars)
    ]

    padded_w = int(plate_width * 1.5)
    padded_h = int(plate_height * 2)
    padded_img = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))

    pad_x = (padded_w - plate_width) // 2
    pad_y = (padded_h - plate_height) // 2
    padded_img.paste(base_plate, (pad_x, pad_y))

    plate_corners = [
        (pad_x, pad_y),
        (pad_x + plate_width - 1, pad_y),
        (pad_x + plate_width - 1, pad_y + plate_height - 1),
        (pad_x, pad_y + plate_height - 1),
    ]

    return padded_img, plate_corners, plate_number, digit_bboxes


# =====================================
# Image Warping and Distortion
# =====================================


def warp_image(image, src_points, alpha, beta, f):
    # Convert RGB to BGR for OpenCV operations
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure src_points is a numpy array
    src_points = np.array(src_points, dtype=np.float32)

    # Convert degrees to radians
    alpha_rad = np.deg2rad(alpha)  # Rotation angle around the y-axis
    beta_rad = np.deg2rad(beta)  # Rotation angle around the x-axis

    # Rotation matrices around the x-axis
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(beta_rad), -np.sin(beta_rad)],
            [0, np.sin(beta_rad), np.cos(beta_rad)],
        ]
    )

    # Rotation matrices around the y-axis
    R_y = np.array(
        [
            [np.cos(alpha_rad), 0, np.sin(alpha_rad)],
            [0, 1, 0],
            [-np.sin(alpha_rad), 0, np.cos(alpha_rad)],
        ]
    )

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
    warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_AREA)

    # Convert BGR back to RGB
    warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    return warped_image_rgb, dst_points


def dewarp_image(image, src_points, dst_points):
    # Convert RGB to BGR for OpenCV operations
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure src_points and dst_points are numpy arrays
    src_points = np.array(src_points, dtype=np.float32)

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
    height, width = image.shape[:2]

    # Calculate cropping coordinates to get the central region
    left = (width - original_width) // 2
    top = (height - original_height) // 2
    right = left + original_width
    bottom = top + original_height

    # Crop and return the image
    cropped_image = image[top:bottom, left:right]
    return cropped_image


def simulate_noise(image, debug_dir=None):
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        step_idx = 0
        cv2.imwrite(
            os.path.join(debug_dir, f"step_{step_idx}_input.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        step_idx += 1

    # apply double edge detection
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    double_edges = cv2.filter2D(image, -1, kernel)
    blurred_image = cv2.addWeighted(image, 0.7, double_edges, 0.3, 0)
    image_bgr = cv2.cvtColor(blurred_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if debug_dir:
        cv2.imwrite(
            os.path.join(debug_dir, f"step_{step_idx}_double_edges+shadows.png"),
            image_bgr,
        )
        step_idx += 1

    # Simplified ISP pipeline
    # Auto white balance (random scaling of R and B channels)
    b, g, r = cv2.split(image_bgr)
    # Apply random scaling to R and B channels
    r_scale = np.random.uniform(0.8, 1.2)  # Random factor between 0.8 and 1.2
    b_scale = np.random.uniform(0.8, 1.2)  # Independent random factor for blue
    r_adjusted = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    b_adjusted = np.clip(b * b_scale, 0, 255).astype(np.uint8)
    # Merge channels back together
    image_bgr = cv2.merge([b_adjusted, g, r_adjusted])

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"step_{step_idx}_white_balance.png"), image_bgr)
        step_idx += 1

    # Denoising (light Gaussian blur)
    image_bgr = cv2.GaussianBlur(image_bgr, (3, 3), sigmaX=1.0, sigmaY=1.0)
    if debug_dir:
        cv2.imwrite(
            os.path.join(debug_dir, f"step_{step_idx}_denoised_GaussianBlur.png"),
            image_bgr,
        )
        step_idx += 1

    # JPEG compression simulation quality 20 block size 8x8
    _, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"step_{step_idx}_jpeg_compressed.png"), image_bgr)
        step_idx += 1
    # Add Contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = -50  # Brightness control (0-100)
    image_bgr = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"step_{step_idx}_contrast.png"), image_bgr)
        step_idx += 1
    # Add 1% Gaussian noise
    std_dev = 0.01 * 255
    noise = np.random.normal(0, std_dev, image_bgr.shape)
    image_bgr = np.clip(image_bgr + noise, 0, 255).astype(np.uint8)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"step_{step_idx}_gaussian_noise.png"), image_bgr)
        step_idx += 1
    # Convert back to RGB and return
    noisy_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"step_{step_idx}_final_output.png"), noisy_rgb)

    return noisy_rgb


# =====================================
# Helpers
# =====================================


def manage_existing_data(output_dir):
    """
    Delete all files in output_dir.
    """
    if not os.path.isdir(output_dir):
        return

    for fname in os.listdir(output_dir):
        path = os.path.join(output_dir, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def downscale_plate(image, bboxes, target_size):
    h, w = image.shape[:2]
    tw, th = target_size
    sx, sy = tw / w, th / h
    blurred_image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5, sigmaY=0.5)
    down_img = cv2.resize(blurred_image, (tw, th), interpolation=cv2.INTER_AREA)
    down_bboxes = [(int(x * sx), int(y * sy), int(w_ * sx), int(h_ * sy)) for (x, y, w_, h_) in bboxes]
    return down_img, down_bboxes


def sample_alpha_beta_sobol(m_power, alpha_min, alpha_max, beta_min, beta_max, s, k, seed=None):

    N = 2**m_power

    # precompute alpha‐marginal CDF
    na = 600
    nb = 6000
    alphas = np.linspace(alpha_min, alpha_max, na)
    dalpha = alphas[1] - alphas[0]

    beta_grid = np.linspace(beta_min, beta_max, nb)
    dbeta = beta_grid[1] - beta_grid[0]

    # location function
    def m_func(a):
        return 84.4086 - np.exp(0.9873 * a - 84.0848)

    # build un-normalized joint g on (alpha,beta) grid
    A, B = np.meshgrid(alphas, beta_grid, indexing="xy")
    alpha_norm = (A - alpha_min) / (alpha_max - alpha_min)
    beta_norm = B / beta_max
    g = beta_norm ** (k * (1 - alpha_norm)) / (1 + np.exp((B - m_func(A)) / s))

    # marginal f_alpha(alpha)
    g_int = np.trapz(g, beta_grid, axis=0)
    f_alpha = g_int / np.trapz(g_int, alphas)
    cdf_alpha = np.cumsum(f_alpha) * dalpha
    cdf_alpha /= cdf_alpha[-1]

    # one Sobol draw in 3D
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed)
    u = sobol.random_base2(m=m_power)  # shape (N,3)
    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]

    samples = np.zeros((N, 2))

    for i in range(N):
        # invert alpha‐CDF
        a = np.interp(u0[i], cdf_alpha, alphas)

        # build & invert beta|alpha
        raw = 1 / (1 + np.exp((beta_grid - m_func(a)) / s))
        weight = (beta_grid / beta_max) ** (k * (1 - (a - alpha_min) / (alpha_max - alpha_min)))
        g_b = raw * weight
        Zb = np.trapz(g_b, beta_grid)
        cdf_b = np.cumsum(g_b / Zb) * dbeta
        b = np.interp(u1[i], cdf_b, beta_grid)

        # pick one of 8 regions
        rid = min(int(u2[i] * 8), 7)
        if rid < 4:
            # f(alpha,beta)
            sx = 1 if (rid & 1) == 0 else -1
            sy = 1 if (rid >> 1 & 1) == 0 else -1
            samples[i] = (sx * a, sy * b)
        else:
            # f(beta,alpha)
            r2 = rid - 4
            sx = 1 if (r2 & 1) == 0 else -1
            sy = 1 if (r2 >> 1 & 1) == 0 else -1
            samples[i] = (sx * b, sy * a)

    return samples[:, 0], samples[:, 1]


# =====================================
# Dataset Generation
# =====================================


def generate_dataset(
    output_path,
    width,
    height,
    font_size,
    seed=None,
    output_size=None,
    mode="random",  # "random" or "grid"
    num_samples=None,  # total samples for random, per-angle cap for grid
):

    # 1) Seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 2) Build list of (alpha, beta) angles
    if mode == "random":
        if num_samples is None:
            raise ValueError("num_samples must be set when mode='random'")
        sobol_power = int(np.log2(num_samples))
        alpha_arr, beta_arr = sample_alpha_beta_sobol(
            m_power=sobol_power, alpha_min=80.0, alpha_max=89.0, beta_min=0.0, beta_max=89.0, s=1.0, k=1.5, seed=seed
        )
        angle_list = [(round(alpha_arr[i], 1), round(beta_arr[i], 1)) for i in range(num_samples)]

    elif mode == "grid":
        if num_samples is None:
            raise ValueError("num_samples must be set when mode='grid'")
        per_angle = min(num_samples, 5)
        angle_list = [(a, b) for a in range(90) for b in range(90) for _ in range(per_angle)]

    else:
        raise ValueError("mode must be 'random' or 'grid'")
    
    os.makedirs(output_path, exist_ok=True)
    metadata_records = []

    for index, (alpha, beta) in enumerate(tqdm(angle_list, desc=f"Generating {output_path}")):
        # a) Draw and warp plate
        pil_img, src_pts, plate_txt, bboxes = create_license_plate(width, height, font_size)
        img_rgb = np.array(pil_img)
        warped_rgb, dst_pts = warp_image(img_rgb, src_pts, alpha, beta, f=width)
        noisy_rgb = simulate_noise(warped_rgb)
        dewarped_rgb = dewarp_image(noisy_rgb, src_pts, dst_pts)

        # b) Crop back to original plate area
        crop_clean = crop_to_original_size(img_rgb, width, height)
        crop_dist = crop_to_original_size(dewarped_rgb, width, height)

        # c) Optional downscale
        if output_size is not None:
            crop_clean, _ = downscale_plate(crop_clean, [], output_size)
            crop_dist, bboxes = downscale_plate(crop_dist, bboxes, output_size)

        # d) Save PNGs
        clean_path = os.path.join(output_path, f"original_{index}.png")
        distorted_path = os.path.join(output_path, f"distorted_{index}.png")
        Image.fromarray(crop_clean).save(clean_path)
        Image.fromarray(crop_dist).save(distorted_path)

        # e) Collect metadata
        metadata_records.append(
            {"index": index, "plate_number": plate_txt, "alpha": alpha, "beta": beta, "digit_bboxes": bboxes}
        )

    # Write one JSON manifest
    manifest_path = os.path.join(output_path, "metadata.json")
    with open(manifest_path, "w") as mf:
        json.dump(metadata_records, mf, indent=1)



# =====================================
# Main Function
# =====================================


def main():
    seed = 42

    base_plate = (128, 32)  # (width, height)
    base_font = 25  # text size at scale=0

    # Scale options: {0: 128×32, 1: 256×64, 2: 512×128, 3: 1024×256}
    in_scale = 2
    out_scale = 1

    width, height = [s * (2**in_scale) for s in base_plate]
    font_size = base_font * (2**in_scale)

    # Only set output_size if you actually want a different scale
    if out_scale != in_scale:
        output_size = tuple(s * (2**out_scale) for s in base_plate)
    else:
        output_size = None

    num_train = 2**13  # 8192
    num_val = 2**10  # 1024
    num_test = 2**10  # 1024
    unique = 32

    generate_dataset(
        output_path="data/unique",
        width=width,
        height=height,
        font_size=font_size,
        seed=seed,
        output_size=output_size,
        mode="random",
        num_samples=unique,
    )

    generate_dataset(
        output_path="data/train",
        width=width,
        height=height,
        font_size=font_size,
        seed=seed + 1,
        output_size=output_size,
        mode="random",
        num_samples=num_train,
    )
    
    generate_dataset(
        output_path="data/val",
        width=width,
        height=height,
        font_size=font_size,
        seed=seed + 2,
        output_size=output_size,
        mode="random",
        num_samples=num_val,
    )

    generate_dataset(
        output_path="data/test",
        width=width,
        height=height,
        font_size=font_size,
        seed=seed + 3,
        output_size=output_size,
        mode="random",
        num_samples=num_test,
    )

    generate_dataset(
        output_path="data/full_grid",
        width=width,
        height=height,
        font_size=font_size,
        seed=seed + 4,
        output_size=output_size,
        mode="grid",
        num_samples=1,
    )


if __name__ == "__main__":
    main()
