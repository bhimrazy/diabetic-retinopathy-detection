import os
from datetime import datetime

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from zoneinfo import ZoneInfo


def crop_and_pad_image(image_path, threshold=20, target_size=(512, 512)):
    """
    Crop and pad an image to a square with the specified target size.

    Args:
        image_path (str): Path to the input image file.
        threshold (int): Threshold value for binarizing the image.
        target_size (tuple): Target size of the output image (width, height).

    Returns:
        PIL.Image.Image: Cropped and padded image.
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Binarize the image
    binary_image_array = np.where(image_array > threshold, 1, 0).astype(np.uint8)

    # Find non-zero elements (non-black pixels)
    non_zero_indices = np.argwhere(binary_image_array)

    # Check if non-zero elements exist
    if non_zero_indices.size == 0:
        raise ValueError(f"No non-zero elements found for the image: {image_path}")

    # Get the bounding box of non-zero elements
    (y1, x1, _), (y2, x2, _) = non_zero_indices.min(0), non_zero_indices.max(0)

    # Crop the Region of Interest (ROI)
    cropped_img = image.crop((x1, y1, x2, y2))

    # Pad the image to make it a square
    squared_img = ImageOps.pad(cropped_img, target_size)

    return squared_img


def track_files(folder_path, extensions=(".jpg", ".jpeg", ".png")):
    """
    Track all the files in a folder and its subfolders.

    Args:
        folder_path (str): The path of the folder to track files in.
        extensions (tuple, optional): Tuple of file extensions to track. Default is ('.jpg', '.jpeg', '.png').

    Returns:
        list: A list containing the paths of all files in the folder and its subfolders.
    """
    # Validate folder_path
    if not os.path.isdir(folder_path):
        raise ValueError("Invalid folder path provided.")

    # Convert extensions to lowercase for case-insensitive comparison
    extensions = tuple(ext.lower() for ext in extensions)

    # Initialize file_list
    file_list = []

    # Walk through the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            _, extension = os.path.splitext(file_path)
            # Check if the file extension is in the list of extensions
            if extension.lower() in extensions:
                file_list.append(file_path)

    return file_list


def crop_circle_roi(image_path):
    """
    Crop the circular Region of Interest (ROI) from a fundus image.

    Args:
    - image_path (str): Path to the fundus image.

    Returns:
    - cropped_roi (numpy.ndarray): The cropped circular Region of Interest.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Assuming the largest contour corresponds to the ROI
    contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the circular ROI using the bounding rectangle
    cropped_roi = image[y : y + h, x : x + w]

    return cropped_roi


def plot_image_grid(image_paths, roi_crop=False):
    """
    Create a grid plot with a maximum of 16 images.

    Args:
    - image_paths (list): A list of image paths to be plotted.

    Returns:
    - None
    """
    num_images = min(len(image_paths), 16)
    num_rows = (num_images - 1) // 4 + 1
    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            if roi_crop:
                img = crop_and_pad_image(image_paths[i])
            else:
                img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def generate_run_id(zone: ZoneInfo = ZoneInfo("Asia/Kathmandu")) -> str:
    """Generate a unique run ID using current UTC date and time.

    Args:
        zone (ZoneInfo, optional): Timezone information. Defaults to Indian Standard Time.

    Returns:
        str: A unique run ID in the format 'run-YYYY-MM-DD-HH-MM-SS'.
    """
    try:
        current_utc_time = datetime.utcnow().astimezone(zone)
        formatted_time = current_utc_time.strftime("%Y-%m-%d-%H-%M-%S")
        return f"run-{formatted_time}"
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error generating run ID: {e}")
        return None  # Or raise an exception if appropriate


if __name__ == "__main__":
    print(generate_run_id())
