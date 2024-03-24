import os
import argparse
from src.concurrent_task_executor import concurrent_task_executor
from src.utils import crop_and_pad_image, track_files

from typing import NamedTuple, Tuple

parser = argparse.ArgumentParser(description="Crop and Resize Images in a folder")
parser.add_argument("--src", type=str, help="source folder", required=True)
parser.add_argument("--dest", type=str, help="destination folder", required=True)
parser.add_argument("--size", type=Tuple[int,int], help="Size of image in pixels, given as a (width, height) tuple", default=(512, 512))


class FileInfo(NamedTuple):
    src: str
    dest: str
    size: Tuple[int, int]  # (width, height) tuple.


def crop_and_save_image(file_info: FileInfo):
    try:
        cropped_image = crop_and_pad_image(file_info.src, target_size=file_info.size)
        cropped_image.save(file_info.dest)
    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dest
    size = args.size

    # check if destination folder exists
    if not os.path.exists(dst_folder):
        print("Destination folder does not exist. Creating folder...")
        os.makedirs(dst_folder, exist_ok=True)

    files = [
        FileInfo(
            src_image_path,
            os.path.join(dst_folder, os.path.basename(src_image_path)),
            size
        )
        for src_image_path in track_files(src_folder)
    ]
    # cropping and resizing images and saving them to the destination folder
    concurrent_task_executor(crop_and_save_image, files, description="Processing images")
