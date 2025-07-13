import os
from shutil import copyfile

def duplicate_images(input_path, output_dir, count):
    """
    Duplicate an image multiple times.

    :param input_path: Path to the input image.
    :param output_dir: Directory to save the duplicated images.
    :param count: Number of duplicates to create.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]

    for i in range(count):
        output_path = os.path.join(output_dir, f"{base_name}_{i+1}{ext}")
        copyfile(input_path, output_path)

# Input images and output directory
input_images = [
    "in_img/person/입력이미지1.png",
    "in_img/person/입력이미지2.png",
    "in_img/person/입력이미지3.png"
]
output_base_dir = "in_img/person/duplicated/"

# Number of duplicates
duplicate_count = 250

# Process each image
for input_image in input_images:
    output_dir = os.path.join(output_base_dir, os.path.splitext(os.path.basename(input_image))[0])
    duplicate_images(input_image, output_dir, duplicate_count)

print("Images duplicated successfully.")