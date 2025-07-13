import os
from shutil import copyfile

def distribute_images(input_paths, output_dir, total_count):
    """
    Distribute images to create a total number of duplicates.

    :param input_paths: List of input image paths.
    :param output_dir: Directory to save the duplicated images.
    :param total_count: Total number of duplicates to create.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_images = len(input_paths)
    base_count = total_count // num_images
    extra_count = total_count % num_images

    count = 0
    for i, input_path in enumerate(input_paths):
        duplicates = base_count + (1 if i < extra_count else 0)
        for j in range(duplicates):
            count += 1
            output_path = os.path.join(output_dir, f"a ({count}).png")
            copyfile(input_path, output_path)

# Input images and output directory
input_images = [
    "in_img/person/입력이미지1.png",
    "in_img/person/입력이미지2.png",
    "in_img/person/입력이미지3.png"
]
output_dir = "in_img/person/duplicated/"

# Total number of duplicates
total_duplicate_count = 250

# Distribute images
distribute_images(input_images, output_dir, total_duplicate_count)

print("Images distributed successfully with new naming format.")