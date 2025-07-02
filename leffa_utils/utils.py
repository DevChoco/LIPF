import os
import cv2
import torch
import numpy as np
from numpy.linalg import lstsq
from PIL import Image, ImageDraw


def resize_and_center(image, target_width, target_height):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    original_height, original_width = img.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    resized_img = cv2.resize(img, (new_width, new_height),
                             interpolation=cv2.INTER_CUBIC)

    padded_img = np.ones((target_height, target_width, 3),
                         dtype=np.uint8) * 255

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    padded_img[top:top + new_height, left:left + new_width] = resized_img

    return Image.fromarray(padded_img)


def list_dir(folder_path):
    # Collect all file paths within the directory
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_paths = sorted(file_paths)
    return file_paths


label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
    "neck": 18,
}


def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist


def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1,
                 mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask


def get_agnostic_mask_hd(model_parse, keypoint, category, size=(384, 512), model_type="hd"):
    width, height = size
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'short_sleeve':
        parse_mask = (parse_array == label_map["upper_clothes"]).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["pants"]).astype(np.float32) + \
                             (parse_array == label_map["skirt"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'shorts':
        parse_mask = (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    
    # MODIFIED: Include short_sleeve in condition
    if category in ['dresses', 'upper_body', 'short_sleeve']:
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, 
                     shoulder_left[1] - ARM_LINE_WIDTH // 2, 
                     shoulder_left[0] + ARM_LINE_WIDTH // 2, 
                     shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, 
                      shoulder_right[1] - ARM_LINE_WIDTH // 2, 
                      shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]

        # MODIFIED: Short sleeve handling
        if category == 'short_sleeve':
            # Draw only shoulder to elbow
            if elbow_right[0] > 1. or elbow_right[1] > 1.:
                arms_draw_right.line([shoulder_right, elbow_right], 'white', ARM_LINE_WIDTH, 'curve')
                arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)
            if elbow_left[0] > 1. or elbow_left[1] > 1.:
                arms_draw_left.line([shoulder_left, elbow_left], 'white', ARM_LINE_WIDTH, 'curve')
                arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)
            # Skip hand masking
        else:
            # Original arm drawing
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                im_arms_right = arms_right
            else:
                wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
                arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 
                                     'white', ARM_LINE_WIDTH, 'curve')
                arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

            if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                im_arms_left = arms_left
            else:
                wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
                arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 
                                    'white', ARM_LINE_WIDTH, 'curve')
                arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

            hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
            hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
            parser_mask_fixed += hands_left + hands_right

    # MODIFIED: Shorts handling (thighs only)
    elif category == 'shorts':
        hip_right = np.multiply(tuple(pose_data[8][:2]), height / 512.0)
        knee_right = np.multiply(tuple(pose_data[9][:2]), height / 512.0)
        hip_left = np.multiply(tuple(pose_data[11][:2]), height / 512.0)
        knee_left = np.multiply(tuple(pose_data[12][:2]), height / 512.0)
        LEG_LINE_WIDTH = int(40 / 512 * height)

        im_legs_left = Image.new('L', (width, height))
        im_legs_right = Image.new('L', (width, height))
        legs_draw_left = ImageDraw.Draw(im_legs_left)
        legs_draw_right = ImageDraw.Draw(im_legs_right)
        
        if knee_left[0] > 1. or knee_left[1] > 1.:
            legs_draw_left.line([hip_left, knee_left], 'white', LEG_LINE_WIDTH, 'curve')
        if knee_right[0] > 1. or knee_right[1] > 1.:
            legs_draw_right.line([hip_right, knee_right], 'white', LEG_LINE_WIDTH, 'curve')

        leg_mask = cv2.dilate(np.logical_or(im_legs_left, im_legs_right).astype('float32'), 
                              np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += leg_mask

    parser_mask_fixed = cv2.erode(parser_mask_fixed, np.ones((5, 5), np.uint16), iterations=1)
    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    
    parse_mask = cv2.dilate(parse_mask, np.ones((10, 10), np.uint16), iterations=5)
    
    # MODIFIED: Include short_sleeve in condition
    if category in ['dresses', 'upper_body', 'short_sleeve']:
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        
        if category == 'short_sleeve':
            arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), 
                                  np.ones((5, 5), np.uint16), iterations=4)
            parse_mask += np.logical_or(parse_mask, arm_mask)
        else:
            arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), 
                                  np.ones((5, 5), np.uint16), iterations=4)
            parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)

    return mask

def get_agnostic_mask_dc(model_parse, keypoint, category, size=(384, 512)):
    parse_array = np.array(model_parse)
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    parse_shape = (parse_array > 0).astype(np.float32)

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32) + \
                 (parse_array == 18).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                        (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["scarf"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                             (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'short_sleeve':
        parse_mask = (parse_array == label_map["upper_clothes"]).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["pants"]).astype(np.float32) + \
                             (parse_array == label_map["skirt"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'shorts':
        parse_mask = (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    parse_head = torch.from_numpy(parse_head)
    parse_mask = torch.from_numpy(parse_mask)
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
    parse_mask = parse_mask.cpu().numpy()

    width = size[0]
    height = size[1]

    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)
    
    # MODIFIED: Include short_sleeve in condition
    if category in ['dresses', 'upper_body', 'short_sleeve']:
        shoulder_right = tuple(np.multiply(pose_data[2, :2], height / 512.0))
        shoulder_left = tuple(np.multiply(pose_data[5, :2], height / 512.0))
        elbow_right = tuple(np.multiply(pose_data[3, :2], height / 512.0))
        elbow_left = tuple(np.multiply(pose_data[6, :2], height / 512.0))
        wrist_right = tuple(np.multiply(pose_data[4, :2], height / 512.0))
        wrist_left = tuple(np.multiply(pose_data[7, :2], height / 512.0))

        # MODIFIED: Short sleeve handling
        if category == 'short_sleeve':
            if elbow_left[0] > 1. or elbow_left[1] > 1.:
                arms_draw.line([shoulder_left, elbow_left], 'white', 30, 'curve')
            if elbow_right[0] > 1. or elbow_right[1] > 1.:
                arms_draw.line([shoulder_right, elbow_right], 'white', 30, 'curve')
            # Skip hand masking
            hands = np.zeros_like(arms)
        else:
            # Original arm drawing
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                    arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right], 'white', 30, 'curve')
                else:
                    arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right], 'white', 30, 'curve')
            elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                    arms_draw.line([shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30, 'curve')
                else:
                    arms_draw.line([elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30, 'curve')
            else:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30, 'curve')
            hands = np.logical_and(np.logical_not(im_arms), arms)

        # Dilation for arms
        if height > 512:
            im_arms = cv2.dilate(np.float32(im_arms), np.ones((10, 10), np.uint16), iterations=5)
        elif height > 256:
            im_arms = cv2.dilate(np.float32(im_arms), np.ones((5, 5), np.uint16), iterations=5)
        
        parse_mask += im_arms
        parser_mask_fixed += hands

    # MODIFIED: Shorts handling (thighs only)
    elif category == 'shorts':
        hip_right = tuple(np.multiply(pose_data[8, :2], height / 512.0))
        knee_right = tuple(np.multiply(pose_data[9, :2], height / 512.0))
        hip_left = tuple(np.multiply(pose_data[11, :2], height / 512.0))
        knee_left = tuple(np.multiply(pose_data[12, :2], height / 512.0))
        LEG_LINE_WIDTH = int(40 / 512 * height)

        im_legs = Image.new('L', (width, height))
        legs_draw = ImageDraw.Draw(im_legs)
        
        if knee_left[0] > 1. or knee_left[1] > 1.:
            legs_draw.line([hip_left, knee_left], 'white', LEG_LINE_WIDTH, 'curve')
        if knee_right[0] > 1. or knee_right[1] > 1.:
            legs_draw.line([hip_right, knee_right], 'white', LEG_LINE_WIDTH, 'curve')

        # Dilation for legs
        if height > 512:
            im_legs = cv2.dilate(np.float32(im_legs), np.ones((10, 10), np.uint16), iterations=5)
        elif height > 256:
            im_legs = cv2.dilate(np.float32(im_legs), np.ones((5, 5), np.uint16), iterations=5)
        parse_mask += im_legs


def preprocess_garment_image(input_path, output_path=None, save_image=False):
    """
    Preprocess a garment image by cropping to a centered square,
    resizing, and pasting it onto a 768x1024 white background.
    """
    img = Image.open(input_path).convert('RGBA')
    
    # Step 1: Get the bounding box of the non-transparent pixels. (the garment)
    alpha = img.split()[-1]
    bbox = alpha.getbbox() # (left, upper, right, lower)
    if bbox is None:
        raise ValueError("No garment found in the image (the image may be fully transparent).")
    
    left, upper, right, lower = bbox
    bbox_width = right - left
    bbox_height = lower - upper
    
    # Step 2: Create a square crop that centers the garment.
    square_size = max(bbox_width, bbox_height)

    center_x = left + bbox_width // 2
    center_y = upper + bbox_height // 2

    new_left = center_x - square_size // 2
    new_upper = center_y - square_size // 2
    new_right = new_left + square_size
    new_lower = new_upper + square_size

    # Adjust the crop if it goes out of the image boundaries.
    if new_left < 0:
        new_left = 0
        new_right = square_size
    if new_upper < 0:
        new_upper = 0
        new_lower = square_size
    if new_right > img.width:
        new_right = img.width
        new_left = img.width - square_size
    if new_lower > img.height:
        new_lower = img.height
        new_upper = img.height - square_size

    # Crop the image to the computed square region.
    square_crop = img.crop((new_left, new_upper, new_right, new_lower))
    
    # Step 3: Resize the square crop.
    # Here we choose 768x768 so that it will occupy the full width when pasted.
    garment_resized = square_crop.resize((768, 768), Image.LANCZOS)
    
    # Step 4: Create a new white background image of 768x1024.
    background = Image.new('RGBA', (768, 1024), (255, 255, 255, 255))
    
    # Compute where to paste the resized garment so that it is centered.
    paste_x = 0
    paste_y = (1024 - 768) // 2
    
    # Paste the garment onto the background.
    background.paste(garment_resized, (paste_x, paste_y), garment_resized)
    
    # Optionally, convert to RGB (if you want to save as JPEG) or keep as PNG.
    final_image = background.convert("RGBA")
    
    if save_image:
        if output_path is None:
            raise ValueError("output_path must be provided if save_image is True.")
        final_image.save(output_path, "PNG")
    
    return final_image