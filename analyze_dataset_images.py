import json
import os
from PIL import Image
import argparse
import re

# Suppress DecompressionBombWarning temporarily for this script,
# as its purpose is to identify large images.
# The user is already aware of this warning from their training logs.
Image.MAX_IMAGE_PIXELS = None

def find_image_paths_in_value(value, base_dir):
    """
    Tries to find image paths in a given string value.
    An image path can be explicitly listed or be within an <image> tag.
    """
    paths = []
    if isinstance(value, str):
        # Check for <image>path/to/image.png</image>
        matches = re.findall(r"<image>(.*?)</image>", value)
        for match in matches:
            paths.append(os.path.join(base_dir, match.strip()))

        # Check if the whole string is a path (after stripping potential non-path characters)
        # This is a heuristic. If it's a sentence with a path, <image> tag is better.
        potential_path = value.strip()
        if any(potential_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']):
            # Further check if it looks like a path (e.g., no spaces in the middle)
            if ' ' not in potential_path.split('/')[-1]: # crude check for filename without spaces
                 paths.append(os.path.join(base_dir, potential_path))
    elif isinstance(value, list): # If value is a list of paths
        for item in value:
            if isinstance(item, str) and any(item.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']):
                paths.append(os.path.join(base_dir, item.strip()))
    return list(set(paths)) # Return unique paths

def analyze_images(dataset_filepath, base_image_dir):
    """
    Analyzes images in a dataset JSON file.
    Dataset is expected to be a list of conversations (ShareGPT-like format)
    or a list of items with 'image' or 'image_path' keys.
    """
    if not os.path.exists(dataset_filepath):
        print(f"Error: Dataset file not found at {dataset_filepath}")
        return

    if base_image_dir is None:
        base_image_dir = os.path.dirname(dataset_filepath)
    print(f"Using base directory for relative image paths: {base_image_dir}")

    try:
        with open(dataset_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {dataset_filepath}: {e}")
        return
    except Exception as e:
        print(f"Error reading dataset file {dataset_filepath}: {e}")
        return

    if not isinstance(data, list):
        print("Error: Dataset JSON is not a list. Please adapt the script if your structure is different.")
        return

    image_paths_to_check = []
    processed_items = 0

    for item_index, item in enumerate(data):
        processed_items += 1
        current_item_images = []
        if isinstance(item, dict):
            # Option 1: ShareGPT-like structure (list of conversations)
            if "conversations" in item and isinstance(item["conversations"], list):
                for turn in item["conversations"]:
                    if isinstance(turn, dict) and "value" in turn:
                        current_item_images.extend(find_image_paths_in_value(turn["value"], base_image_dir))
            # Option 2: Direct image path keys
            elif "image" in item: # LLaVA-like direct key
                current_item_images.extend(find_image_paths_in_value(item["image"], base_image_dir))
            elif "image_path" in item:
                current_item_images.extend(find_image_paths_in_value(item["image_path"], base_image_dir))
            elif "image_paths" in item and isinstance(item["image_paths"], list): # list of paths
                 for img_path_val in item["image_paths"]:
                    current_item_images.extend(find_image_paths_in_value(img_path_val, base_image_dir))
            # Option 3: Image path in a general 'value' or 'text' field if not conversations
            elif "value" in item:
                 current_item_images.extend(find_image_paths_in_value(item["value"], base_image_dir))
            elif "text" in item:
                 current_item_images.extend(find_image_paths_in_value(item["text"], base_image_dir))

        image_paths_to_check.extend(list(set(current_item_images)))


    image_paths_to_check = list(set(image_paths_to_check)) # Consolidate all unique paths
    print(f"Found {len(image_paths_to_check)} unique image path references to check from {processed_items} items.")

    checked_images_count = 0
    oversized_images = []
    not_found_count = 0
    corrupted_count = 0

    # Thresholds
    DIM_THRESHOLD = 4096
    TOTAL_PIXELS_THRESHOLD_WARN = 16 * 1024 * 1024  # 16 Megapixels
    TOTAL_PIXELS_THRESHOLD_HIGH = 80 * 1024 * 1024  # 80 Megapixels (near PIL default)


    for img_path in image_paths_to_check:
        if not os.path.isabs(img_path): # Redundant if os.path.join was used correctly, but good check
            actual_img_path = os.path.join(base_image_dir, img_path)
        else:
            actual_img_path = img_path

        actual_img_path = os.path.normpath(actual_img_path)


        if not os.path.exists(actual_img_path):
            # Try stripping <image> and </image> again if path has them literally
            # This case should ideally be handled by find_image_paths_in_value
            stripped_path_for_check = re.sub(r"</?image>", "", actual_img_path)
            if not os.path.exists(stripped_path_for_check) :
                print(f"Warning: Image file not found at {actual_img_path} (and {stripped_path_for_check})")
                not_found_count += 1
                continue
            else:
                actual_img_path = stripped_path_for_check


        try:
            with Image.open(actual_img_path) as img:
                width, height = img.size
                total_pixels = width * height
                checked_images_count += 1

                is_oversized = False
                if width > DIM_THRESHOLD or height > DIM_THRESHOLD:
                    is_oversized = True
                if total_pixels > TOTAL_PIXELS_THRESHOLD_WARN:
                    is_oversized = True

                if is_oversized:
                    oversized_images.append({
                        "path": actual_img_path,
                        "width": width,
                        "height": height,
                        "total_pixels": total_pixels
                    })

                if total_pixels > TOTAL_PIXELS_THRESHOLD_HIGH:
                    print(f"HIGHLY OVERSIZED: {actual_img_path}, Dimensions: {width}x{height}, Total Pixels: {total_pixels}")


        except FileNotFoundError: # Should be caught by os.path.exists, but as fallback
            print(f"Warning: Image file not found at {actual_img_path} (during open)")
            not_found_count += 1
        except Exception as e:
            print(f"Warning: Could not open or read image {actual_img_path}. Error: {e}")
            corrupted_count += 1

    print(f"\n--- Analysis Summary ---")
    print(f"Total unique image path references found in JSON: {len(image_paths_to_check)}")
    print(f"Images successfully checked: {checked_images_count}")
    print(f"Images not found: {not_found_count}")
    print(f"Images corrupted or unreadable: {corrupted_count}")
    print(f"Number of images exceeding thresholds (dim > {DIM_THRESHOLD} or pixels > {TOTAL_PIXELS_THRESHOLD_WARN/(1024*1024):.0f}MP): {len(oversized_images)}")

    if oversized_images:
        print(f"\n--- List of up to 20 Oversized Images ---")
        # Sort by total pixels descending to show largest first
        oversized_images.sort(key=lambda x: x["total_pixels"], reverse=True)
        for i, img_info in enumerate(oversized_images[:20]):
            print(f"Path: {img_info['path']}, Dimensions: {img_info['width']}x{img_info['height']}, Total Pixels: {img_info['total_pixels']:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze image sizes in a JSON dataset.")
    parser.add_argument("dataset_filepath", help="Path to the JSON dataset file.")
    parser.add_argument("--base_image_dir", help="Base directory for relative image paths. Defaults to the directory of the dataset file.", default=None)

    args = parser.parse_args()
    analyze_images(args.dataset_filepath, args.base_image_dir)
