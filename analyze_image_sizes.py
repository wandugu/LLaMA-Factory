import json
import os
import re
from PIL import Image, UnidentifiedImageError

# Disable PIL DecompressionBombError for large images by setting a higher threshold
# Default is 89 Megapixels (89 * 1024 * 1024 pixels)
Image.MAX_IMAGE_PIXELS = None # Or set to a very large number like 200 * 1024 * 1024

def find_image_paths_in_value(value_str, base_dir):
    """
    Tries to find image paths in a string value.
    Looks for <image> tags or direct paths to .png, .jpg, .jpeg, .webp files.
    """
    paths = []
    # Regex to find file paths with common image extensions
    # It also tries to capture paths that might be within <image> tags or quoted
    potential_paths = re.findall(r'(?:<image>)?([\w\-\_\./\\]+\.(?:png|jpe?g|webp))(?:</image>)?', str(value_str), re.IGNORECASE)
    for p_path in potential_paths:
        # Clean up potential extra characters if inside a tag
        p_path = p_path.strip().replace("<image>", "").replace("</image>", "")
        if os.path.isabs(p_path):
            paths.append(p_path)
        else:
            # Handle cases where the path might be like "images/image.png" or just "image.png"
            # Assume relative to base_dir
            paths.append(os.path.join(base_dir, os.path.basename(p_path))) # Take basename to avoid issues with nested paths in JSON relative to JSON
            # Also consider the path as is, if it's already structured like "subdir/image.png"
            paths.append(os.path.join(base_dir, p_path))


    # Fallback: if the value itself looks like a path
    if isinstance(value_str, str) and any(value_str.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
        if os.path.isabs(value_str):
            paths.append(value_str)
        else:
            paths.append(os.path.join(base_dir, os.path.basename(value_str)))
            paths.append(os.path.join(base_dir, value_str))

    # Deduplicate and verify existence for non-absolute paths before returning
    # For this script, actual existence check happens later, focus on path construction
    unique_paths = []
    for p in paths:
        # Normalize paths to handle mixed slashes and remove duplicates
        normalized_p = os.path.normpath(p)
        if normalized_p not in unique_paths:
            unique_paths.append(normalized_p)
    return unique_paths


def analyze_dataset_images(json_filepath):
    base_image_dir = os.path.dirname(json_filepath)
    print(f"Base directory for relative image paths: {base_image_dir}")

    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return

    total_image_entries = 0
    images_checked_successfully = 0
    oversized_images_count = 0
    oversized_images_details = []

    processed_abs_paths = set() # To avoid checking the same image multiple times if linked differently

    for item_index, item in enumerate(dataset):
        image_paths_in_item = []

        # Standard LLaVA-like format: conversations list
        if isinstance(item, dict) and "conversations" in item and isinstance(item["conversations"], list):
            for turn in item["conversations"]:
                if isinstance(turn, dict):
                    if turn.get("from", "").lower() == "user" or turn.get("from", "").lower() == "human":
                        value = turn.get("value", "")
                        # Try to extract paths if value contains typical image indicators
                        if "<image>" in str(value) or any(ext in str(value).lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                            image_paths_in_item.extend(find_image_paths_in_value(value, base_image_dir))
                    # Check for direct image keys in turn (less common for LLaMA Factory SFT)
                    if "image" in turn: # direct path
                        image_paths_in_item.extend(find_image_paths_in_value(turn["image"], base_image_dir))
                    if "image_path" in turn:
                         image_paths_in_item.extend(find_image_paths_in_value(turn["image_path"], base_image_dir))

        # Check for top-level image keys if not in conversations
        elif isinstance(item, dict):
            if "image" in item:
                 image_paths_in_item.extend(find_image_paths_in_value(item["image"], base_image_dir))
            if "image_path" in item:
                 image_paths_in_item.extend(find_image_paths_in_value(item["image_path"], base_image_dir))
            if "images" in item and isinstance(item["images"], list): # common for multi-image
                for img_ref in item["images"]:
                     image_paths_in_item.extend(find_image_paths_in_value(img_ref, base_image_dir))
            # If item itself is a string path (less likely for a list of conversations)
            elif isinstance(item, str) and any(item.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                 image_paths_in_item.extend(find_image_paths_in_value(item, base_image_dir))


        # Deduplicate paths found within the current item before processing
        unique_paths_in_item = []
        for p in image_paths_in_item:
            abs_path = os.path.normpath(p) # Already made absolute or based on base_image_dir
            if abs_path not in unique_paths_in_item:
                unique_paths_in_item.append(abs_path)

        if not unique_paths_in_item and item_index < 5: # Log if no images found in first few items
            # print(f"Debug: No image paths extracted from item {item_index}: {str(item)[:200]}")
            pass


        for raw_img_path in unique_paths_in_item:
            # Path construction logic:
            # 1. If raw_img_path is already absolute, use it.
            # 2. If relative, join with base_image_dir.
            if os.path.isabs(raw_img_path):
                img_abs_path = os.path.normpath(raw_img_path)
            else:
                # This case should ideally be handled by find_image_paths_in_value making them effectively absolute
                # or correctly relative to base_image_dir.
                # We re-join here just in case find_image_paths_in_value returns something like "image.png"
                # and it was meant to be relative to JSON file location
                img_abs_path = os.path.normpath(os.path.join(base_image_dir, raw_img_path))


            if not img_abs_path.lower().endswith(".png"): # Ensure we only process PNGs as per request, though regex allows more
                # print(f"Skipping non-PNG file: {img_abs_path}")
                continue

            if img_abs_path in processed_abs_paths:
                continue
            processed_abs_paths.add(img_abs_path)
            total_image_entries += 1

            try:
                if not os.path.exists(img_abs_path):
                    # Try one more common pattern: path is relative to a subdirectory like 'images' within base_image_dir
                    potential_path_in_images_subdir = os.path.normpath(os.path.join(base_image_dir, "images", os.path.basename(raw_img_path)))
                    if os.path.exists(potential_path_in_images_subdir):
                        img_abs_path = potential_path_in_images_subdir
                    else:
                        # Try relative to base_image_dir/png/ (if paths in json are like 0.png)
                        potential_path_in_png_subdir = os.path.normpath(os.path.join(base_image_dir, "png", os.path.basename(raw_img_path)))
                        if os.path.exists(potential_path_in_png_subdir):
                            img_abs_path = potential_path_in_png_subdir
                        else:
                           print(f"Warning: Image file not found at {img_abs_path} (derived from {raw_img_path}). Skipping.")
                           continue

                with Image.open(img_abs_path) as img:
                    width, height = img.size
                    images_checked_successfully += 1
                    pixels = width * height

                    is_oversized = False
                    reason = []
                    if width > 4096:
                        is_oversized = True
                        reason.append(f"Width > 4096 ({width}px)")
                    if height > 4096:
                        is_oversized = True
                        reason.append(f"Height > 4096 ({height}px)")
                    if pixels > 16000000: # Approx 4k x 4k
                        is_oversized = True
                        # reason.append(f"Total pixels > 16M ({pixels/1_000_000:.1f}M)") # Avoid duplicate for very large
                    if pixels > 80000000: # Close to PIL default limit
                        is_oversized = True
                        reason.append(f"Total pixels > 80M ({pixels/1_000_000:.1f}M)")

                    if is_oversized:
                        oversized_images_count += 1
                        if len(oversized_images_details) < 20:
                            oversized_images_details.append({
                                "path": img_abs_path,
                                "width": width,
                                "height": height,
                                "pixels": pixels,
                                "reason": ", ".join(list(dict.fromkeys(reason))) # Unique reasons
                            })

            except FileNotFoundError:
                # This specific check is mostly handled above, but as a fallback
                print(f"Warning: Image file not found at {img_abs_path} (derived from {raw_img_path}). Skipping.")
            except UnidentifiedImageError:
                print(f"Warning: Cannot identify image file (possibly corrupted or not an image): {img_abs_path}. Skipping.")
            except Image.DecompressionBombWarning: # Should be disabled by MAX_IMAGE_PIXELS=None
                print(f"Warning: Image is too large and triggered DecompressionBombWarning (PIL limit): {img_abs_path}. Skipping.")
            except Exception as e:
                print(f"Error processing image {img_abs_path}: {e}. Skipping.")

    print("\n--- Image Analysis Summary ---")
    print(f"Total image entries found in JSON: {total_image_entries}")
    print(f"Images checked successfully: {images_checked_successfully}")
    print(f"Number of images exceeding size thresholds: {oversized_images_count}")

    if oversized_images_details:
        print("\nUp to 20 oversized images:")
        for img_info in oversized_images_details:
            print(f"  Path: {img_info['path']}, Dimensions: {img_info['width']}x{img_info['height']}, "
                  f"Pixels: {img_info['pixels']/1_000_000:.1f}M, Reason: {img_info['reason']}")
    elif oversized_images_count > 0:
        print("Details for oversized images were collected but not printed as the list was too long (or an error occurred).")

if __name__ == "__main__":
    # dataset_filepath = "/ossfs/workspace/llama-factory-copy/data/OmniDoc_full_train_png.json"
    # For local testing, you might use a placeholder:
    # dataset_filepath = input("Enter the full path to the JSON dataset file: ")
    dataset_filepath = "/ossfs/workspace/llama-factory-copy/data/OmniDoc_full_train_png.json" # Hardcoded for this task
    if not os.path.exists(dataset_filepath):
        print(f"Provided dataset path does not exist: {dataset_filepath}")
    else:
        analyze_dataset_images(dataset_filepath)

# Example of how the JSON structure might look (for robust parsing):
# [
#   { // Item 1: LLaVA-like SFT
#     "id": "item1",
#     "conversations": [
#       {"from": "human", "value": "Describe this image <image>"},
#       {"from": "gpt", "value": "It shows a cat.", "image_path": "path/to/cat.png"} // image_path here is unusual for LLaMA-Factory SFT but good to check
#     ],
#     "image": "another/path/to/image.png" // Also possible top-level image
#   },
#   { // Item 2: Simpler structure, maybe for pretraining or custom
#     "id": "item2",
#     "text": "User asked about <image> from images/example.png",
#     "image_path": "images/example.png"
#   },
#   { // Item 3: List of images
#       "id": "item3",
#       "conversations": [{"from": "human", "value": "Look at these <image> and <image>"}],
#       "images": ["img1.png", "subdir/img2.png"]
#   }
# ]
# The script will try to infer paths like "path/to/cat.png", "another/path/to/image.png", "images/example.png", "img1.png", "subdir/img2.png"
# and make them absolute relative to the JSON file's directory.
# It will also try to extract from "Describe this image <image>" if the <image> tag is followed by a path or if the path is the value itself.
# The current find_image_paths_in_value focuses on extracting paths from the value string.
# The main loop then iterates over common top-level keys like "image", "image_path", "images".
# The base path for relative images is critical: `/ossfs/workspace/llama-factory-copy/data/`
# So if json has "images/foo.png", it becomes `/ossfs/workspace/llama-factory-copy/data/images/foo.png`
# If json has just "foo.png", it becomes `/ossfs/workspace/llama-factory-copy/data/foo.png` (if it's directly in data/)
# or `/ossfs/workspace/llama-factory-copy/data/png/foo.png` or `/ossfs/workspace/llama-factory-copy/data/images/foo.png` (if in subdirs)
