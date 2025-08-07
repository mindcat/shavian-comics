# Core libraries
import os
import subprocess
import io
from pathlib import Path
import shutil # For clearing temp directory if needed
import math

# Image processing
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np

# Object detection/segmentation
from ultralytics import YOLO # Preferred for running .pt files

# OCR
import pytesseract # Requires tesseract executable install
import easyocr
from autocorrect import Speller
import nltk
nltk.download('words')
word_list = set(nltk.corpus.words.words())
# nltk.download('words')
from symspellpy import SymSpell, Verbosity
# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# Define path to your downloaded dictionary
dictionary_path = "dict/frequency_dictionary_en_82_765.txt"
# Load the dictionary
try:
    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        print(f"Error: Dictionary file not found or failed to load from {dictionary_path}")
        # Handle error appropriately - maybe raise an exception or set sym_spell to None
        raise FileNotFoundError(f"SymSpell dictionary not found at {dictionary_path}")
    print("SymSpell dictionary loaded successfully.")
except Exception as e:
    print(f"An unexpected error occurred loading the SymSpell dictionary: {e}")
    raise e

# Optional: Display images in notebook
# %matplotlib inline
# import matplotlib.pyplot as plt

# --- Directories ---
INPUT_DIR = Path("img/in")
TEMP_DIR = Path("img/tmp") # For intermediate bubble images (optional use)
OUTPUT_DIR = Path("img/out")
MODEL_DIR = Path("models")
FONT_DIR = Path("fonts") # Directory containing the .otf font file
DICT_DIR = Path("dict") # Directory containing the dictionary for shaw.py

# --- Model ---
# Using the .pt file is generally easier with the ultralytics library
MODEL_NAME = "model.pt" # Or .onnx if using ONNX runtime
MODEL_PATH = MODEL_DIR / MODEL_NAME # Adjust subdirs as needed

# --- OCR ---
# Set this if tesseract is not in your system PATH
# TESSERACT_CMD = r'/path/to/your/tesseract.exe' # Example for Windows
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- Transliteration Script ---
SHAW_SCRIPT_PATH = "shaw.py" # Assuming it's in the same dir as the notebook or PATH
SHAW_DICT_PATH = DICT_DIR / "amer.dict"

# --- Font ---
# SHAVIAN_FONT_NAME = "backlogshavian.otf"
SHAVIAN_FONT_NAME = "stampatello-faceto.otf"
# SHAVIAN_FONT_NAME = "bantershaviantriline.otf"
SHAVIAN_FONT_PATH = FONT_DIR / SHAVIAN_FONT_NAME
DEFAULT_FONT_SIZE = 10 # Starting font size (might need adjustment)

# --- Other Settings ---
# Confidence threshold for bubble detection (adjust as needed)
CONFIDENCE_THRESHOLD = 0.5
# Padding around detected bounding boxes when extracting/drawing
PADDING = 5
# Background color to fill bubbles before drawing Shavian text (RGBA)
# Use white (255, 255, 255) with full opacity (255)
FILL_COLOR = (255, 255, 255, 255)
# Set to True to clear the temp directory each run
CLEAR_TEMP_DIR = False


def segment_speech_bubbles(image_path, model, expected_count):
    """
    Detects speech bubbles in an image using the loaded YOLO model.

    Args:
        image_path (Path or str): Path to the input image.

    Returns:
        List[Dict]: A list of dictionaries, where each dict contains:
                    'mask': A PIL Image (mode 'L') representing the bubble mask.
                    'bbox': A tuple (x1, y1, x2, y2) for the bounding box.
                    'confidence': Detection confidence score.
    """
    results = model(str(image_path), conf=CONFIDENCE_THRESHOLD)
    detected_bubbles = []

    # Results is a list (usually length 1 for single image)
    if not results or not results[0].masks:
        print(f"No masks found in results for {image_path.name}")
        return []



    # Process masks from the first result object
    res = results[0]
    img_shape = res.orig_shape # (height, width)
    print(f"Processing masks for {image_path.name}. Original shape: {img_shape}")

    if res.masks:
        masks_data = res.masks.data.cpu().numpy() # Get masks as numpy arrays on CPU
        bboxes_data = res.boxes.xyxy.cpu().numpy() # Get bounding boxes
        confs_data = res.boxes.conf.cpu().numpy() # Get confidences

        print(f"Found {len(masks_data)} potential masks.")

        for i, mask_np in enumerate(masks_data):
            if confs_data[i] < CONFIDENCE_THRESHOLD:
                continue # Skip low confidence detections

            # Ensure mask is binary (0 or 255) and correct size
            # Ultralytics masks are often normalized or directly binary
            # Ensure it's uint8
            mask_np = (mask_np > 0.5).astype(np.uint8) * 255 # Binarize

            # Convert numpy mask to PIL Image (mode 'L' for grayscale mask)
            mask_pil = Image.fromarray(mask_np, mode='L')

            # Resize mask to original image size if necessary (ultralytics usually returns full-size masks)
            if mask_pil.size != (img_shape[1], img_shape[0]): # size is (width, height)
                 print(f"Resizing mask from {mask_pil.size} to {(img_shape[1], img_shape[0])}")
                 mask_pil = mask_pil.resize((img_shape[1], img_shape[0]), Image.NEAREST)


            # Get bounding box directly from PIL mask for consistency
            bbox = mask_pil.getbbox()
            if not bbox: # Handle empty masks if they occur
                print(f"Warning: Empty mask found for detection {i}.")
                continue

            # Use the bounding box from the model if preferred, but ensure it's int
            # bbox_model = tuple(map(int, bboxes_data[i]))

            detected_bubbles.append({
                'mask': mask_pil,
                'bbox': bbox, # (x1, y1, x2, y2)
                'confidence': confs_data[i]
            })
            print(f"  Added bubble {i+1}: bbox={bbox}, conf={confs_data[i]:.2f}")

    else:
        print(f"No masks attribute found in results for {image_path.name}")


    return detected_bubbles

def extract_bubble_region(original_image, bubble_info):
    """
    Extracts the content of a speech bubble using its mask.
    Creates a white background image cropped to the bubble's bbox,
    and pastes the original content onto it using the mask.

    Args:
        original_image (PIL.Image): The full original comic image.
        bubble_info (dict): Dictionary containing 'mask' and 'bbox'.

    Returns:
        PIL.Image: An image containing only the speech bubble content
                   on a white background, cropped to the bbox.
                   Returns None if bbox is invalid.
    """
    mask = bubble_info['mask']
    bbox = bubble_info['bbox'] # (x1, y1, x2, y2)

    if not bbox or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        print(f"Warning: Invalid bounding box {bbox}. Skipping extraction.")
        return None

    # Crop the mask and the original image to the bounding box
    cropped_mask = mask.crop(bbox)
    cropped_original = original_image.crop(bbox).convert('RGBA')

    # Create a white background matching the cropped size
    bg_color = (255, 255, 255, 0) # RGB white
    # If original has alpha, use RGBA white
    # if original_image.mode == 'RGBA':
    #      bg_color = (255, 255, 255, 0)
    clear_bg = Image.new(original_image.mode.replace('P', 'RGB'), cropped_mask.size, bg_color)


    # Paste the original content onto the white background using the cropped mask
    # Ensure mask is mode 'L' for pasting
    if cropped_mask.mode != 'L':
        cropped_mask = cropped_mask.convert('L')

    clear_bg.paste(cropped_original, (0, 0), cropped_mask)

    upscaled = clear_bg.resize((clear_bg.width * 2, clear_bg.height * 2), Image.Resampling.LANCZOS)
    final = upscaled.filter(ImageFilter.SHARPEN)

    return final

def perform_ocr(bubble_image, sym_spell_instance):
    """
    Performs OCR on an image, likely containing speech bubble text.

    Args:
        bubble_image (PIL.Image): Image of the isolated speech bubble.

    Returns:
        str: The extracted text, with leading/trailing whitespace stripped.
             Returns an empty string if OCR fails or finds no text.
    """
    if bubble_image is None:
        return ""
    try:
        # Use Page Segmentation Mode (PSM) 6: Assume a single uniform block of text.
        # Adjust PSM if needed (e.g., 3 for fully automatic, 11 for sparse text)
        # custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.!\'\"\ '
        # # custom_config = r'--oem 3 --psm 6'
        # text = pytesseract.image_to_string(bubble_image, lang='eng', config=custom_config)
        reader =  easyocr.Reader(['en'])
        bubble_image_np = np.array(bubble_image)
        allowlist_chars = "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?'"
        results = reader.readtext(bubble_image_np, allowlist=allowlist_chars, detail=0, paragraph=True)
        text = ' '.join(results).lower()

        # text = ' '.join(text.split()).lower()
        print(f"  OCR Result: '{text}'")
        spell = Speller(only_replacements=True)
        corrected_text = spell(text)
        print(f"  Autocorrect: '{corrected_text}'")


        # corrected_text = corrected_text.split()
        # valid_words = [word for word in corrected_text if word in word_list]
        # corrected_text = ' '.join(valid_words)
        # print(f"  Nonsense dropped: '{corrected_text}'")

        if sym_spell_instance:
            # Use lookup_compound for potentially better handling of combined words
            # max_edit_distance=2 allows for corrections within 2 edits (adjust as needed)
            suggestions = sym_spell_instance.lookup_compound(corrected_text, max_edit_distance=2)

            # Get the top suggestion
            if suggestions:
                corrected_text = suggestions[0].term
                print(f"  SymSpell Corrected: '{corrected_text}'")
                return corrected_text
            else:
                print("  SymSpell: No correction suggestion found.")
                return corrected_text # Return raw text if no suggestion
        else:
            # If sym_spell object wasn't passed or failed to initialize
            print("  SymSpell: Corrector not available, returning raw OCR text.")
            return corrected_text


        return corrected_text
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract executable not found. Please install Tesseract or set TESSERACT_CMD.")
        raise # Stop execution if Tesseract isn't configured
    except Exception as e:
        print(f"  Error during OCR: {e}")
        return ""

def transliterate_to_shavian(text):
    """
    Transliterates English text to Shavian using the external shaw.py script.

    Args:
        text (str): English text to transliterate.

    Returns:
        str: Transliterated Shavian text, or empty string on error/no input.
    """
    if not text:
        return ""

    command = [
        "python3", # Or just "python" depending on your environment
        SHAW_SCRIPT_PATH,
        str(SHAW_DICT_PATH)
    ]

    try:
        # Use subprocess.run to send text via stdin and capture stdout
        process = subprocess.run(
            command,
            input=text, # Encode input text
            capture_output=True,
            text=True, # Decodes stdout/stderr automatically using default encoding
            check=False # Don't raise exception on non-zero exit code, check manually
        )

        if process.returncode != 0:
            print(f"  Error running shaw.py (return code {process.returncode}):")
            print(f"  Stderr: {process.stderr.strip()}")
            return "" # Indicate failure

        shavian_text = process.stdout.strip()
        print(f"  Shavian Transliteration: '{shavian_text}'")
        return shavian_text

    except FileNotFoundError:
        print(f"Error: The script '{SHAW_SCRIPT_PATH}' or python interpreter was not found.")
        return ""
    except Exception as e:
        print(f"  Error during transliteration: {e}")
        return ""

def draw_shavian_text_in_bubble(draw, bubble_info, shavian_text, font):
    """
    Draws the Shavian text onto the image, wrapped to fit and centered
    within the bubble's bbox, after filling the bubble area.

    Args:
        draw (ImageDraw.Draw): The drawing context for the image.
        bubble_info (dict): Dictionary containing 'mask' and 'bbox'.
        shavian_text (str): The text to draw (already transliterated).
        font (ImageFont.FreeTypeFont): The Shavian font object.
    """
    bbox = bubble_info['bbox'] # (x1, y1, x2, y2)

    if not shavian_text or not bbox:
        print("  Skipping text drawing: No text or bounding box.")
        return

    # Define some padding inside the bounding box
    text_padding = 5 # Pixels from the bbox edge on each side
    bubble_width = bbox[2] - bbox[0]
    bubble_height = bbox[3] - bbox[1]
    max_line_width = bubble_width - 2 * text_padding

    # Check if the bubble is wide enough for text with padding
    if max_line_width <= 0:
        print("  Skipping text drawing: Bubble too narrow for padding.")
        return

    # --- Word Wrapping Logic ---
    words = shavian_text.split()
    if not words:
        print("  Skipping text drawing: Text contains no words.")
        return

    C = 0.4

    smart_font_size = C * math.sqrt((bubble_width * bubble_height) / len(words))
    smart_font_size = max(DEFAULT_FONT_SIZE, smart_font_size)

    font = ImageFont.truetype(font, smart_font_size)

    lines = []
    current_line = ""

    for word in words:
        # Measure the width of the potential line with the new word added
        test_line_with_space = (current_line + " " + word).strip()
        test_line_no_space = (current_line + word).strip() # For first word on line

        # Use the line with space unless current_line is empty
        test_line = test_line_with_space if current_line else test_line_no_space

        line_width = draw.textlength(test_line, font=font)

        if line_width <= max_line_width:
            # Word fits, add it to the current line (use the correctly spaced version)
            current_line = test_line
        else:
            # Word doesn't fit the current line.
            if current_line:
                # Finalize the previous line if it wasn't empty
                lines.append(current_line)
                current_line = "" # Reset for the new line

            # Now, start the new line with the current word.
            # Check if the single word itself is too long for a line.
            word_width = draw.textlength(word, font=font)
            if word_width > max_line_width:
                 print(f"  Warning: Word '{word}' is wider ({word_width:.0f}px) than the available space ({max_line_width:.0f}px). Placing it on its own line.")
                 # Add the oversized word as its own line. It will likely overflow visually.
                 lines.append(word)
                 current_line = "" # Ensure the next word starts a new line
            else:
                 # The word fits on a new line by itself. Start the new line.
                 current_line = word

    # Add the last remaining line after the loop finishes
    if current_line:
        lines.append(current_line)

    # Check if any lines were actually formed
    if not lines:
        print("  Skipping text drawing: No lines could be formed after wrapping.")
        return

    # Join the lines with newline characters for drawing
    wrapped_text = '\n'.join(lines)

    # --- Calculate Position and Draw ---
    try:
        # Calculate the center coordinates of the original bounding box
        center_x = bbox[0] + bubble_width / 2
        center_y = bbox[1] + bubble_height / 2

        # Draw the entire wrapped text block.
        # Use anchor="mm" to center the text block's midpoint at (center_x, center_y).
        # Use align="center" to center each line relative to the others within the block.
        print(f"  Drawing wrapped text (centered):\n------\n{wrapped_text}\n------")
        draw.text(
            (center_x, center_y), # Anchor point is the center of the bbox
            wrapped_text,
            font=font,
            fill=(0, 0, 0, 255),  # Black text, full opacity
            anchor="mm",          # Anchor the text block's middle point
            align="center"        # Center-align multiple lines within the block
        )

        # --- Optional: Add a check/warning if text block height exceeds bubble height ---
        # This provides feedback if the wrapped text is too tall, even if centered.
        block_bbox = draw.textbbox((center_x, center_y), wrapped_text, font=font, anchor='mm', align='center')
        block_height = block_bbox[3] - block_bbox[1]
        available_height = bubble_height - 2 * text_padding # Consider padding for height too
        if block_height > available_height:
             print(f"  Warning: Wrapped text block height ({block_height:.0f}px) may exceed available bubble height ({available_height:.0f}px).")

    except Exception as e:
        print(f"  Error calculating text size or drawing wrapped text: {e}")


def main():
    # Create output and temp directories if they don't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FONT_DIR.mkdir(parents=True, exist_ok=True) # Ensure font dir exists
    DICT_DIR.mkdir(parents=True, exist_ok=True) # Ensure dict dir exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure model dir exists

    # Clear temp directory if requested
    if CLEAR_TEMP_DIR and TEMP_DIR.exists():
        print(f"Clearing temporary directory: {TEMP_DIR}")
        for item in TEMP_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Check if required files exist
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not SHAW_SCRIPT_PATH: # Basic check, assumes it's runnable
        print(f"Warning: Ensure '{SHAW_SCRIPT_PATH}' is accessible.")
    if not SHAW_DICT_PATH.exists():
        raise FileNotFoundError(f"Shavian dictionary not found at {SHAW_DICT_PATH}")
    if not SHAVIAN_FONT_PATH.exists():
        raise FileNotFoundError(f"Shavian font not found at {SHAVIAN_FONT_PATH}")

    print("Directories checked/created.")

    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
        # Example: Check model info (optional)
        # model.info()
    except Exception as e:
        print(f"Error loading model: {e}")
        # You might want to stop execution here if the model fails to load
        raise e

    try:
        shavian_font = str(SHAVIAN_FONT_PATH)
        print(f"Loaded font: {SHAVIAN_FONT_PATH} with size {DEFAULT_FONT_SIZE}")
    except IOError:
        print(f"Error: Could not load font at {SHAVIAN_FONT_PATH}. Please check the path and file.")
        raise

    # Find image files (png, jpg, jpeg, webp)
    image_files = list(INPUT_DIR.glob('*.png')) + \
                list(INPUT_DIR.glob('*.jpg')) + \
                list(INPUT_DIR.glob('*.jpeg')) + \
                list(INPUT_DIR.glob('*.webp'))

    print(f"Found {len(image_files)} images in {INPUT_DIR}")

    # Process each image
    for image_path in image_files:
        print(f"\n--- Processing: {image_path.name} ---")

        try:
            # Load the original image
            original_image = Image.open(image_path).convert("RGBA") # Convert to RGBA for alpha compositing

            # --- Step 1: Detect Speech Bubbles ---
            detected_bubbles = segment_speech_bubbles(image_path, model)

            if not detected_bubbles:
                print("  No speech bubbles detected. Skipping.")
                # Optionally copy the original image to output
                # shutil.copy(image_path, OUTPUT_DIR / image_path.name)
                continue

            # Create a copy for drawing (output image)
            output_image = original_image.copy()
            draw = ImageDraw.Draw(output_image) # Create drawing context ONCE for the output image

            # --- Process Each Bubble ---
            for i, bubble_info in enumerate(detected_bubbles):
                print(f"\nProcessing Bubble {i+1} (bbox: {bubble_info['bbox']})")

                # --- Step 2: Extract Bubble Region (Optional - primarily for OCR) ---
                # You could OCR directly on a masked version of the original,
                # but extracting first might improve OCR accuracy by isolating content.
                bubble_image_for_ocr = extract_bubble_region(original_image, bubble_info)
                if bubble_image_for_ocr:
                    # Optional: Save intermediate bubble image to temp dir
                    temp_filename = TEMP_DIR / f"{image_path.stem}_bubble_{i+1}.png"
                    bubble_image_for_ocr.save(temp_filename)
                    pass # Keep it in memory
                else:
                    print("  Skipping bubble due to extraction error.")
                    continue

                # --- Step 3: Perform OCR ---
                original_text = perform_ocr(bubble_image_for_ocr, sym_spell)

                if not original_text:
                    print("  No text found by OCR. Skipping transliteration for this bubble.")
                    continue

                # --- Step 4: Transliterate to Shavian ---
                shavian_text = transliterate_to_shavian(original_text)

                if not shavian_text:
                    print("  Transliteration failed or returned empty. Skipping drawing for this bubble.")
                    continue

                # --- Step 5: Draw Shavian Text ---
                try:
                    # Ensure mask is 'L' mode (single channel) for numpy indexing
                    fill_mask = bubble_info['mask'].convert('L')
                    mask_np = np.array(fill_mask)
                    # Original image should already be RGBA from loading
                    original_np = np.array(original_image)

                    # Select pixels from the original RGBA image where the mask is white (>128)
                    pixels_in_bubble = original_np[mask_np > 128]

                    if pixels_in_bubble.size > 0:
                        # Import Counter if not already done at the top of the script
                        from collections import Counter
                        # Pixels are likely RGBA tuples. Map numpy rows to tuples to make them hashable for Counter.
                        color_counts = Counter(map(tuple, pixels_in_bubble))
                        # Find the most common color tuple
                        calculated_fill_color, _ = color_counts.most_common(1)[0]
                        # Ensure the color has 4 components (RGBA) if needed, add alpha if missing (unlikely if original is RGBA)
                        if len(calculated_fill_color) == 3:
                            calculated_fill_color = (*calculated_fill_color, 255) # Add full opacity alpha
                        print(f"  Calculated mode fill color: {calculated_fill_color}")
                    else:
                        # Fallback if no pixels are found under the mask (e.g., empty mask)
                        print("  Warning: No pixels found under mask. Defaulting fill to white.")
                        # Use the original FILL_COLOR or a default white
                        calculated_fill_color = (255, 255, 255, 255)

                except Exception as e:
                    # Fallback in case of any error during color calculation
                    print(f"  Error calculating mode color: {e}. Defaulting fill to white.")
                    calculated_fill_color = (255, 255, 255, 255) # Use the original FILL_COLOR or a default white

                # --- Fill the bubble area on the output image using the calculated mode color ---
                # Create a solid fill color image matching the output image's mode and size
                # Use the dynamically determined 'calculated_fill_color'
                fill_img = Image.new(output_image.mode, output_image.size, calculated_fill_color)
                # Paste the solid fill color onto the output image, but only where the mask is white
                # The fill_mask (single channel 'L' mode) correctly masks the paste operation
                output_image.paste(fill_img, (0, 0), fill_mask)

                # Now draw the text on top of the filled area
                # Re-create draw object if needed after paste? Usually not necessary.
                draw_shavian_text_in_bubble(draw, bubble_info, shavian_text, shavian_font)


            # --- Step 6: Save Output Image ---
            output_filename = OUTPUT_DIR / f"{image_path.stem}_shavian.png" # Save as PNG to keep transparency
            output_image.save(output_filename)
            print(f"\nSaved modified image to: {output_filename}")

            # Optional: Display results in notebook
            # plt.figure(figsize=(10, 10))
            # plt.imshow(output_image)
            # plt.title(f"Output: {output_filename.name}")
            # plt.axis('off')
            # plt.show()


        except FileNotFoundError:
            print(f"Error: Input image not found at {image_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred processing {image_path.name}: {e}")
            # Consider adding more specific error handling (e.g., for PIL errors, OCR errors)

    print("\n--- Pipeline Finished ---")

    # try:
    #     output_files = list(OUTPUT_DIR.glob('*_shavian.png'))
    #     if output_files:
    #         print(f"Displaying sample output: {output_files[0].name}")
    #         display_img = Image.open(output_files[0])
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(display_img)
    #         plt.axis('off')
    #         plt.show()
    #     else:
    #         print("No output files found to display.")
    # except Exception as e:
    #     print(f"Could not display output image: {e}")



if __name__ == "__main__":
    main()
