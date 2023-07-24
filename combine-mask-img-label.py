import cv2
from tqdm import trange
from glob import glob
import os

# Define directories for image and mask files
image_dir = 'nor_img'
mask_dir = 'norm_img_mask'
label_dir = 'nor_label'

# Create the "combo-maskandimage" directory if it doesn't exist
combo_dir = 'combo-maskandimage'
os.makedirs(combo_dir, exist_ok=True)

# Get lists of image and mask files
image_files = sorted(glob(os.path.join(image_dir, '*.png')))
mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
label_files = sorted(glob(os.path.join(label_dir, '*.txt')))

# Loop through each image file and process them
for i in trange(len(image_files), desc="Processing Images"):
    fname = os.path.basename(image_files[i])  # Get the original image name
    img = cv2.imread(image_files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load the corresponding mask
    mask_name = os.path.basename(image_files[i]).replace('.png', '.png')
    mask_file = os.path.join(mask_dir, mask_name)
    mask = cv2.imread(mask_file)

    # Load the corresponding label
    label_name = os.path.basename(image_files[i]).replace('.png', '.txt')
    label_file = os.path.join(label_dir, label_name)
    with open(label_file, 'r') as f:
        label_text = f.read()

    # Overlay the mask on the image
    overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)

    # Add the label text as a visual label on the combo image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White color for the label text
    text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = img.shape[0] - 20
    cv2.putText(overlay, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Save the resulting combo image with the original image name as PNG
    combo_path = os.path.join(combo_dir, f'{fname[:-4]}_combo.png')  # Keeping original image name
    cv2.imwrite(combo_path, overlay)
