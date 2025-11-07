import cv2, os, random
import numpy as np
from glob import glob

# Resolve paths relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def first_existing(*paths):
	for p in paths:
		if os.path.exists(p):
			return p
	return paths[-1]

GESTURES_DIR = first_existing(
	os.path.join(ROOT_DIR, 'gestures'),  # prefer project root dataset
	os.path.join(BASE_DIR, 'gestures'),  # then local Code/gestures
	'gestures'
)

def get_image_size():
	# Prefer inferring from any available sample; else default to 50x50
	candidates = glob(os.path.join(GESTURES_DIR, '*', '*.jpg')) if os.path.isdir(GESTURES_DIR) else []
	if candidates:
		img = cv2.imread(candidates[0], 0)
		if img is not None:
			return img.shape  # (height, width)
	return (50, 50)

# Collect gesture folder names as strings so we can format like original
if not os.path.isdir(GESTURES_DIR):
	print(f"gestures folder not found at: {GESTURES_DIR}")
	raise SystemExit(1)

gestures = [d for d in os.listdir(GESTURES_DIR) if os.path.isdir(os.path.join(GESTURES_DIR, d)) and d.isdigit()]
gestures.sort(key=int)

if not gestures:
	print("No gesture folders found. Create some with create_gestures.py first.")
	raise SystemExit(1)

per_row = 5
image_h, image_w = get_image_size()
# label band height under each tile
label_h = max(18, min(32, image_h // 4))
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = max(0.4, min(0.6, image_h / 120.0))
font_thickness = 1

def make_tile(img_gray, label_text: str):
	# Ensure image is the expected size
	if img_gray is None or img_gray.shape != (image_h, image_w):
		if img_gray is None:
			img_gray = np.zeros((image_h, image_w), dtype=np.uint8)
		else:
			img_gray = cv2.resize(img_gray, (image_w, image_h))
	tile = np.zeros((image_h + label_h, image_w), dtype=np.uint8)
	tile[0:image_h, 0:image_w] = img_gray
	if label_text:
		# center text within the label band
		text_size, baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
		text_w, text_h = text_size
		x = max(2, (image_w - text_w) // 2)
		y = image_h + (label_h + text_h) // 2
		cv2.putText(tile, label_text, (x, y - baseline), font, font_scale, 255, font_thickness, cv2.LINE_AA)
	return tile

rows = (len(gestures) + per_row - 1) // per_row

full_img = None
for i in range(rows):
	row_img = None
	start = i * per_row
	end = min(start + per_row, len(gestures))
	row_classes = gestures[start:end]

	# Build the row by selecting a random image from each class folder
	for cls in row_classes:
		cls_dir = os.path.join(GESTURES_DIR, cls)
		files = glob(os.path.join(cls_dir, '*.jpg'))
		img = None
		if files:
			img_path = random.choice(files)
			img = cv2.imread(img_path, 0)
		tile = make_tile(img, cls)
		row_img = tile if row_img is None else np.hstack((row_img, tile))

	# Pad to full width (exact 5 columns) with blanks if needed
	while (end - start) < per_row:
		pad = make_tile(None, "")
		row_img = pad if row_img is None else np.hstack((row_img, pad))
		end += 1

	full_img = row_img if full_img is None else np.vstack((full_img, row_img))

# Show exactly at the array's native size (no scaling)
cv2.imshow("gestures", full_img)
cv2.imwrite(os.path.join(BASE_DIR, 'full_img.jpg'), full_img)
cv2.waitKey(0)
