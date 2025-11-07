import cv2
import numpy as np
import pickle, os, sqlite3, random

# Target image size (must match training model)
IMAGE_X, IMAGE_Y = 50, 50

# ----------------- DATABASE & FOLDER SETUP -----------------
def get_hand_hist():
    if not os.path.exists("hist"):
        print("❌ 'hist' file not found. Please run set_hand_histogram.py first.")
        exit(1)
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    """Create the gestures folder and database if not exist."""
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        conn.execute("""
            CREATE TABLE gesture (
                g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                g_name TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Created gesture_db.db")

def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)

def store_in_db(g_id, g_name):
    """Store gesture ID and name in SQLite database."""
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input(f"⚠️ g_id {g_id} already exists. Update name? (y/n): ")
        if choice.lower() == "y":
            cursor.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (g_name, g_id))
            print("✅ Updated record.")
        else:
            print("⏩ Skipped database update.")
            conn.close()
            return
    conn.commit()
    conn.close()

# ----------------- GESTURE IMAGE CAPTURE -----------------
def store_images(g_id):
    total_pics = 1200
    hist = get_hand_hist()

    # open camera
    cam = None
    for i in [1, 0, 2]:
        cam = cv2.VideoCapture(i)
        if cam.read()[0]:
            break
        cam.release()
    if not cam or not cam.isOpened():
        print("❌ No working camera found.")
        return

    x, y, w, h = 300, 100, 300, 300
    create_folder(f"gestures/{g_id}")

    pic_no, frames = 0, 0
    flag_start_capturing = False

    print("📸 Press 'C' to toggle capturing ON/OFF. Press 'Q' to quit early.")

    while True:
        ret, img = cam.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Backprojection using saved histogram
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)

        # Smooth and threshold
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        roi_thresh = thresh_gray[y:y+h, x:x+w]

        contours, _ = cv2.findContours(roi_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Minimum contour area threshold (adaptive)
        min_area = int(0.05 * (w * h))

        if contours and flag_start_capturing:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > min_area and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = roi_thresh[y1:y1+h1, x1:x1+w1]

                # Padding to make square
                if w1 > h1:
                    pad = (w1 - h1) // 2
                    save_img = cv2.copyMakeBorder(save_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    pad = (h1 - w1) // 2
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))

                save_img = cv2.resize(save_img, (IMAGE_X, IMAGE_Y))

                # Random horizontal flip for augmentation
                if random.randint(0, 1):
                    save_img = cv2.flip(save_img, 1)

                pic_no += 1
                cv2.imwrite(f"gestures/{g_id}/{pic_no}.jpg", save_img)
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 255, 255), 2)

        # Draw ROI & info
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"Images: {pic_no}/{total_pics}", (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Capturing gesture", img)
        cv2.imshow("Threshold ROI", roi_thresh)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0
            print("🟢 Capturing started." if flag_start_capturing else "🔴 Capturing paused.")
        elif keypress == ord('q'):
            print("🛑 Quit early by user.")
            break

        if flag_start_capturing:
            frames += 1

        if pic_no >= total_pics:
            print("✅ Finished capturing all gesture images.")
            break

    cam.release()
    cv2.destroyAllWindows()

# ----------------- MAIN PROGRAM -----------------
if __name__ == "__main__":
    init_create_folder_database()
    g_id = input("Enter gesture number (e.g., 1): ")
    g_name = input("Enter gesture name/text: ").strip()
    store_in_db(g_id, g_name)
    store_images(g_id)
