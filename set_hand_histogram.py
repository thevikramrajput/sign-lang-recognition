import cv2
import numpy as np
import pickle
import os

def build_squares(img):
    """Draw sampling squares and return cropped color regions."""
    x, y, w, h = 420, 140, 10, 10
    d = 10
    full_crop = None

    for i in range(10):
        row_crop = None
        for j in range(5):
            roi = img[y:y+h, x:x+w]
            row_crop = roi if row_crop is None else np.hstack((row_crop, roi))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        full_crop = row_crop if full_crop is None else np.vstack((full_crop, row_crop))
        x = 420
        y += h + d

    return full_crop


def get_hand_hist():
    """Capture HSV histogram of hand skin using sampling squares."""
    # Try to open a valid camera
    cam = None
    for idx in [1, 0, 2]:
        cam = cv2.VideoCapture(idx)
        ok, _ = cam.read()
        if ok:
            break
        cam.release()
    if not cam or not cam.isOpened():
        print("❌ No working camera found.")
        return

    print("Press 'C' to capture histogram and 'S' to save & exit.")

    hist, imgCrop = None, None
    flagPressedC, flagPressedS = False, False

    while True:
        ok, img = cam.read()
        if not ok:
            continue

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF

        # Capture histogram when 'C' is pressed
        if keypress == ord('c'):
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                flagPressedC = True
                print("✅ Histogram captured. You can press 'S' to save.")
            else:
                print("⚠️ Could not capture ROI. Move your hand inside squares.")

        # Save histogram when 'S' is pressed
        elif keypress == ord('s'):
            flagPressedS = True
            break

        # Display backprojection after capture
        if flagPressedC and hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresholded Backprojection", thresh)

        # Build sampling squares until saved
        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.putText(img, "Press C: Capture  |  S: Save & Exit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Set Hand Histogram", img)

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

    # Save histogram if available
    if hist is not None:
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
        print("💾 Histogram saved successfully as 'hist'")
    else:
        print("⚠️ Histogram not saved (no capture performed).")


if __name__ == "__main__":
    get_hand_hist()
