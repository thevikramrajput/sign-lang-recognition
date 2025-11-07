import os
import warnings
import cv2
import numpy as np
import pickle
import sqlite3
import pyttsx3
from threading import Thread
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def get_model():
    paths = ["cnn_model_keras2.h5", os.path.join(os.getcwd(), "cnn_model_keras2.h5")]
    for p in paths:
        if os.path.exists(p):
            return load_model(p)
    print("Model file not found. Train the model first.")
    return None

def get_hand_hist():
    if not os.path.exists("hist"):
        print("Histogram not found. Run set_hand_histogram.py first.")
        return None
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_image_size():
    return 50, 50

def get_pred_text_from_db(pred_class):
    if not os.path.exists("gesture_db.db"):
        return str(pred_class)
    conn = sqlite3.connect("gesture_db.db")
    try:
        cur = conn.execute("SELECT g_name FROM gesture WHERE g_id=?", (pred_class,))
        row = cur.fetchone()
        return row[0] if row else str(pred_class)
    finally:
        conn.close()

def keras_process_image(img):
    image_x, image_y = get_image_size()
    img = cv2.resize(img, (image_y, image_x))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    if model is None:
        return 0.0, -1
    processed = keras_process_image(image)
    preds = model.predict(processed, verbose=0)[0]
    prediction = np.argmax(preds)
    confidence = preds[prediction]
    return confidence, prediction

def get_img_contour_thresh(img, hist):
    x, y, w, h = 300, 100, 300, 300
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = thresh[y:y+h, x:x+w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return img, contours, thresh

def say_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def recognize():
    model = get_model()
    if model is None:
        return
    hist = get_hand_hist()
    if hist is None:
        return

    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    text, word, count_same_frame = "", "", 0
    while True:
        ret, img = cam.read()
        if not ret:
            break
        img, contours, thresh = get_img_contour_thresh(img, hist)
        old_text = text
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 8000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1+h1, x1:x1+w1]
                if save_img.size != 0:
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)
                    confidence, pred_class = keras_predict(model, save_img)
                    if confidence > 0.6:
                        text = get_pred_text_from_db(pred_class)
                    else:
                        text = ""
                    if old_text == text:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0
                    if count_same_frame > 15 and text != "":
                        Thread(target=say_text, args=(text,)).start()
                        word += text
                        count_same_frame = 0
        else:
            if word != "":
                Thread(target=say_text, args=(word,)).start()
                word = ""

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Predicted: " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Gesture Recognition", res)
        cv2.imshow("Threshold", thresh)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('c'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
