# ✋ Sign Language Recognition System

A computer vision-based project for **real-time hand gesture recognition** using a **Convolutional Neural Network (CNN)**.
It detects hand gestures through a webcam and converts them into **text or speech**, enabling basic sign language interpretation.

---

## 📁 Project Structure

```
├── cnn_model_train.py        # CNN architecture and model training
├── create_gestures.py        # Capture and store gesture images
├── set_hand_histogram.py     # Create hand histogram for skin detection
├── load_images.py            # Load and pickle gesture image datasets
├── Rotate_images.py          # Data augmentation (horizontal flips)
├── display_gestures.py       # Display collected gesture samples
├── final.py                  # Real-time gesture recognition system
├── Install_Packages.txt      # Required Python packages
├── gestures/                 # Folder containing gesture class images
├── gesture_db.db             # SQLite database mapping IDs to gestures
└── hist                      # Saved histogram for skin detection
```

---

## 🧠 Features

* Real-time **hand gesture detection and classification**
* **CNN-based deep learning model** for gesture recognition
* **Text-to-Speech output** (via `pyttsx3`)
* **Custom dataset creation** using webcam
* **SQLite database** for gesture ID → label mapping
* Modular pipeline: dataset creation → training → recognition

---

## 🧩 Requirements

Make sure Python (≥3.8) is installed.
Install dependencies listed in `Install_Packages.txt`:

```bash
pip install -r Install_Packages.txt
```

Dependencies include:

```
h5py
numpy
scikit-learn
opencv-python
pyttsx3
tensorflow
```

---

## ⚙️ Setup and Usage

### Step 1️⃣ - Create Hand Histogram

Run:

```bash
python set_hand_histogram.py
```

* A webcam window will open.
* Adjust lighting and position your hand in the squares.
* Press **‘c’** to capture histogram.
* Press **‘s’** to save it (creates a file named `hist`).

---

### Step 2️⃣ - Create Gesture Dataset

Run:

```bash
python create_gestures.py
```

* Enter a **gesture ID** and **gesture name**.
* Press **‘c’** to start/stop capturing images.
* 1200 images per gesture are saved in the `gestures/` folder.
* The gesture name is stored in `gesture_db.db`.

---

### Step 3️⃣ - Augment Dataset

To increase dataset size (flip images horizontally):

```bash
python Rotate_images.py
```

---

### Step 4️⃣ - Prepare Data for Training

Run:

```bash
python load_images.py
```

This script:

* Loads all gesture images,
* Splits them into **train**, **validation**, and **test** sets,
* Saves them as pickled files (`train_images`, `val_images`, etc.).

---

### Step 5️⃣ - Train CNN Model

Run:

```bash
python cnn_model_train.py
```

* Trains a CNN on the dataset.
* Saves model as `cnn_model_keras2.h5`.

---

### Step 6️⃣ - Display Collected Gestures (optional)

Run:

```bash
python display_gestures.py
```

Displays all gesture classes in a grid (`full_img.jpg` is saved).

---

### Step 7️⃣ - Run Final Recognition System

Run:

```bash
python final.py
```

* Opens webcam feed.
* Detects gestures in real-time.
* Converts recognized gestures to text and voice.
* Press **‘q’** or **‘c’** to exit.

---

## 🧱️ Model Details

* **Input size:** 50 × 50 grayscale images
* **Architecture:**

  * 3 convolutional layers (16 → 32 → 64 filters)
  * Max pooling after each conv layer
  * Dense layer with 128 neurons + dropout (0.3)
  * Output layer: softmax over gesture classes
* **Optimizer:** SGD (learning rate = 0.001, momentum = 0.9)
* **Loss:** Categorical Cross-Entropy
* **Accuracy:** Prints validation accuracy after training

---

## 🗃️ Database Structure (gesture_db.db)

| Column | Type | Description              |
| ------ | ---- | ------------------------ |
| g_id   | INT  | Gesture ID (primary key) |
| g_name | TEXT | Gesture name/text label  |

---

## 🔊 Voice Output

The recognition system uses `pyttsx3` for offline speech synthesis.
Each recognized gesture is **spoken aloud** once confidence exceeds the threshold (`PRED_THRESHOLD = 0.6`).

---

## 💡 Tips

* Ensure consistent **lighting** when creating and recognizing gestures.
* Keep **ROI (green box)** size consistent across training and recognition.
* To retrain on new gestures, delete the old `cnn_model_keras2.h5` and retrain.
* If predictions are inaccurate, verify histogram and dataset quality.

---

## 📸 Sample Workflow

1. Run `set_hand_histogram.py` → capture hand tone.
2. Run `create_gestures.py` → record gestures (A, B, C...).
3. Run `load_images.py` → prepare datasets.
4. Run `cnn_model_train.py` → train model.
5. Run `final.py` → real-time recognition + speech output.

---

## 👨‍💻 Author

**Vikram Madhad**
B.Tech CSE (AI/ML), Bennett University
Email: [[vikrammadhad@gmail.com](mailto:vikrammadhad@gmail.com)]
Project developed for **Sign Language Recognition using CNN**

---

## 🏆 Acknowledgements

* TensorFlow/Keras for model training
* OpenCV for image capture and preprocessing
* Pyttsx3 for text-to-speech output

---

## 📜 License

This project is released under the **MIT License** - free to use, modify, and distribute.
