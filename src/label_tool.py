"""
Quick manual labeling tool for die-face crops.

Run this after collecting crops via the live DiceDetector (they land in
config.WATCH_FOLDER). Each image is shown full-size in a window; press a
key to sort it:

    1-6   -> move into DATASET_DIR/<digit>/
    d     -> discard (bad crop, blur, no die visible, etc.) -> DATASET_DIR/discard/
    s     -> skip for now, leave in the incoming folder
    q     -> quit

Labeled crops accumulate in DATASET_DIR/<digit>/ across runs, so you can
label in short batches. This is the dataset the ResNet classifier will
eventually train on.
"""
import os
import shutil
import cv2 as cv  # type: ignore[import-not-found]
from config import WATCH_FOLDER, DATASET_DIR

LABEL_KEYS = {ord(str(i)): str(i) for i in range(1, 7)}
DISCARD_KEY = ord('d')
SKIP_KEY = ord('s')
QUIT_KEY = ord('q')
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

WINDOW_NAME = "Label die face  |  1-6 = label, d = discard, s = skip, q = quit"


def _ensure_dirs():
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    for i in range(1, 7):
        os.makedirs(os.path.join(DATASET_DIR, str(i)), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "discard"), exist_ok=True)


def _dataset_counts():
    counts = {}
    for i in range(1, 7):
        d = os.path.join(DATASET_DIR, str(i))
        counts[str(i)] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return counts


def label_images():
    _ensure_dirs()

    files = sorted(
        f for f in os.listdir(WATCH_FOLDER)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    if not files:
        print(f"No unlabeled images found in '{WATCH_FOLDER}'.")
        return

    print(f"{len(files)} image(s) to label.")
    print("Keys: 1-6 = label, d = discard, s = skip, q = quit\n")

    labeled = 0
    for filename in files:
        src_path = os.path.join(WATCH_FOLDER, filename)
        img = cv.imread(src_path)
        if img is None:
            print(f"[WARN] Could not read {filename}, skipping.")
            continue

        # Upscale small crops so they're actually visible/legible
        h, w = img.shape[:2]
        scale = max(1, 300 // max(1, min(h, w)))
        display = cv.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv.INTER_NEAREST)

        cv.imshow(WINDOW_NAME, display)
        key = cv.waitKey(0) & 0xFF

        if key == QUIT_KEY:
            print("Quitting.")
            break
        elif key == SKIP_KEY:
            continue
        elif key == DISCARD_KEY:
            shutil.move(src_path, os.path.join(DATASET_DIR, "discard", filename))
            print(f"discard  <- {filename}")
        elif key in LABEL_KEYS:
            label = LABEL_KEYS[key]
            shutil.move(src_path, os.path.join(DATASET_DIR, label, filename))
            print(f"{label}        <- {filename}")
            labeled += 1
        else:
            print("Unrecognized key — use 1-6, d, s, or q.")

    cv.destroyAllWindows()

    print(f"\nLabeled {labeled} image(s) this session.")
    print("Dataset totals:", _dataset_counts())


if __name__ == "__main__":
    label_images()