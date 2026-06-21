import cv2
import numpy as np
from PIL import Image

def grabcut_leaf(image):

    img = np.array(image.convert("RGB"))

    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)

    rect = (
        int(w * 0.05),
        int(h * 0.05),
        int(w * 0.90),
        int(h * 0.90)
    )

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img,
        mask,
        rect,
        bgdModel,
        fgdModel,
        2,
        cv2.GC_INIT_WITH_RECT
    )

    mask2 = np.where(
        (mask == cv2.GC_BGD) |
        (mask == cv2.GC_PR_BGD),
        0,
        1
    ).astype("uint8")

    white_bg = np.ones_like(img) * 255
    white_bg[mask2 == 1] = img[mask2 == 1]

    contours, _ = cv2.findContours(
        mask2,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)

        x, y, w_box, h_box = cv2.boundingRect(largest)

        pad = 20

        x = max(0, x - pad)
        y = max(0, y - pad)

        x2 = min(img.shape[1], x + w_box + pad * 2)
        y2 = min(img.shape[0], y + h_box + pad * 2)

        cropped = white_bg[y:y2, x:x2]

        return Image.fromarray(cropped)

    return Image.fromarray(white_bg)