import cv2
import numpy as np
from utils.arabic_utils import put_arabic_text, measure_arabic_text

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False


def draw_corner_box(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, corner_length_ratio=0.2):
    w = x2 - x1
    h = y2 - y1
    l = int(min(w, h) * corner_length_ratio)
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)
    return img


def draw_label_box(img, label_en, label_ar, x1, y1, x2, box_color, font_size=20):
    """
    Draws a clean label above a bounding box.

    Layout (top → bottom):
    ┌─────────────────────────────┐
    │  سيارة              ← Arabic (right-aligned, yellow, PIL)
    │  car 0.91           ← English (left-aligned, white, cv2)
    └─────────────────────────────┘

    x1,y1 = top-left of bounding box
    x2    = right edge of bounding box (used for Arabic right-align)
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    padding   = 5
    ar_h      = font_size + 6
    (en_w, en_h), _ = cv2.getTextSize(label_en, font, 0.6, 2)
    en_h_pad  = en_h + padding

    total_h   = ar_h + en_h_pad + padding * 2
    box_w     = max(en_w + padding * 2, 100, x2 - x1)

    bg_y1 = max(0, y1 - total_h)
    bg_y2 = y1

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, bg_y1), (x1 + box_w, bg_y2), box_color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # English label — left-aligned, bottom of box
    en_y = bg_y2 - padding
    cv2.putText(img, label_en, (x1 + padding, en_y),
                font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Arabic label — RIGHT-aligned, top of box
    ar_y = bg_y1 + padding + font_size
    right_x = x1 + box_w - padding
    put_arabic_text(img, label_ar,
                    position=(right_x, ar_y),
                    font_size=font_size,
                    color=(255, 255, 0),
                    align="right")

    return img