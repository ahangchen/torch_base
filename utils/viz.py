import cv2
import numpy as np


def label2rgb(label_np):
    label_color = np.argmax(label_np, axis=0)
    label_color = label_color / np.max(label_color) * 255
    label_color = cv2.applyColorMap(label_color.astype(np.uint8), 'jet')
    return label_color