import os
import json
import numpy as np

def load_quickdraw_ndjson(data_folder, categories):
    data = []
    labels = []
    for label_index, category in enumerate(categories):
        path = os.path.join(data_folder, category + ".ndjson")
        with open(path, 'r') as f:
            for line in f:
                example = json.loads(line)
                drawing = example['drawing']  # List of strokes
                img = rasterize_drawing(drawing)
                data.append(img)
                labels.append(label_index)
    return np.array(data, dtype=np.float32), np.array(labels)  # CHANGED

def rasterize_drawing(drawing, width=28, height=28):
    canvas = np.zeros((height, width), dtype=np.uint8)
    for stroke in drawing:
        x, y = stroke[:2]
        for i in range(len(x) - 1):
            x0, y0 = int(x[i] * width / 256), int(y[i] * height / 256)
            x1, y1 = int(x[i+1] * width / 256), int(y[i+1] * height / 256)
            canvas = draw_line(canvas, x0, y0, x1, y1)
    return canvas.reshape(-1) / 255.0

def draw_line(img, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]:
            img[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return img
