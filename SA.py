import collections
import math
import os
import cv2
import numpy as np
import time

MAX_LINES = 3000
N_PINS = 300
MIN_LOOP = 20               # To avoid getting stuck in a loop
MIN_DISTANCE = 20           # To avoid very short lines
LINE_WEIGHT = 15            # Tweakable parameter
FILENAME = "mario 2.webp"
SCALE = 25                  # For making a very high resolution render, to attempt to accurately gauge how thick the thread must be
HOOP_DIAMETER = 0.625       # To calculate total thread length

tic = time.perf_counter()

# Intenta cargar la imagen y verifica si la carga fue exitosa
img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {FILENAME}")

# Calcular el lado más corto de la imagen no cuadrada
shortest_side = min(img.shape[0], img.shape[1])

# Utilizar el lado más corto como longitud de los lados del cuadrado recortado
start_row = (img.shape[0] - shortest_side) // 2
end_row = start_row + shortest_side
start_col = (img.shape[1] - shortest_side) // 2
end_col = start_col + shortest_side
sub_img = np.empty((shortest_side, shortest_side), dtype=np.uint8)

# Redimensionar la imagen original para que coincida con las dimensiones de la región de sub_img
resized_img = cv2.resize(img, (end_col - start_col, end_row - start_row))

# Asignar la imagen redimensionada a la región de sub_img
sub_img[:,:] = resized_img

# Calcular las coordenadas del centro del cuadrado recortado
center_x = shortest_side / 2
center_y = shortest_side / 2

# Calcular el radio del círculo
radius = shortest_side / 2 - 1/2

# Cortar todo alrededor de un círculo central
X, Y = np.ogrid[0:shortest_side, 0:shortest_side]
circlemask = (X - center_x) ** 2 + (Y - center_y) ** 2 > radius * radius
sub_img[circlemask] = 0xFF

pin_coords = np.empty((N_PINS, 2), dtype=int)
center = shortest_side / 2
radius = shortest_side / 2 - 1/2

# Precalculate the coordinates of every pin
angles = np.linspace(0, 2*math.pi, N_PINS, endpoint=False)
pin_coords[:, 0] = np.floor(center + radius * np.cos(angles))
pin_coords[:, 1] = np.floor(center + radius * np.sin(angles))

line_cache_y = np.empty((N_PINS, N_PINS), dtype=object)
line_cache_x = np.empty((N_PINS, N_PINS), dtype=object)
line_cache_length = np.zeros((N_PINS, N_PINS), dtype=int)

print("Precalculating all lines... ", end='', flush=True)

for a in range(N_PINS):
    for b in range(a + MIN_DISTANCE, N_PINS):
        x0 = pin_coords[a][0]
        y0 = pin_coords[a][1]

        x1 = pin_coords[b][0]
        y1 = pin_coords[b][1]

        d = int(math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0)*(y1 - y0)))

        xs = np.linspace(x0, x1, d, dtype=int)
        ys = np.linspace(y0, y1, d, dtype=int)

        line_cache_y[a, b] = ys
        line_cache_y[b, a] = ys
        line_cache_x[a, b] = xs
        line_cache_x[b, a] = xs
        line_cache_length[a, b] = d
        line_cache_length[b, a] = d

print("done")

error = np.ones(sub_img.shape, dtype=int) * 0xFF - sub_img.copy()

result = np.ones((sub_img.shape[0] * SCALE, sub_img.shape[1] * SCALE), np.uint8) * 0xFF

line_sequence = []
pin = 0
line_sequence.append(pin)

thread_length = 0

last_pins = collections.deque(maxlen=MIN_LOOP)

for l in range(MAX_LINES):
    if l % 100 == 0:
        print("%d " % l, end='', flush=True)

        img_result = cv2.resize(result, sub_img.shape, interpolation=cv2.INTER_AREA)

        diff = img_result - sub_img
        mul = np.uint8(img_result < sub_img) * 254 + 1
        absdiff = diff * mul
        print(absdiff.sum() / (shortest_side * shortest_side))

    max_err = -math.inf
    best_pin = -1

    for offset in range(MIN_DISTANCE, N_PINS - MIN_DISTANCE):
        test_pin = (pin + offset) % N_PINS
        if test_pin in last_pins:
            continue

        xs = line_cache_x[test_pin, pin]
        ys = line_cache_y[test_pin, pin]

        line_err = np.sum(error[ys, xs]) * LINE_WEIGHT

        if line_err > max_err:
            max_err = line_err
            best_pin = test_pin

    line_sequence.append(best_pin)

    xs = line_cache_x[best_pin, pin]
    ys = line_cache_y[best_pin, pin]

    line_mask = np.zeros(sub_img.shape, np.float64)
    line_mask[ys, xs] = LINE_WEIGHT
    error = error - line_mask
    error.clip(0, 255)

    cv2.line(result,
             (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
             (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE),
             color=0, thickness=4, lineType=8)

    x0 = pin_coords[pin][0]
    y0 = pin_coords[pin][1]
    x1 = pin_coords[best_pin][0]
    y1 = pin_coords[best_pin][1]

    dist = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0)*(y1 - y0))
    thread_length += HOOP_DIAMETER / shortest_side * dist

    last_pins.append(best_pin)
    pin = best_pin

img_result = cv2.resize(result, sub_img.shape, interpolation=cv2.INTER_AREA)

diff = img_result - sub_img
mul = np.uint8(img_result < sub_img) * 254 + 1
absdiff = diff * mul

print(absdiff.sum() / (shortest_side * shortest_side))

print('\x07')
toc = time.perf_counter()
print("%.1f seconds" % (toc - tic))

# Guardar la imagen de resultado en formato .jpg
cv2.imwrite(os.path.splitext(FILENAME)[0] + "-out.jpg", result)

with open(os.path.splitext(FILENAME)[0] + ".json", "w") as f:
    f.write(str(line_sequence))
