"""import cv2

import numpy as np

a = np.load("/datasets/carracing/12332333312.npz")["obs"]

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('output.avi', fourcc, 50.0, (84, 96))

for i in range(a.shape[0]):
    out.write(a[i])

out.release()"""


import numpy as np
import os

dataset_path = 'datasets/carracing'
thread_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if 'thread' in d and os.path.isdir(os.path.join(dataset_path, d))]

# Check the contents of the first file in the first thread directory
file_path = None
for dir in thread_dirs:
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npz')]
    if files:
        file_path = files[0]
        break

if file_path:
    data = np.load(file_path)
    print(f"Contents of {file_path}: {data.files}")
else:
    print("No .npz files found in the dataset directories.")

