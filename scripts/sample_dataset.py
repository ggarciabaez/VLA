import cv2
import numpy as np

ds = np.load("../data/dataset_shards/mt10_grip/ep0000.npz")
tasks = np.transpose(ds["images"], (1, 0, 3, 4, 2))
for vid in tasks:
    for frame in vid:
        cv2.imshow("frame", frame)
        cv2.waitKey(16)

"""
from glob import glob
tasks = glob("../data/dataset_shards/mt10_grip/ep0000/*.npz")
print(tasks)
for task in tasks:
    ds = np.load(task)
    vid = np.transpose(ds["images"], (0, 2, 3, 1))
    for frame in vid:
        cv2.imshow("frame", frame)
        cv2.waitKey(16)
"""