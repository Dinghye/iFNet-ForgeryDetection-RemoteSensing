import numpy as np
import cv2

img_pth = r"N:\wangye\my_figure15452.png"

img = cv2.imread(img_pth, cv2.IMREAD_LOAD_GDAL).astype(np.int64)

static = np.bincount(img.flatten())
print(static)
