import cv2
import numpy as np


Full = []
for i in range(0, 10, 3):
    img1 = cv2.imread(f'ridge_{i}.png')
    if i != 9:
        img2 = cv2.imread(f'ridge_{i + 1}.png')
        img3 = cv2.imread(f'ridge_{i + 2}.png')
    else:
        img2 = np.ones_like(img1) * 255
        img3 = np.ones_like(img1) * 255

    Img = np.concatenate([img1, img2, img3], axis=1)
    Full.append(Img)

Full = np.concatenate(Full, axis=0)
cv2.imwrite("Ridge.png", Full)
