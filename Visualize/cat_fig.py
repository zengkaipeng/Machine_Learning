import cv2
import numpy as np


Full = []
for i in range(0, 10, 2):
	img1 = cv2.imread(f'logistic_kpart_9_{i}.png')
	img2 = cv2.imread(f'logistic_kpart_9_{i + 1}.png')

	Img = np.concatenate([img1, img2], axis=1)
	Full.append(Img)

Full = np.concatenate(Full, axis=0)
cv2.imwrite("Logistic.png", Full)
