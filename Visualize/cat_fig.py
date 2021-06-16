import cv2
import numpy as np

for i in range(0, 10, 2):
	img1 = cv2.imread(f'logistic_kpart_9_{i}.png')
	img2 = cv2.imread(f'logistic_kpart_9_{i + 1}.png')

	Img = np.concatenate([img1, img2], axis=0)
	cv2.imshow('test', Img)
	cv2.waitKey(0)
	exit()