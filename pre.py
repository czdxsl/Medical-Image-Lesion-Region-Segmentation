import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imdecode(np.fromfile('1-2.bmp', dtype=np.uint8), -1)

# dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# 图像去噪
dst2 = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst2)
# plt.subplot(123), plt.imshow(dst2)
plt.show()

