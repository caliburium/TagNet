import numpy as np
from PIL import Image


class ToBlackAndWhite(object):
    def __call__(self, img):
        img = img.convert('L')
        img = np.array(img)
        img = (img > 127).astype(np.uint8) * 255
        img = Image.fromarray(img)
        return img
