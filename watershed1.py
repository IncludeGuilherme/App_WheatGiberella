# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy.ndimage import label
from skimage import measure

class app:
    # constructor class to handle image
    def __init__(self, file):
        self.img = cv2.imread(file)
        self.original = self.img.copy()
        if self.img is None:
            raise Exception(f'Falha ao carregar o arquivo {file} ')
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)    
        _, self.img_bin = cv2.threshold(self.img_gray, 0, 255,
                cv2.THRESH_OTSU)
        self.img_bin = cv2.morphologyEx(self.img_bin, cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=int))

    # watershed method for segmentation

    def watershed(self, a, img):
        self.border = cv2.dilate(img, None, iterations=5)
        self.border = self.border - cv2.erode(self.border, None)
        self.dt = cv2.distanceTransform(img, 2, 3)
        self.dt = ((self.dt - self.dt.min()) / (self.dt.max() - self.dt.min()) * 255).astype(np.uint8)
        _, self.dt = cv2.threshold(self.dt, 10, 255, cv2.THRESH_BINARY)
        self.lbl, self.ncc = label(self.dt)
        self.lbl = self.lbl * (255 / (self.ncc + 1))
        self.lbl[self.border == 255] = 255
        self.lbl = self.lbl.astype(np.int32)
        self.lbl = cv2.watershed(a, self.lbl)
        self.lbl[self.lbl == -1] = 0
        self.lbl = self.lbl.astype(np.uint8)
        self.result =  255 - self.lbl
        self.result[self.result != 255] = 0
        result = cv2.dilate(self.result, None)
        self.img[result == 255] = (255, 0, 0) 
        self.get_roi()

    def get_roi(self):
        markers = self.lbl.astype(np.uint8)
        ret, m2 = cv2.threshold(markers, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print(f'Foram encontrados {len(contours)} grãos')
        num = 1
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            self.roi = self.original[y:y+h, x:x+w]
            cv2.imwrite('grain_{}.png'.format(num), self.roi)
            num += 1
        cv2.imwrite('Grain_Detected.png', self.img) 


    def run(self):
      self.watershed(self.img, self.img_bin)


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'corn.jpeg'
    app(fn).run()
