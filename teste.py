# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy.ndimage import label


class app:
    # constructor class to handle image
    def __init__(self, file):
        self.img = cv2.imread(file)
        self.original = self.img.copy()
        if self.img is None:
            raise Exception(f'Falha ao carregar o arquivo {file} ')
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_OTSU)
        self.kernel = np.ones((3, 3), np.uint8)
        self.closing = cv2.morphologyEx(
            self.thresh, cv2.MORPH_OPEN, self.kernel)


    # watershed method for segmentation

    def watershed(self):
        border = cv2.dilate(self.img, None, iterations=5)
        border = border - cv2.erode(border, None)
        dt = cv2.distanceTransform(self.img, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
         # Completing the markers now. 
        lbl[border == 255] = 255
        lbl = lbl.astype(np.int32)
        cv2.watershed(self.original, lbl)

        lbl[lbl == -1] = 0
        lbl = lbl.astype(np.uint8)
        print("Foram encontrados {} grãos".format(len(np.unique(self.markers))-1))
        self.get_roi()
        cv2.imwrite('Image_WatershedSegmentation1.png', lbl)
   
    
    def get_roi(self):
        for marker in np.unique(self.markers):
            if marker <= 1:
                continue
            mask = np.zeros(self.gray.shape, dtype="uint8")
            mask[self.markers == marker] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            self.roi = self.original[y:y+h, x:x+w]
            cv2.imwrite('grain_{}.png'.format(marker - 1), self.roi)
        

    def run(self):
        self.watershed()


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'corn.jpeg'
    app(fn).run()
