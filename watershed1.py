# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


class app:
    # constructor class to handle image
    def __init__(self, file):
        self.img = cv2.imread(file)
        self.original = self.img.copy()
        if self.img is None:
            raise Exception(f'Falha ao carregar o arquivo {file} ')
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.kernel = np.ones((3, 3), np.uint8)
        self.closing = cv2.morphologyEx(
            self.thresh, cv2.MORPH_CLOSE, self.kernel, iterations=10)
        self.sure_bg = cv2.dilate(self.closing, self.kernel, iterations=5)
        self.dist_transform = cv2.distanceTransform(
            self.closing, cv2.DIST_L2, 3)
        self.ret, self.sure_fg = cv2.threshold(
            self.dist_transform, 0.2*self.dist_transform.max(), 255, 0)
        self.sure_fg = np.uint8(self.sure_fg)
        self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)

    # watershed method for segmentation

    def watershed(self):
        self.ret, self.markers = cv2.connectedComponents(self.sure_fg)
        self.markers = self.markers+1
        self.markers[self.unknown == 255] = 0
        self.markers = cv2.watershed(self.img, self.markers)
        self.img[self.markers == -1] = [255, 0, 0]
        print("Foram encontrados {} grãos".format(len(np.unique(self.markers))-1))
        self.get_roi()
        cv2.imwrite('Image_WatershedSegmentation1.png', self.img)
   
    
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
