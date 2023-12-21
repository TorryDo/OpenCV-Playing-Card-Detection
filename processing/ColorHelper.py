import cv2


class ColorHelper:

    @staticmethod
    def gray2bin(img):
        return cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    @staticmethod
    def reverse(img):
        return cv2.bitwise_not(img)
