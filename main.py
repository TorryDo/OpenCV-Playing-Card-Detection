import time

import cv2
import numpy as np

from processing import process
from utils import display
from utils.Loader import Loader

frameWidth = 640
frameHeight = 480

debug = True

# url = 'https://192.168.1.173:8080'

# change to 1 if using USB webcam
cap = cv2.VideoCapture("test/2cards.mp4")
# cap = cv2.VideoCapture(url + "/video")
frame_rate = 30

cap.set(3, frameWidth)  # width is id number 3
cap.set(4, frameHeight)  # height is id 4
cap.set(10, 150)  # change brightness to 150

flatten_card_set = []

prev = 0

train_ranks = Loader.load_ranks('imgs/ranks')
train_suits = Loader.load_suits('imgs/suits')

black_img = np.zeros((300, 200))

while True:
    time_elapsed = time.time() - prev

    success, img = cap.read()

    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        imgResult = img.copy()
        imgResult2 = img.copy()

        thresh = process.get_thresh(img)
        four_corners_set = process.find_corners_set(thresh, imgResult, draw=True)
        flatten_card_set = process.find_flatten_cards(imgResult2, four_corners_set)
        cropped_images = process.get_corner_snip(flatten_card_set)

        if debug:
            if len(flatten_card_set) <= 0:
                cv2.imshow('flat', black_img)

            for flat in flatten_card_set:
                cv2.imshow('flat', flat)

        ranksuit_list: list = list()

        if debug and len(cropped_images) <= 0:
            cv2.imshow("crop", black_img)
            cv2.imshow("rank-suit", black_img)

        for i, (img, original) in enumerate(cropped_images):

            if debug:
                hori = np.concatenate((img, original), axis=1)
                cv2.imshow("crop", hori)

            drawable = img.copy()
            original_copy = original.copy()

            ranksuit = process.split_rank_suit(drawable, original_copy, debug=debug)

            ranksuit_list.append(ranksuit)

        try:
            for rank, suit in ranksuit_list:
                rank = cv2.resize(rank, (70, 100), 0, 0)
                suit = cv2.resize(suit, (70, 100), 0, 0)
                if debug:
                    h = np.concatenate((rank, suit), axis=1)
                    cv2.imshow("rank-suit", h)
        except:
            cv2.imshow("rank-suit", black_img)

        rs = list[str]()

        for _rank, _suit in ranksuit_list:
            predict_rank, predict_suit = process.template_matching(_rank, _suit, train_ranks, train_suits)
            prediction = f"{predict_rank} {predict_suit}"
            rs.append(prediction)
            print(prediction)

        process.show_text(
            predictions=rs,
            four_corners_set=four_corners_set,
            img=imgResult
        )

        # show the overall image
        time.sleep(0.05)
        cv2.imshow('Result', display.stack_images(0.55, [imgResult, thresh]))

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
