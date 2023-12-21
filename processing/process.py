import cv2
import matplotlib.pyplot as plt
import numpy as np

from processing.ColorHelper import ColorHelper
from utils import constants
from utils.MathHelper import MathHelper


def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bin = ColorHelper.gray2bin(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((2, 2))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)

    return dial


def find_corners_set(img, original, draw=False):
    # find the set of contours on the threshed image
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # sort them by highest area
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []

    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        # only select those contours with a good area
        if area > 10000:
            # find out the number of corners
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            num_corners = len(approx)

            if num_corners == 4:
                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)

                if draw:
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # make sure the image is oriented right: top left, bot left, bot right, top right
                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                finalOrder = []

                # sort by X vlaue
                sortedX = sorted([l1, l2, l3, l4], key=lambda x: x[0][0])

                # sortedX[0] and sortedX[1] are the left half
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))

                four_corners_set.append(finalOrder)

                if draw:
                    for a in approx:
                        cv2.circle(original, (a[0][0], a[0][1]), 6, (255, 0, 0), 3)

    return four_corners_set


def find_flatten_cards(img, set_of_corners, debug=False):
    width, height = 200, 300
    img_outputs = []

    for i, corners in enumerate(set_of_corners):
        top_left = corners[0][0]
        bottom_left = corners[1][0]
        bottom_right = corners[2][0]
        top_right = corners[3][0]

        vertical_left = MathHelper.length(top_left[0], top_left[1], bottom_left[0], bottom_left[1])
        horizontal_top = MathHelper.length(top_left[0], top_left[1], top_right[0], top_right[1])

        # get the 4 corners of the card
        pts1 = np.float32([top_left, bottom_left, bottom_right, top_right])
        # now define which corner we are referring to
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        img_output = cv2.warpPerspective(img, matrix, (width, height))

        img_outputs.append(img_output)

    return img_outputs


def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        # crop the image to where the corner might be
        # vertical, horizontal
        crop = img[5:110, 1:38]

        # resize by a factor of 4
        crop = cv2.resize(crop, None, fx=4, fy=4)

        # threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bin_img = ColorHelper.gray2bin(gray)
        bilateral = cv2.bilateralFilter(bin_img, 11, 174, 17)
        canny = cv2.Canny(bilateral, 40, 24)
        kernel = np.ones((1, 1))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        # append the thresholded image and the original one
        corner_images.append([result, bin_img])

    return corner_images


def split_rank_suit(img, original, debug=False) -> list:
    """
    :param debug: display opencv or not
    :param img:
    :param original: original image
    :return: list of image, index 0: rank, index 1: suit
    """

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])

    cv2.drawContours(img, cnts_sort, -1, (0, 255, 0), 1)

    ranksuit = list()

    _rank = None

    for i, cnt in enumerate(cnts_sort):

        x, y, w, h = cv2.boundingRect(cnt)
        x2, y2 = x + w, y + h

        crop = original[y:y2, x:x2]

        if i == 0:  # rank: 70, 125
            crop = cv2.resize(crop, (70, 125), 0, 0)
            _rank = crop
        else:  # suit: 70, 100
            crop = cv2.resize(crop, (70, 100), 0, 0)
            if debug and _rank is not None:
                r = cv2.resize(_rank, (70, 100), 0, 0)
                s = cv2.resize(crop, (70, 100), 0, 0)
                h = np.concatenate((r, s), axis=1)
                h = cv2.resize(h, (250, 200), 0, 0)
                cv2.imshow("crop2", h)

        crop = ColorHelper.gray2bin(crop)
        crop = ColorHelper.reverse(crop)
        ranksuit.append(crop)

    return ranksuit


def template_matching(rank, suit, train_ranks, train_suits, show_plt=False) -> tuple[str, str]:
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    for train_rank in train_ranks:

        diff_img = cv2.absdiff(rank, train_rank.img)

        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_name = train_rank.name

            if show_plt:
                print(f'diff score: {rank_diff}')
                plt.subplot(1, 2, 1)
                plt.imshow(diff_img, 'gray')

        plt.show()

    # Same processing with suit images
    for train_suit in train_suits:

        diff_img = cv2.absdiff(suit, train_suit.img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < best_suit_match_diff:
            best_suit_match_diff = suit_diff
            best_suit_name = train_suit.name

            if show_plt:
                print(f'diff score: {suit_diff}')
                plt.subplot(1, 2, 2)
                plt.imshow(diff_img, 'gray')

        plt.show()

    if best_rank_match_diff < 2300:
        best_rank_match_name = best_rank_name

    if best_suit_match_diff < 1000:
        best_suit_match_name = best_suit_name

    plt.show()

    return best_rank_match_name, best_suit_match_name


def show_text(predictions: list[str], four_corners_set, img):
    for i, prediction in enumerate(predictions):
        # figure out where to place the text
        corners = np.array(four_corners_set[i])
        corners_flat = corners.reshape(-1, corners.shape[-1])
        start_x = corners_flat[0][0] + 0
        half_y = corners_flat[0][1] - 40

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, prediction, (start_x, half_y), font, 0.8, (50, 205, 50), 2, cv2.LINE_AA)


# region trash

# def split_rank_and_suit(cropped_images):
#     rank_suit_mapping = []
#
#     for img, original in cropped_images:
#
#         # find the contours (we want the rank and suit contours)
#         contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#         # find the largest two contours
#         highest_two = dict()
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             # if area < 2000, its not of relevance to us, so just fill it with black
#             if area < 2000:
#                 cv2.fillPoly(img, pts=[cnt], color=0)
#                 continue
#
#             perimeter = cv2.arcLength(cnt, closed=True)
#             # append the contour and the perimeter
#             highest_two[area] = [cnt, perimeter]
#
#         # select the largest two in this image
#         mapping = []
#
#         for area in sorted(highest_two)[0:2]:
#             cnt = highest_two[area][0]
#             perimeter = highest_two[area][1]
#             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, closed=True)
#             x, y, w, h = cv2.boundingRect(approx)
#             crop = original[y:y + h][:]
#
#             sharpened = Augment.contrast(crop, 30)
#
#             for i in range(sharpened.shape[0]):
#                 for j in range(sharpened.shape[1]):
#                     if sharpened[i, j] < 220:
#                         sharpened[i, j] = max(0, sharpened[i, j] - 100)
#                     if sharpened[i, j] > 221:
#                         sharpened[i, j] = 255
#
#             mapping.append([sharpened, y])
#
#         # store rank and then suit
#         mapping.sort(key=lambda x: x[1])
#
#         for m in mapping:
#             del m[1]
#
#         # # now we don't need the last item so we can remove
#         if mapping and len(mapping) == 2:
#             rank_suit_mapping.append([mapping[0][0], mapping[1][0]])
#
#     return rank_suit_mapping

def eval_rank_suite(rank_suit_mapping, modelRanks, modelSuits):
    pred = []

    for rank, suit in rank_suit_mapping:
        # resize the rank and suit to our desired size
        rank = cv2.resize(rank, (constants.CARD_WIDTH, constants.CARD_HEIGHT))
        suit = cv2.resize(suit, (constants.CARD_WIDTH, constants.CARD_HEIGHT))

        # get the predictions for suit and rank
        bestSuitPredictions = model_wrapper.model_predict(modelSuits, suit, 'suits')  # min(suitDict, key=suitDict.get)
        bestRankPredictions = model_wrapper.model_predict(modelRanks, rank, 'ranks')  # min(rankDict, key=rankDict.get)

        # get the names and percentage of best and second best suits and ranks
        bestSuitName, bestSuitPer = model_wrapper.model_predictions_to_name(bestSuitPredictions)
        bestRankName, bestRankPer = model_wrapper.model_predictions_to_name(bestRankPredictions)
        sbestSuitName, sbestSuitPer = model_wrapper.model_predictions_to_name(bestSuitPredictions, loc=-2)
        sbestRankName, sbestRankPer = model_wrapper.model_predictions_to_name(bestRankPredictions, loc=-2)

        # show both guesses
        totalPer = bestRankPer + sbestRankPer + bestSuitPer + sbestSuitPer
        guess1 = '{}/{}/{}%'.format(bestRankName, bestSuitName, round(((bestSuitPer + bestRankPer) / totalPer) * 100))
        guess2 = '{}/{}/{}%'.format(sbestRankName, sbestSuitName,
                                    round(((sbestSuitPer + sbestRankPer) / totalPer) * 100))

        pred.append('{}\n{}'.format(guess1, guess2))

    return pred

# endregion
