import cv2
import operator
import numpy as np
from keras.models import load_model
import sudo

class result:
    def __init__(self, img = None):
        self.img = img

    def pre_process_image(self, img, skip_dilate=False, flag=0):
        proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
        proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        proc = cv2.bitwise_not(proc, proc)
        if not skip_dilate:
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
            proc = cv2.dilate(proc, kernel)
        return proc


    def find_corners_of_largest_polygon(self, img):
        contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = contours[0]

        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


    def distance_between(self, p1, p2):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))


    def crop_and_warp(self, img, crop_rect, flag=0):
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        side = max([
            self.distance_between(bottom_right, top_right),
            self.distance_between(top_left, bottom_left),
            self.distance_between(bottom_right, bottom_left),
            self.distance_between(top_left, top_right)
        ])
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
        m = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img, m, (int(side), int(side)))
        return warp


    def infer_grid(self, img):
        squares = []
        side = img.shape[0] / 9
        for j in range(9):
            for i in range(9):
                p1 = (i * side, j * side)
                p2 = ((i + 1) * side, (j + 1) * side)
                squares.append((p1, p2))
        return squares


    def cut_from_rect(self, img, rect):
        return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


    def scale_and_centre(self, img, size, margin=0, background=0):
        h, w = img.shape[:2]

        def centre_pad(length):

            if length % 2 == 0:
                side1 = int((size - length) / 2)
                side2 = side1
            else:
                side1 = int((size - length) / 2)
                side2 = side1 + 1
            return side1, side2

        def scale(r, x):
            return int(r * x)

        if h > w:
            t_pad = int(margin / 2)
            b_pad = t_pad
            ratio = (size - margin) / h
            w, h = scale(ratio, w), scale(ratio, h)
            l_pad, r_pad = centre_pad(w)
        else:
            l_pad = int(margin / 2)
            r_pad = l_pad
            ratio = (size - margin) / w
            w, h = scale(ratio, w), scale(ratio, h)
            t_pad, b_pad = centre_pad(h)

        img = cv2.resize(img, (w, h))
        img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
        return cv2.resize(img, (size, size))


    def find_largest_feature(self, inp_img, scan_tl=None, scan_br=None):
        img = inp_img.copy()
        height, width = img.shape[:2]

        max_area = 0
        seed_point = (None, None)

        if scan_tl is None:
            scan_tl = [0, 0]

        if scan_br is None:
            scan_br = [width, height]

        for x in range(scan_tl[0], scan_br[0]):
            for y in range(scan_tl[1], scan_br[1]):
                if img.item(y, x) == 255 and x < width and y < height:
                    area = cv2.floodFill(img, None, (x, y), 64)
                    if area[0] > max_area:
                        max_area = area[0]
                        seed_point = (x, y)

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 255 and x < width and y < height:
                    cv2.floodFill(img, None, (x, y), 64)

        mask = np.zeros((height + 2, width + 2), np.uint8)

        if all([p is not None for p in seed_point]):
            cv2.floodFill(img, mask, seed_point, 255)

        top, bottom, left, right = height, 0, width, 0

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 64:
                    cv2.floodFill(img, mask, (x, y), 0)

                if img.item(y, x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right

        bbox = [[left, top], [right, bottom]]
        return np.array(bbox, dtype='float32')


    def extract_digit(self, img, rect, size):
        digit = self.cut_from_rect(img, rect)
        h, w = digit.shape[:2]
        margin = int(np.mean([h, w]) / 2.5)
        bbox = self.find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
        digit = self.cut_from_rect(digit, bbox)

        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
            return self.scale_and_centre(digit, size, 4)
        else:
            return np.zeros((size, size), np.uint8)


    def get_digits(self, img, squares, size, flag=0):
        digits = []
        img = self.pre_process_image(img.copy(), skip_dilate=True, flag=flag)
        for square in squares:
            digits.append(self.extract_digit(img, square, size))
        return digits


    def pre_parse(self, img, flag=0):
        original = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed = self.pre_process_image(original, flag=flag)
        corners = self.find_corners_of_largest_polygon(processed)
        cropped = self.crop_and_warp(original, corners, flag)
        return cropped


    def parse_grid(self, cropped, flag=0):
        squares = self.infer_grid(cropped)
        digits = self.get_digits(cropped, squares, 28, flag=flag)
        return digits


    def predict_digits(self, model, digits):
        dic = {}
        char = "1234567890"
        for i, c in enumerate(char):
            dic[i + 1] = c
        sudoku = []
        row = []
        for i, dig in enumerate(digits):
            img = dig.reshape(-1, 28, 28, 1)
            pred = model.predict_classes(img)[0]
            try:
                character = dic[pred]
            except:
                character = 0
            row.append(int(character))
            if ((i + 1) % 9 == 0):
                sudoku.append(row)
                row = []
        return np.array(sudoku)


    def draw_text(self, img, sudoku, sudoku_tosolvee):
        side = img.shape[0] / 9
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = (side / 35)
        color = (0, 0, 255)
        thickness = 2
        for i in range(9):
            for j in range(9):
                if sudoku[i][j] == 0:
                    cv2.putText(img, str(sudoku_tosolvee[i][j]),
                                (int(j * side + side / 5), int((i + 1) * side - side / 5)), font, fontScale, color,
                                thickness, cv2.LINE_AA)
        cv2.imshow('Solved', img)
        cv2.waitKey(0)

    def main(self):
        pre = self.pre_parse(self.img.copy())
        digits = self.parse_grid(pre, flag=0)
        model = load_model('digit_model.h5')
        sudoku_predict = self.predict_digits(model, digits)
        sudoku_to_solve = sudo.sudoku(sudoku_predict.copy())
        sudoku_to_solve.solve()
        self.draw_text(pre, sudoku_predict, sudoku_to_solve.bo)


