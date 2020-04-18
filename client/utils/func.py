import cv2


def correct(pos, max_size):
    if pos < 0:
        return 0
    elif pos > max_size:
        return max_size
    else:
        return pos


def marks_draw(img, landmarks, start, end):
    for i in range(start, end):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)