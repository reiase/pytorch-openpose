import cv2

from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')

from towhee.functional import DataCollection

scale = 0.1


def resize(im):
    w = int(im.shape[1] * scale)
    h = int(im.shape[0] * scale)
    return cv2.resize(im, (w, h))


def openpose(im):
    candidate, subset = body_estimation(im)
    return candidate, subset


def draw(canvas, result):
    candidate, subset = result
    canvas = util.draw_bodypose(canvas, candidate, subset, 1 / scale)
    return canvas

small, oriImg = DataCollection \
    .from_camera(1) \
    .map(resize, lambda x: x)
oriImg.zip(small.map(openpose)) \
    .map(lambda x: draw(*x)) \
    .imshow()
