import cv2

from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')

from towhee.functional import DataCollection

scale = 0.1


def openpose(oriImg):
    w = int(oriImg.shape[1] * scale)
    h = int(oriImg.shape[0] * scale)
    im = cv2.resize(oriImg, (w, h))
    # oriImg = im
    candidate, subset = body_estimation(im)
    canvas = oriImg#copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset, 1/scale)
    return canvas

DataCollection.from_camera(1) \
    .map(openpose) \
    .imshow()
