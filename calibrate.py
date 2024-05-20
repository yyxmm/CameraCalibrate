import numpy as np
import cv2
import glob

# 棋盘格角点数
board_w = 9
board_h = 6
# 棋盘格尺寸，单位mm
board_size = 25.0

# 设置迭代终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# 获取棋盘格角点的世界坐标
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
objp = objp * board_size

# 存储对象点和图像点的数组
obj_points = []  # 在世界坐标系中的3D点
img_points = []  # 在图像平面的2D点

images = glob.glob("images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘角点
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None, flags)

    # 如果找到，添加对象点，图像点
    if ret:
        # 亚像素精确化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 保存对象点和图像点
        obj_points.append(objp)
        img_points.append(corners)

        # 绘制并显示角点
        cv2.drawChessboardCorners(img, (board_w, board_h), corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
