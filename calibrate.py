import numpy as np
import cv2
import glob


def camera_calibration(images, board_w, board_h, board_size):
    # 设置迭代终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 获取棋盘格角点的世界坐标
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp = objp * board_size

    # 存储对象点和图像点的数组
    obj_points = []  # 在世界坐标系中的3D点
    img_points = []  # 在图像平面的2D点

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
    # 计算重投影误差
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        print(f"image {i} error: {error}")
        mean_error += error
    print("total error: ", mean_error / len(obj_points))
    #
    return ret, mtx, dist, rvecs, tvecs


def image_point_to_latlon(image_point, camera_position, rvec, tvec, mtx, dist):
    # Step 1: Undistort the image point
    image_point_undistorted = cv2.undistortPoints(np.array([image_point]), mtx, dist)

    # Step 2: Convert image coordinates to normalized camera coordinates
    x_normalized = (image_point_undistorted[0][0][0] - mtx[0, 2]) / mtx[0, 0]
    y_normalized = (image_point_undistorted[0][0][1] - mtx[1, 2]) / mtx[1, 1]

    # Step 3: Convert normalized camera coordinates to world coordinates
    camera_rotation_matrix = cv2.Rodrigues(rvec)[0]
    world_point = np.dot(np.linalg.inv(camera_rotation_matrix), np.array([x_normalized, y_normalized, 1]))

    # Step 4: Convert world coordinates to lat/lon
    earth_radius = 6371000  # in meters
    dx, dy, dz = world_point * camera_position[2] / world_point[2]
    lat = camera_position[0] + (dy / earth_radius) * (180 / np.pi)
    lon = camera_position[1] + (dx / (earth_radius * np.cos(camera_position[0]))) * (180 / np.pi)

    return lat, lon


if __name__ == "__main__":
    # 棋盘格角点数
    board_w = 9
    board_h = 6
    # 棋盘格尺寸，单位mm
    board_size = 25.0
    
    ret, mtx, dist, rvecs, tvecs = camera_calibration(glob.glob("images/*.jpg"), board_w, board_h, board_size)
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    
    # Example usage
    image_point = (100, 200)
    camera_position = (37.7749, -122.4194, 100)  # lat, lon, altitude
    rvec = rvecs[0]
    tvec = tvecs[0]
    lat, lon = image_point_to_latlon(image_point, camera_position, rvec, tvec, mtx, dist)
    print(lat, lon)
