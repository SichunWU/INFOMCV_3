import glm
import random
import numpy as np
import cv2
import p1

block_size = 1.0
w = 8
h = 6

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations

# capture images from intrinsics.avi
def capture(Cam):
    cap = cv2.VideoCapture('./data/cam{}/intrinsics.avi'.format(Cam))
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = random.sample(range(totalFrames), 50)
    frameIndex = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameIndex in index:
            filename = f'./data/cam{Cam}/capture/capturedImg_{frameIndex:04d}.jpg'
            cv2.imwrite(filename, frame)
        frameIndex += 1
    cap.release()
    cv2.destroyAllWindows()

# get the extrinsic parameters
def getExtrinsics(Cam):
    # get the image points (only run once)
    # cap = cv2.VideoCapture('./data/cam{}/checkerboard.avi'.format(Cam))
    # ret, frame = cap.read()
    # filename = f'./data/cam{Cam}/capture.jpg'
    # cv2.imwrite(filename, frame)
    # p1.firstRun(filename)

    # load the image points and intrinsics
    fs = cv2.FileStorage('./data/cam{}/imageCorners.xml'.format(Cam), cv2.FILE_STORAGE_READ)
    data = fs.getNode('corners').mat()
    imageCorners = np.array(data)
    fs = cv2.FileStorage('./data/cam{}/intrinsics.xml'.format(Cam), cv2.FILE_STORAGE_READ)
    mtx = fs.getNode('mtx').mat()
    dist = fs.getNode('dist').mat()

    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * 115

    # get the extrinsic parameters
    retval, rvec, tvec = cv2.solvePnP(objectPoints=objp, imagePoints=imageCorners, cameraMatrix=mtx,
                                      distCoeffs=dist)
    R, _ = cv2.Rodrigues(rvec)  # change rotation vector to matrix
    T, _ = cv2.Rodrigues(tvec)  # change translation vector to matrix

    fs = cv2.FileStorage('./data/cam{}/config.xml'.format(Cam), cv2.FILE_STORAGE_WRITE)
    fs.write("mtx", mtx)
    fs.write("dist", dist)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)

    # draw the X, Y, Z axes on the image
    img = cv2.imread('./data/cam{}/capture.jpg'.format(Cam))
    worldPoints = np.float32([[0, 0, 0], [300, 0, 0], [0, 300, 0], [0, 0, 300]])
    image_pts, _ = cv2.projectPoints(worldPoints, rvec, tvec, mtx, dist)

    image_pts = np.int32(image_pts).reshape(-1, 2)
    img = cv2.line(img, tuple(image_pts[0]), tuple(image_pts[1]), (0, 0, 255), 1)  # X axis
    img = cv2.line(img, tuple(image_pts[0]), tuple(image_pts[2]), (0, 255, 0), 1)  # Y axis
    img = cv2.line(img, tuple(image_pts[0]), tuple(image_pts[3]), (255, 0, 0), 1)  # Z axis
    cv2.imwrite('./data/cam{}/captureXYZ.jpg'.format(Cam), img)

    cv2.imshow('capture', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


# find corners automatically, this does not work, can only get the points of the paper.
def findCorners(Cam):
    cap = cv2.VideoCapture('./data/cam{}/checkerboard.avi'.format(Cam))
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

    # find the polygon contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
    # cv2.drawContours(frame, [approx], 0, (128, 0, 0), 1)

    # get the corner points
    cornerPoints = approx.reshape(-1, 2)
    # print(corner_points)
    for point in cornerPoints:
        cv2.circle(frame, tuple(point), 1, (0, 0, 255), -1)

    # set the pixels outside the polygon to green
    # mask = np.zeros(frame.shape[:2], np.uint8)
    # cv2.drawContours(mask, [approx], 0, 255, -1)
    # result = np.full(frame.shape, (0, 128, 0), dtype=np.uint8)
    # result[mask != 0] = frame[mask != 0]
    # cv2.imshow('image', result)
    # filename = f'./data/cam{Cam}/extractChessboard.jpg'
    # cv2.imwrite(filename, result)

    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cornerPoints

# get camera parameters (only run once)
def getCameraParam():
    for i in range(4):
        capture(i+1)            # capture images
        filename = f'./data/cam{i+1}/capture/*.jpg'
        p1.run(25, filename)    # get intrinsics
        findCorners(i+1)        # find corners automatically (doesn't work)
        getExtrinsics(i+1)      # get extrinsics

if __name__ == '__main__':
    print('1')

