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
    width = 16
    height = 8
    depth = 16
    #cube_num = width * height * depth
    #flags = np.ones((4, cube_num))
    voxel_size = 0.2
    data0 = []
    for x in np.arange(0, width, voxel_size):
        for y in np.arange(0, height, voxel_size):
            for z in np.arange(0, depth, voxel_size):
                data0.append([x, y, z])

    flags = np.ones((4, len(data0)))
    data0 = np.float32(data0)
    #print(len(data0))

    for i in range(4):
        foreground = cv2.imread('./data/cam{}/foreground.jpg'.format(i + 1))
        fs = cv2.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv2.FILE_STORAGE_READ)
        mtx = fs.getNode('mtx').mat()
        dist = fs.getNode('dist').mat()
        rvec = fs.getNode('rvec').mat()
        tvec = fs.getNode('tvec').mat()

        pts, jac = cv2.projectPoints(data0, rvec, tvec, mtx, dist)

        pts = np.int32(pts)
        for j in range(len(data0)):    #cube_num
            try:
                if foreground[pts[j][0][1]][pts[j][0][0]].sum() == 0:
                     flags[i][j] = 0
            except:
                print("Out of range!")
                continue

        # cv2.imshow('Foreground Mask', foreground)
        # cv2.waitKey(2000)

    cv2.destroyAllWindows()

    data = []
    columnSum = flags.sum(axis=0)
    #print(columnSum, len(columnSum))
    for i in range(len(data0)):
        if columnSum[i] == 4:
            data.append(data0[i])

    # rotate array -90 degree along the x-axis.
    Rx = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    dataR = [Rx.dot(p) for p in data]

    return dataR


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cam_position = []
    for i in range(4):
        fs = cv2.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv2.FILE_STORAGE_READ)
        tvec = fs.getNode('tvec').mat()
        rvec = fs.getNode('rvec').mat()
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        position = -R_inv.dot(tvec)     # get camera position
        # get camera position in voxel space units(swap the y and z coordinates)
        Vposition = np.array([position[0], position[2], position[1] * 2.0])
        cam_position.append(Vposition)

    return cam_position


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])

    cam_rotations = []
    for i in range(4):
        fs = cv2.FileStorage('./data/cam{}/config.xml'.format(i+1), cv2.FILE_STORAGE_READ)
        rvec = fs.getNode('rvec').mat()
        R, _ = cv2.Rodrigues(rvec)

        R[:, 1], R[:, 2] = R[:, 2], R[:, 1].copy()  # swap y and z (exchange the second and third columns)
        R[1, :] = -R[1, :]      # invert rotation on y (multiply second row by -1)
        # rotation matrix: rotation 90 degree about the y
        rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        R = np.matmul(R, rot)

        # convert to mat4x4 format
        RM = np.eye(4)
        RM [:3, :3] = R
        RM  = glm.mat4(*RM .flatten())
        cam_rotations.append(RM)
    #print(cam_rotations)
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
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # get the extrinsic parameters
    retval, rvec, tvec = cv2.solvePnP(objp, imageCorners, mtx, dist)
    R, _ = cv2.Rodrigues(rvec)  # change rotation vector to matrix
    T, _ = cv2.Rodrigues(tvec)  # change translation vector to matrix

    fs = cv2.FileStorage('./data/cam{}/config.xml'.format(Cam), cv2.FILE_STORAGE_WRITE)
    fs.write("mtx", mtx)
    fs.write("dist", dist)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)

    # draw the X, Y, Z axes on the image
    img = cv2.imread('./data/cam{}/capture.jpg'.format(Cam))
    axis = np.float32([[5,0,0], [0,5,0], [0,0,5]])
    axispts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

    p1.draw_axis(img, np.int32(imageCorners[0][0]), axispts)
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


# create a background model by averaging 50 frames
def backgroundModel(Cam):
    cap = cv2.VideoCapture('./data/cam{}/background.avi'.format(Cam))
    frams = None
    i = 0

    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frams is None:
            frams = gray.astype('float')
        else:
            frams += gray.astype('float')
        i += 1

    print(i)
    avgFrame = (frams/i).astype('uint8')

    cv2.imshow('Background Model', avgFrame)
    cv2.imwrite('./data/cam{}/background.jpg'.format(Cam), avgFrame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# background subtraction (this does not work)
def backgroundSub2(Cam, threshold_h, threshold_s, threshold_v):
    cap = cv2.VideoCapture('./data/cam{}/video.avi'.format(Cam))
    background = cv2.imread('./data/cam{}/background.jpg'.format(Cam))

    hsv_bg = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the difference in each channel
        diff_h = cv2.absdiff(hsv[:, :, 0], hsv_bg[:, :, 0])
        diff_s = cv2.absdiff(hsv[:, :, 1], hsv_bg[:, :, 1])
        diff_v = cv2.absdiff(hsv[:, :, 2], hsv_bg[:, :, 2])

        # Threshold the differences


        mask_h = cv2.threshold(diff_h, threshold_h, 255, cv2.THRESH_BINARY)[1]
        mask_s = cv2.threshold(diff_s, threshold_s, 255, cv2.THRESH_BINARY)[1]
        mask_v = cv2.threshold(diff_v, threshold_v, 255, cv2.THRESH_BINARY)[1]

        # Combine the masks
        mask = cv2.bitwise_or(mask_h, mask_s)
        mask = cv2.bitwise_or(mask, mask_v)

        # Apply morphology operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Apply the mask to the foreground image
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Set the foreground pixels to white (255) and the background pixels to black (0)
        result[mask == 255] = (255, 255, 255)
        result[mask == 0] = (0, 0, 0)

        cv2.imshow('Foreground Mask', mask)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# background subtraction
def backgroundSub(Cam):
    background = cv2.imread('./data/cam{}/background.jpg'.format(Cam))
    grayBG = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture('./data/cam{}/video.avi'.format(Cam))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sub = cv2.absdiff(grayFrame, grayBG)                        # subtract the background
        _, mask = cv2.threshold(sub, 50, 225, cv2.THRESH_BINARY)    # create a mask of the foreground

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)      # remove noise
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.dilate(morph, kernel, iterations=3)             # fill in small holes

        # Remove unconnected parts and only keep the horseman
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # get binary image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        min_size = 4500         # minimum size of connected component to keep
        mask2 = np.zeros(thresh.shape, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask2[labels == i] = 255
        result = cv2.bitwise_and(morph, morph, mask=mask2)

        # kernel = np.ones((2, 2), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=2)  # remove small isolated pixels

        # cv2.imshow('Origin', frame)
        # cv2.imshow('mask', mask)
        # cv2.imshow('morph', morph)
        cv2.imshow('result', result)
        cv2.imwrite('./data/cam{}/foreground.jpg'.format(Cam), result)
        # cv2.waitKey(0)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run background subtraction
def bgSubtraction(): 
    for i in range(4):
        # backgroundModel(i+1)
        backgroundSub(i+1)

if __name__ == '__main__':
    for i in range(4):
        backgroundSub2(i+1, 100, 100, 120)

    # getCameraParam()          # task1
    # bgSubtraction()           # task2
    # get_cam_rotation_matrices()
    # get_cam_positions()
    # set_voxel_positions(16, 8, 16)
