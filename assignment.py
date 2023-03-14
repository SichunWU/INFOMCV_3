import os
import glm
import random
import numpy as np
import cv2
import p1
import p3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

camera_handles = []
background_models = []

block_size = 3.0
w = 8
h = 6
prevForeground = [None for _ in range(4)]
lookup = None

path_bg = ["./4persons/background/Take26.54389819.20141124164130.avi",
          "./4persons/background/Take26.59624062.20141124164130.avi",
          "./4persons/background/Take26.60703227.20141124164130.avi",
          "./4persons/background/Take26.62474905.20141124164130.avi"]

path_video = ["./4persons/video/Take30.54389819.20141124164749.avi",
         "./4persons/video/Take30.59624062.20141124164749.avi",
         "./4persons/video/Take30.60703227.20141124164749.avi",
         "./4persons/video/Take30.62474905.20141124164749.avi"]

def draw_mesh(positions):
    voxel = np.int32(positions)
    width = np.max(voxel[:, 0]) - np.min(voxel[:, 0])
    depth = np.max(voxel[:, 1]) - np.min(voxel[:, 1])
    height = np.max(voxel[:, 2]) - np.min(voxel[:, 2])

    grid = np.zeros((width+1, height+1, depth+1), dtype=bool)

    print(grid.shape)

    for i in range(len(voxel)):
        grid[voxel[i][0] - np.min(voxel[:, 0])][1 - (voxel[i][2] - np.min(voxel[:, 2]))][voxel[i][1] - np.min(voxel[:, 1])] = True

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(grid, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, int(1.5 * width))  
    ax.set_ylim(0, int(1.5 * height))  
    ax.set_zlim(0, int(1.5 * depth))  

    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def saveTable(lookup):
    newfile = './4persons/video/lookupTable.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    for i in range(4):
        flag = np.array([x[0] for x in lookup[i]])
        coord = np.array([x[1] for x in lookup[i]])
        fs.write("flag", flag)
        fs.write("coord", coord)
    fs.release()

def saveCoord(coord, path):
    path = path[0][43:-4]
    newfile = './4persons/video/voxelCoords' + path + '.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    fs.write("coord", np.array(coord))
    fs.release()

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    width = 100
    depth = 100
    data, colors = [], []
    for x in range(-20, width-20):
        for z in range(-30, depth-30):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0.5, 0.5, 0.5])
    return data, colors


def set_voxel_positions(width, height, depth, bg, pressNum):
    # Generates random voxel locations
    global prevForeground, lookup
    width = 25
    height = 25
    depth = 25

    voxel_size = 0.4
    data0 = []
    colors = []
    for x in np.arange(-3, width-3, voxel_size):
        for y in np.arange(-2, height-2, voxel_size):
            for z in np.arange(-3, depth-3, voxel_size):
                data0.append([x, y, z])

    # flags to save data and 2D coordinates
    flags = [[[[], []] for _ in range(len(data0))] for _ in range(4)]
    data0 = np.float32(data0)
    # first frame, compute lookup table
    if (pressNum <= 2726 ):     #total frame number
    # if (pressNum == 0):
        for i in range(4):
            # foreground = cv2.imread(bg[i])

            # train MOG2 on background video, remove shadows, default learning rate
            background_models.append(cv2.createBackgroundSubtractorMOG2())
            background_models[i].setShadowValue(0)

            # open background.avi
            camera_handle = cv2.VideoCapture(path_bg[i])
            num_frames = int(camera_handle.get(cv2.CAP_PROP_FRAME_COUNT))

            # train background model on each frame
            for i_frame in range(num_frames):
                ret, image = camera_handle.read()
                if ret:
                    background_models[i].apply(image)

            # close background.avi
            camera_handle.release()

            # open video.avi
            camera_handles = cv2.VideoCapture(path_video[i])
            fn = 0
            while True:
                # read frame
                ret, image = camera_handles.read()
                # cv2.imshow('foreground', image)
                # cv2.waitKey(20)
                if fn == pressNum:
                    # determine foreground
                    foreground = background_subtraction(image, background_models[i])
                    if i == 1:
                        knnImage = image
                        knnFg = foreground
                    break
                fn += 1

            newpath = bg[i].replace('video/Take30', 'extrinsics/Take25')
            newpath = newpath[:38] + 'config.xml'
            fs = cv2.FileStorage(newpath, cv2.FILE_STORAGE_READ)
            mtx = fs.getNode('mtx').mat()
            dist = fs.getNode('dist').mat()
            rvec = fs.getNode('rvec').mat()
            tvec = fs.getNode('tvec').mat()

            pts, jac = cv2.projectPoints(data0, rvec, tvec, mtx, dist)
            pts = np.int32(pts)

            for j in range(len(data0)):
                #cv2.circle(image, tuple([pts[j][0][0], pts[j][0][1]]), 1, (0, 0, 255), -1)
                try:
                    if foreground[pts[j][0][1]][pts[j][0][0]].sum() == 0:   # if point falls into the background
                        flags[i][j] = [0, [pts[j][0][1], pts[j][0][0]]]
                    else:
                        flags[i][j] = [1, [pts[j][0][1], pts[j][0][0]]]
                except:
                    flags[i][j] = [0, [pts[j][0][1], pts[j][0][0]]]
                    continue
            prevForeground[i] = foreground
            # cv2.imshow('foreground', image)
            # cv2.waitKey(20)
        lookup = flags
        saveTable(lookup)

    # the rest frames, XOR result not fully correct
    # else:
    #     for i in range(4):
    #         # train MOG2 on background video, remove shadows, default learning rate
    #         background_models.append(cv2.createBackgroundSubtractorMOG2())
    #         background_models[i].setShadowValue(0)
    #         # open background.avi
    #         camera_handle = cv2.VideoCapture(path_bg[i])
    #         num_frames = int(camera_handle.get(cv2.CAP_PROP_FRAME_COUNT))
    #         # train background model on each frame
    #         for i_frame in range(num_frames):
    #             ret, image = camera_handle.read()
    #             if ret:
    #                 background_models[i].apply(image)
    #         # close background.avi
    #         camera_handle.release()
    #         # open video.avi
    #         camera_handles.append(cv2.VideoCapture(path_video[i]))
    #
    #         fn = 0
    #         while True:
    #             # read frame
    #             ret, image = camera_handles[i].read()
    #             if fn == pressNum:
    #                 # determine foreground
    #                 foreground = background_subtraction(image, background_models[i])
    #                 cv2.imshow('foreground', foreground)
    #                 cv2.waitKey(20)
    #                 break
    #             fn += 1
    #
    #         # foreground = cv2.imread(bg[i])
    #         # create a dictionary for the coordinate
    #         coordDict = {tuple(lu[1]): ind for ind, lu in enumerate(lookup[i])}
    #         # change in the images compared to the previous one
    #         diff = cv2.bitwise_xor(foreground, prevForeground[i])
    #         change = np.where(diff > 0)
    #         points = []
    #         for j in range(len(change[0])):
    #             points.append([change[0][j], change[1][j]])
    #         # delete the duplicated points
    #         uniPoints = [list(t) for t in set(tuple(element) for element in points)]
    #         # change the flag of the different points
    #         for j in range(len(uniPoints)):
    #             try:
    #                 index = coordDict[(uniPoints[j][0], uniPoints[j][1])]
    #                 if lookup[i][index][0] == 1:
    #                     lookup[i][index][0] = 0
    #                 else:
    #                     lookup[i][index][0] = 1
    #             except:
    #                 continue
    #         prevForeground[i] = foreground
    camera_handle.release()
    cv2.destroyAllWindows()

    data = []
    columnSum = np.zeros(len(data0))
    # colorpath = './4persons/video/Take30.59624062.video.jpg'
    # clip = cv2.imread(colorpath)
    for i in range(len(data0)):
        for j in range(len(lookup)):
            columnSum[i] += lookup[j][i][0]

    # if voxels in all views are visible, show it on the screen
    for i in range(len(data0)):
        if columnSum[i] == 4:
            data.append(data0[i])
    #       colors.append(clip[lookup[1][i][1][0]][lookup[1][i][1][1]] / 256)

    saveCoord(data, bg)

    # cluster the voxels
    p3.knn(pressNum, knnImage, knnFg)

    # rotate array -90 degree along the x-axis.
    Rx = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    dataR = [Rx.dot(p) for p in data]
    dataR = [np.multiply(DR, 5) for DR in dataR]
    return dataR, colors


# applies background subtraction to obtain foreground mask
def background_subtraction(image, background_model):
    foreground_image = background_model.apply(image, learningRate=0)

    # remove noise through dilation and erosion
    erosion_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_image = cv2.dilate(foreground_image, dilation_elt)
    foreground_image = cv2.erode(foreground_image, erosion_elt)

    # Remove unconnected parts and only keep the horseman
    _, thresh = cv2.threshold(foreground_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # get binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    min_size = 4500  # minimum size of connected component to keep
    mask2 = np.zeros(thresh.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask2[labels == i] = 255
    result = cv2.bitwise_and(foreground_image, foreground_image, mask=mask2)

    kernel = np.ones((2, 2), np.uint8)
    result = cv2.erode(result, kernel, iterations=2)  # remove small isolated pixels

    return result   #foreground_image



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
        Vposition = np.array([position[0]*5, position[2]*5, position[1] * 5])
        cam_position.append(Vposition)
        color = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_position, color


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
def getExtrinsics(path1, path2):
    # get the image points (only run once)
    # cap = cv2.VideoCapture('./data/cam{}/checkerboard.avi'.format(Cam))
    # cap = cv2.VideoCapture(path1)
    # ret, frame = cap.read()
    # # filename = f'./data/cam{Cam}/capture.jpg'
    # spl = path1.split('/')
    # splName = spl[3][:15]
    # filename = f'./4persons/extrinsics/{splName}.jpg'
    # cv2.imwrite(filename, frame)
    # p1.firstRun(filename)

    # load the image points and intrinsics
    # fs = cv2.FileStorage('./data/cam{}/imageCorners.xml'.format(Cam), cv2.FILE_STORAGE_READ)
    fs = cv2.FileStorage(path2, cv2.FILE_STORAGE_READ)
    data = fs.getNode('corners').mat()
    imageCorners = np.array(data)
    # fs = cv2.FileStorage('./data/cam{}/intrinsics.xml'.format(Cam), cv2.FILE_STORAGE_READ)
    fs = cv2.FileStorage('./data/cam{}/intrinsics.xml'.format(1), cv2.FILE_STORAGE_READ)
    mtx = fs.getNode('mtx').mat()
    dist = fs.getNode('dist').mat()

    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # get the extrinsic parameters
    retval, rvec, tvec = cv2.solvePnP(objp, imageCorners, mtx, dist)
    R, _ = cv2.Rodrigues(rvec)  # change rotation vector to matrix
    T, _ = cv2.Rodrigues(tvec)  # change translation vector to matrix
    newfile = path2[:-16] + 'config.xml'
    # fs = cv2.FileStorage('./data/cam{}/config.xml'.format(Cam), cv2.FILE_STORAGE_WRITE)
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    fs.write("mtx", mtx)
    fs.write("dist", dist)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)

    # draw the X, Y, Z axes on the image
    spl = path2.split('/')
    splName = spl[3][:15]
    newfileDraw = f'./4persons/extrinsics/{splName}.jpg'
    # img = cv2.imread('./data/cam{}/capture.jpg'.format(Cam))
    img = cv2.imread(newfileDraw)

    axis = np.float32([[5,0,0], [0,5,0], [0,0,5]])
    axispts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    p1.draw_axis(img, np.int32(imageCorners[0][0]), axispts)
    newfileSave = path2[:-16] + 'capture.jpg'
    # cv2.imwrite('./data/cam{}/captureXYZ.jpg'.format(Cam), img)
    cv2.imwrite(newfileSave, img)

    cv2.imshow('capture', img)
    cv2.waitKey(0)
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
def backgroundModel(path):
    #cap = cv2.VideoCapture('./data/cam{}/background.avi'.format(Cam))
    cap = cv2.VideoCapture(path)
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

    avgFrame = (frams/i).astype('uint8')

    cv2.imshow('Background Model', avgFrame)
    newfile = path[:-18] + 'background.jpg'
    cv2.imwrite(newfile, avgFrame)
    #cv2.imwrite('./data/cam{}/background.jpg'.format(Cam), avgFrame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# background subtraction (this does not work)
def backgroundSub2(path1, path2, threshold_h, threshold_s, threshold_v):
    # background = cv2.imread('./data/cam{}/background.jpg'.format(Cam))
    background = cv2.imread(path1)
    hsv_bg = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    # frame = cv2.imread('./data/cam{}/video.jpg'.format(Cam))
    frame = cv2.imread(path2)
    
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Apply the mask to the foreground image
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Set the foreground pixels to white (255) and the background pixels to black (0)
    result[mask == 255] = (255, 255, 255)
    result[mask == 0] = (0, 0, 0)

    cv2.imshow('result', result)
    newfile = path2[:-9] + 'foreground.jpg'
    cv2.imwrite(newfile, result)
    cv2.waitKey(0)


    # cv2.imwrite('./data/cam{}/hsv.jpg'.format(Cam), result)

# background subtraction
def backgroundSub(path1, path2, time):
    # background = cv2.imread('./data/cam{}/background.jpg'.format(Cam))
    background = cv2.imread(path1)
    grayBG = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # cap = cv2.VideoCapture('./data/cam{}/video.avi'.format(Cam))
    cap = cv2.VideoCapture(path2)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameNum = int(time * fps)

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    capFrams = []
    for i in range(0, totalFrames, 100):
        capFrams.append(i)
    print(capFrams)

    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn in capFrams:
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sub = cv2.absdiff(grayFrame, grayBG)                        # subtract the background
            # _, mask = cv2.threshold(sub, 50, 225, cv2.THRESH_BINARY)    # create a mask of the foreground
            _, mask = cv2.threshold(sub, 27, 225, cv2.THRESH_BINARY)  # create a mask of the foreground

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

            kernel = np.ones((2, 2), np.uint8)
            result = cv2.erode(result, kernel, iterations=2)  # remove small isolated pixels

            cv2.imshow('result', result)
            newfile = path2[:-18] + f'foreground{fn}.jpg'
            # cv2.imwrite('./data/cam{}/frames/foreground{}.jpg'.format(Cam, framNum), result)
            cv2.imwrite(newfile, result)
            cv2.waitKey(20)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        fn += 1
    cap.release()
    cv2.destroyAllWindows()


# run background subtraction
def bgSubtraction(): 
    for i in range(4):
        # backgroundModel(i+1)
        backgroundSub(i+1)

from engine.config import config

window_width, window_height = config['window_width'], config['window_height']

if __name__ == '__main__':
    for i in range(4):
         backgroundSub2(i+1, 110, 180, 40)
    # bg1 = cv2.imread('./data/cam{}/frames/foreground{}.jpg'.format(1, 0))
    # bg2 = cv2.imread('./data/cam{}/frames/foreground{}.jpg'.format(2, 0))
    # bg3 = cv2.imread('./data/cam{}/frames/foreground{}.jpg'.format(3, 0))
    # bg4 = cv2.imread('./data/cam{}/frames/foreground{}.jpg'.format(4, 0))
    # bg = [bg1, bg2, bg3, bg4]
    # positions, colors = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'], bg, 0)
    # draw_mesh(positions)
