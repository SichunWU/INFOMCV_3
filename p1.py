import cv2
import numpy as np
import glob
import time

w = 8
h = 6

# termination criteria, maximum number of loops = 30 and maximum error tolerance = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# checkerboard points in the world coordinate system 3D
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

# checkerboard points in the world coordinate system 2D, grid length = 23mm
subWorkCoord = np.zeros((w * h, 2), np.float32)
subWorkCoord[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * 115
subWorkCoord = np.array(subWorkCoord, np.float32)

# Store the world coordinates and image coordinates of the checkerboard grid
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Store the mouse click number, click coordinates, manually computed image coordinates
clickNum = 0
coordinates = []
subCoord = []

# image number for saving the graph
imgNum = 1

fName = []

# Display the coordinates of the points clicked on the image and find all points
def click_event(event, x, y, flags, params):
    global coordinates, subCoord, clickNum
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # displaying the coordinates on the Shell
        if clickNum < 4:
            #cv2.putText(params, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
            cv2.circle(params, (x, y), 1, (128, 0, 0), -1)
            cv2.imshow('findCorners', params)
            coordinates.append([x, y])
        clickNum += 1
        if clickNum == 4:
            subcoordinates(params)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # displaying the coordinates on the Shell
        if clickNum < 4:
            b = params[y, x, 0]
            g = params[y, x, 1]
            r = params[y, x, 2]
            #cv2.putText(params, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
            cv2.circle(params, (x, y), 1, (0, 255, 0), -1)
            cv2.imshow('findCorners', params)
            coordinates.append([x, y])
        clickNum += 1
        if clickNum == 4:
            subcoordinates(params)

# Calculate the transformation matrix and use it to find all points
def subcoordinates(img):
    global coordinates, subCoord, clickNum, imgNum, fName

    # compute all coordinates
    worldCoord = np.array([[0,(h-1)*115], [(w-1)*115,(h-1)*115], [(w-1)*115,0], [0,0]], np.float32)
    coordinates_array = np.array(coordinates, np.float32)
    M = cv2.getPerspectiveTransform(worldCoord, coordinates_array)
    res = cv2.perspectiveTransform(subWorkCoord.reshape(-1, 1, 2), M)

    # save the corners into a file
    corners = np.array(res, np.float32)
    # print(corners)
    newfile = fName.replace('capture.jpg', 'imageCorners.xml')
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    fs.write("corners", corners)

    # show the chessboard grid
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # cv2.drawChessboardCorners(img, (w, h), corners, True)
    corners = np.int32(corners)
    corner_points = corners.reshape(-1, 2)
    for point in corner_points:
        cv2.circle(img, tuple(point), 1, (0, 0, 255), -1)
    cv2.imshow('findCorners', img)

    # save the corners
    objpoints.append(objp)
    imgpoints.append(np.array(corners, np.float32))

    # save the image
    # cv2.imwrite("./ChessboardCornersImg/Run1/image{}.jpg".format(imgNum), img)
    # imgNum += 1

    #clear the numbers
    coordinates = []
    subCoord = []
    clickNum = 0

# first run
def firstRun(filename):
    global imgNum, fName
    fName = filename
    images = glob.glob(filename)
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # If found, save the information
        if ret == True:
            # Finding sub-pixel corners based on the original corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            # Draw and save the corners
            cv2.drawChessboardCorners(img, (w, h), corners2, ret)
            #cv2.imwrite("./ChessboardCornersImg/Run1/image{}.jpg".format(imgNum), img)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.waitKey(200)
            imgNum += 1
        else:
            if i == 0:
                # win32api.MessageBox(0, "If detect corners fail, please choose 4 corners clockwise, "
                #                       "starting from the top-left.","Notice", win32con.MB_OK)
                i += 1
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.setMouseCallback('findCorners', click_event, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # np.save('./CameraParams/Run1/mtx.npy', mtx)
    # np.save('./CameraParams/Run1/dist.npy', dist)
    # np.save('./CameraParams/Run1/rvecs.npy', rvecs)
    # np.save('./CameraParams/Run1/tvecs.npy', tvecs)

    # reject low quality results
    # rejection(rvecs, tvecs, mtx, dist, gray, 1)

# second and third run
def run(round, filename):
    global objpoints, imgpoints
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(filename)
    i = 0
    for fname in images:
        if i < round:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
            # If found, save the information
            if ret == True:
                # Finding sub-pixel corners based on the original corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                # Draw and save the images
                cv2.drawChessboardCorners(img, (w, h), corners2, ret)
                # if round == 10:
                #     cv2.imwrite("./ChessboardCornersImg/Run2/image{}.jpg".format(i+1), img)
                # if round == 5:
                #     cv2.imwrite("./ChessboardCornersImg/Run3/image{}.jpg".format(i+1), img)

                print(fname)
                # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('findCorners', 640, 480)
                # cv2.imshow('findCorners', img)
                # cv2.waitKey(1000)
                i += 1
        else:
            break
        cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(filename, i)
    # runNum = 0
    # if round == 10:
    #     runNum = 2
    # if round == 5:
    #     runNum = 3
    # np.save('./CameraParams/Run{}/mtx.npy'.format(runNum), mtx)
    # np.save('./CameraParams/Run{}/dist.npy'.format(runNum), dist)
    # np.save('./CameraParams/Run{}/rvecs.npy'.format(runNum), rvecs)
    # np.save('./CameraParams/Run{}/tvecs.npy'.format(runNum), tvecs)

    newfile = filename.replace('capture/*.jpg', 'intrinsics.xml')
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    fs.write("mtx", mtx)
    fs.write("dist", dist)
    # reject low quality results
    # rejection(rvecs, tvecs, mtx, dist, gray, runNum)

# draw XYZ axis
def draw_axis(img, corner, imgpts):
    img = cv2.line(img, corner, tuple(np.int32(imgpts[0][0])), (255,0,0), 2)
    img = cv2.line(img, corner, tuple(np.int32(imgpts[1][0])), (0,255,0), 2)
    img = cv2.line(img, corner, tuple(np.int32(imgpts[2][0])), (0,0,255), 2)
    return img

# draw contours online
def draw_cube(img, imgpts, shadowpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Draw floor
    img = cv2.drawContours(img, [imgpts[:4]], -1, (29,133,223), -2)

    # Draw shadow
    shadowpts = np.int32(shadowpts).reshape(-1, 2)
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(shadowpts[i]), tuple(shadowpts[j]), (120, 120, 120), 2)
    img = cv2.drawContours(img, [shadowpts[4:]], -1, (120, 120, 120), 2)

    # Draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (29,133,223), 2)
    # Draw top
    img = cv2.drawContours(img, [imgpts[4:]], -1, (29,133,223), 2)
    return img

# cast shadow
def shadow(p1, p2):
    # Define the xOy axis plane as z = 0
    plane_normal = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 0])

    # Compute the direction vector of the line connecting the two points
    line_direction = p2 - p1

    # Compute the parameter t of the line at which it intersects the plane
    t = np.dot(plane_normal, (plane_point - p1)) / np.dot(plane_normal, line_direction)

    # Compute the intersection point
    intersection_point = p1 + t * line_direction
    return intersection_point

# Online phase: Capture picture using webcam
def online(mtx, dist, rvecs, tvecs, run):
    camera = cv2.VideoCapture(0)

    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Position of the light source
    light = np.float32([-4, -4, -6])

    # Axes endpoints
    axis = np.float32([[5,0,0], [0,5,0], [0,0,-5]])

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

            axispts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            draw_axis(frame, np.int32(corners2[0][0]), axispts)

            t = (time.time()%12)/6
            
            var1 = np.cos(t*np.pi)*np.sqrt(2)
            var2 = np.sin(t*np.pi)*np.sqrt(2)
            p000 = np.float32([1+var1, 1+var2, 0])
            p001 = np.float32([1+var2, 1-var1, 0])
            p010 = np.float32([1-var1, 1-var2, 0])
            p011 = np.float32([1-var2, 1+var1, 0])
            p100 = np.float32([1+var1, 1+var2, -2])
            p101 = np.float32([1+var2, 1-var1, -2])
            p110 = np.float32([1-var1, 1-var2, -2])
            p111 = np.float32([1-var2, 1+var1, -2])
            cube = np.float32([p000, p001, p010, p011, p100, p101, p110, p111])
            
            # Project 3D points to image plane
            cubepts, jac = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
            # Project shadow of the cube to the chessboard from a fixed light
            shadowpts = np.float32([p000, p001, p010, p011, shadow(p100, light), shadow(light, p101), shadow(light, p110), shadow(light, p111)])

            shadowpts, jac = cv2.projectPoints(shadowpts, rvecs, tvecs, mtx, dist)
                                               
            draw_cube(frame, cubepts, shadowpts)

        cv2.imshow('Camera', frame)
        k = cv2.waitKey(1)
        if k == 27 or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:   #press Esc to quit
            break
        if k == ord('s'):   # Save the captured image to the disk
            cv2.imwrite("./ChessboardCornersImg/Run{}_imageOnline.jpg".format(run), frame)
            break

    camera.release()
    cv2.destroyAllWindows()

# load parameters
def onlineRun(run):
    mtx = np.load('./CameraParams/Run{}/mtx.npy'.format(run))
    dist = np.load('./CameraParams/Run{}/dist.npy'.format(run))
    rvecs = np.load('./CameraParams/Run{}/rvecs.npy'.format(run))
    tvecs = np.load('./CameraParams/Run{}/tvecs.npy'.format(run))
    online(mtx, dist, rvecs, tvecs, run)
    #print(run,". Intrinsic matrix K: \n", mtx)

# calculate the mean and SD of error, then reject low quality pics based on them
def rejection(rvecs, tvecs, mtx, dist, gray, runNum):
    global objpoints, imgpoints
    # Calculate the mean error
    meanError = 0
    for i in range(len(objpoints)):
        imgpointsNew, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpointsNew, cv2.NORM_L2) / len(imgpointsNew)
        meanError += error
    meanError /= len(objpoints)

    # Calculate the standard deviation
    SD = 0
    for i in range(len(objpoints)):
        imgpointsNew, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpointsNew, cv2.NORM_L2) / len(imgpointsNew)
        SD += (error - meanError) ** 2
    SD = np.sqrt(SD /len(objpoints))

    print("Mean Error:", meanError)
    print("Standard Deviation:", SD)

    # Remove images from the calibration process
    try:
        for i in range(len(objpoints) - 1, -1, -1):
            imgpointsNew, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpointsNew, cv2.NORM_L2) / len(imgpointsNew)
            if error > meanError + 2 * SD:    #delete the data if thw reprojection error > 2 * SD
                objpoints.pop(i)
                imgpoints.pop(i)
                # print("i", i)
    except:
        print("Error, please try again.")
        pass
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print("Renew", mtx)
    np.save('./CameraParams/Run{}/mtxRenew.npy'.format(runNum), mtx)
    np.save('./CameraParams/Run{}/distRenew.npy'.format(runNum), dist)
    np.save('./CameraParams/Run{}/rvecsRenew.npy'.format(runNum), rvecs)
    np.save('./CameraParams/Run{}/tvecsRenew.npy'.format(runNum), tvecs)

# offline phase, 3 runs
def offlinePhase():
    firstRun()  # run1
    run(10)     # run2
    run(5)      # run3

# online phase, 3 runs
def onlinePhase():
    onlineRun(1)
    onlineRun(2)
    onlineRun(3)

if __name__ == "__main__":
    offlinePhase()
    onlinePhase()
