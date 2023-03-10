import cv2
import numpy as np

import assignment

def videoFrame(path, time):
    cap = cv2.VideoCapture(path)
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
            newfile = path[:-18] + f'video{fn}.jpg'
            cv2.imwrite(newfile, frame)
        fn += 1
    cap.release()
    cv2.destroyAllWindows()

def loadTable():
    newfile = './4persons/video/lookupTable.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_READ)
    flag = fs.getNode('flag').mat()
    coord = fs.getNode('coord').mat()

def loadCoord(pressNum):
    newfile = f'./4persons/video/voxelCoords{pressNum}.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_READ)
    coord = fs.getNode('coord').mat()
    return coord

def saveLabel(labels, pressNum):
    newfile = f'./4persons/video/voxelCoords{pressNum}.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_APPEND)
    fs.write("label", labels)
    fs.release()


def knn(pressNum):
    coordOrigin = loadCoord(pressNum)
    coord = coordOrigin[:, [0, 1]]    # Remove the height of the voxel
    numClusters = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(coord), numClusters, None, criteria, 10, flags)
    # saveLabel(labels, pressNum)

    labels = np.squeeze(labels)
    # project to a view
    # colors = np.zeros((len(labels), 3))
    # colors[labels == 0] = [255, 0, 0]   # red for label 0
    # colors[labels == 1] = [0, 255, 0]   # green for label 1
    # colors[labels == 2] = [0, 0, 255]   # blue for label 2
    # colors[labels == 3] = [255, 255, 0]  # yellow for label 3
    #
    path = './4persons/extrinsics/Take25.59624062.config.xml'
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mtx = fs.getNode('mtx').mat()
    dist = fs.getNode('dist').mat()
    rvec = fs.getNode('rvec').mat()
    tvec = fs.getNode('tvec').mat()
    #
    # pts, jac = cv2.projectPoints(np.float32(coordOrigin), rvec, tvec, mtx, dist)
    # pts = np.int32(pts)
    # #print(pts)
    img = cv2.imread(f'./4persons/video/Take30.59624062.video{pressNum}.jpg')
    # for i in range(len(pts)):
    #     cv2.circle(img, tuple([pts[i][0][0], pts[i][0][1]]), 2, colors[i], -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    C0 = coordOrigin[labels == 0]
    C1 = coordOrigin[labels == 1]
    C2 = coordOrigin[labels == 2]
    C3 = coordOrigin[labels == 3]
    C2D = [C0, C1, C2, C3]

    RGBdata = []
    for i in range(4):
        pts, jac = cv2.projectPoints(np.float32(C2D[i]), rvec, tvec, mtx, dist)
        pts = np.int32(pts)
        pixels = []
        for j in range(len(pts)):
            pixels.append(img[pts[j][0][1]][pts[j][0][0]].tolist())
            cv2.circle(img, tuple([pts[j][0][0], pts[j][0][1]]), 2, img[pts[j][0][1]][pts[j][0][0]].tolist(), -1)
        RGBdata.append(pixels)

        cv2.imshow('img', img)
        cv2.waitKey(200)
    # print(len(RGBdata))
    # color(RGBdata, pressNum)
    cv2.destroyAllWindows()


# get the GMM parameters for each person
def color(coord, pressNum):
    newfile = f'./4persons/video/colorModel{pressNum}.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    i = 0
    for c in coord:
        gmm = cv2.ml.EM_create()
        gmm.setClustersNumber(3)
        gmm.trainEM(np.array(c))

        means = gmm.getMeans()
        covs = gmm.getCovs()
        weights = gmm.getWeights()

        fs.write(f"means{i}", np.array(means))
        fs.write(f"covs{i}", np.array(covs))
        fs.write(f"weights{i}", np.array(weights))
        i += 1
    fs.release()


if __name__ == "__main__":
    paths1 = ["./4persons/extrinsics/Take25.54389819.20141124164119.avi",
             "./4persons/extrinsics/Take25.59624062.20141124164119.avi",
             "./4persons/extrinsics/Take25.60703227.20141124164119.avi",
             "./4persons/extrinsics/Take25.62474905.20141124164119.avi"]
    paths2 = ["./4persons/extrinsics/Take25.54389819.imageCorners.xml",
              "./4persons/extrinsics/Take25.59624062.imageCorners.xml",
              "./4persons/extrinsics/Take25.60703227.imageCorners.xml",
              "./4persons/extrinsics/Take25.62474905.imageCorners.xml"]

    paths3 = ["./4persons/background/Take26.54389819.20141124164130.avi",
             "./4persons/background/Take26.59624062.20141124164130.avi",
             "./4persons/background/Take26.60703227.20141124164130.avi",
             "./4persons/background/Take26.62474905.20141124164130.avi"]

    #for i in range(4):
        # assignment.getExtrinsics(paths1[i], paths2[i])    # Calibrate the extrinsics
        # assignment.backgroundModel(paths3[i])             # create background model

    path1 = ["./4persons/background/Take26.54389819.background.jpg",
             "./4persons/background/Take26.59624062.background.jpg",
             "./4persons/background/Take26.60703227.background.jpg",
             "./4persons/background/Take26.62474905.background.jpg"]
    path2 = ["./4persons/video/Take30.54389819.20141124164749.avi",
             "./4persons/video/Take30.59624062.20141124164749.avi",
             "./4persons/video/Take30.60703227.20141124164749.avi",
             "./4persons/video/Take30.62474905.20141124164749.avi"]

    # for i in range(4):
    #   assignment.backgroundSub(path1[i], path2[i], 3)
    #   videoFrame(path2[i], 0)

    # videoFrame(path2[0], 11.58)
    # videoFrame(path2[1], 0)
    # videoFrame(path2[2], 9.58)
    # videoFrame(path2[3], 18.15)

    path3 = './4persons/video/Take30.59624062.video.jpg'
    #assignment.backgroundSub2(path1, path3, 110, 180, 40)


    bg = ['./4persons/video/Take30.54389819.foreground.jpg',
          './4persons/video/Take30.59624062.foreground.jpg',
          './4persons/video/Take30.60703227.foreground.jpg',
          './4persons/video/Take30.62474905.foreground.jpg']
    # assignment.set_voxel_positions(1, 1, 1, bg, 0)

