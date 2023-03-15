import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import assignment

#tracing position and color
trackingP = []
trackingC = []

paths_ex = ["./4persons/extrinsics/Take25.54389819.config.xml",
          "./4persons/extrinsics/Take25.59624062.config.xml",
          "./4persons/extrinsics/Take25.60703227.config.xml",
          "./4persons/extrinsics/Take25.62474905.config.xml"]
path_video = ["./4persons/video/Take30.54389819.20141124164749.avi",
         "./4persons/video/Take30.59624062.20141124164749.avi",
         "./4persons/video/Take30.60703227.20141124164749.avi",
         "./4persons/video/Take30.62474905.20141124164749.avi"]

# update voxel position
def update(pressNum, trainedGMMs):
    global trackingP, trackingR
    try:
        coord, label, center = loadCoord(pressNum)
        label = np.squeeze(label)
        coord = np.array(coord)
        Cood0 = coord[label == 0]
        Cood1 = coord[label == 1]
        Cood2 = coord[label == 2]
        Cood3 = coord[label == 3]
        C3D = [Cood0, Cood1, Cood2, Cood3]

        predicted_label_4cam = []
        predicted_likelihoods_4cam = []
        for i in range(4):
            camera_handles = cv2.VideoCapture(path_video[i])
            fn = 0
            while True:
                ret, image = camera_handles.read()
                if fn == pressNum:
                    img = image
                    # cv2.imshow('foreground', image)
                    # cv2.waitKey(2000)
                    break
                fn += 1

            fsEx = cv2.FileStorage(paths_ex[i], cv2.FILE_STORAGE_READ)
            mtx = fsEx.getNode('mtx').mat()
            dist = fsEx.getNode('dist').mat()
            rvec = fsEx.getNode('rvec').mat()
            tvec = fsEx.getNode('tvec').mat()

            pts, jac = cv2.projectPoints(np.float32(coord), rvec, tvec, mtx, dist)
            pts = np.int32(pts)
            pixels = []
            for j in range(len(pts)):
                pixels.append(img[pts[j][0][1]][pts[j][0][0]].tolist())
                cv2.circle(img, tuple([pts[j][0][0], pts[j][0][1]]), 2, img[pts[j][0][1]][pts[j][0][0]].tolist(), -1)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            pixels = np.array(pixels)
            C0 = pixels[label == 0]
            C1 = pixels[label == 1]
            C2 = pixels[label == 2]
            C3 = pixels[label == 3]
            C2D = [C0, C1, C2, C3]

            predicted_label = []
            predicted_likelihoods = []
            for n in range(4):
                likelihoods = [gmm.score(C2D[n]) for gmm in trainedGMMs[1]]
                predicted_label.append(likelihoods.index(max(likelihoods)))
                predicted_likelihoods.append(likelihoods)
            predicted_label_4cam.append(predicted_label)
            #predicted_likelihoods_4cam.append(predicted_likelihoods)

        predicted_label_4cam = np.array(predicted_label_4cam)
        #print(np.array(predicted_label_4cam))

        final_label = []
        for row in predicted_label_4cam:
            if len(row) == len(set(row)):
                final_label = row
                break
        if final_label == []:
            counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(predicted_label_4cam))),
                                         axis=1, arr=predicted_label_4cam)
            # Find the row with the least number of repeated items
            x = np.argmin(np.sum(counts > 1, axis=1))
            result = predicted_label_4cam[x]
            final_label = result

        #print(final_label)

        color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 165, 0]]  # red 0, green 1, blue 2, yellow 3
        colors = []
        for j in range(4):
            for C in C3D[j]:
                colors.append(color[final_label[j]])
            trackingC.append(color[final_label[j]])
            trackingP.append(np.int32(center[j]))

        position = []
        for C in C3D:
            for item in C:
                position.append(item)

        Rx = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0, -1, 0]])
        positions = [Rx.dot(p) for p in position]
        positions = [np.multiply(DR, 5) for DR in positions]

    except:
        pass

    return positions, colors

# draw path, after press G
def draw():
    global trackingP, trackingC
    imgTracking = np.zeros((720, 720, 3), np.uint8)
    imgTracking.fill(192)
    for i in range(len(trackingP)):
        cv2.circle(imgTracking, trackingP[i]*15 + 250, 4, trackingC[i], -1)
        cv2.imshow("Tracking", imgTracking)
        cv2.waitKey(20)

# train GMMs
def trainGMM(pressNum):
    global paths_ex, path_video
    path = f'./4persons/video/voxelCoords{pressNum}.xml'
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    coord = fs.getNode('coord').mat()
    label = fs.getNode('label').mat()

    label = np.squeeze(label)
    C0 = coord[label == 0]
    C1 = coord[label == 1]
    C2 = coord[label == 2]
    C3 = coord[label == 3]
    C3D = [C0, C1, C2, C3]

    trainedGMMs = []
    for i in range(4):
        fsEx = cv2.FileStorage(paths_ex[i], cv2.FILE_STORAGE_READ)
        mtx = fsEx.getNode('mtx').mat()
        dist = fsEx.getNode('dist').mat()
        rvec = fsEx.getNode('rvec').mat()
        tvec = fsEx.getNode('tvec').mat()

        camera_handles = cv2.VideoCapture(path_video[i])
        fn = 0
        while True:
            ret, image = camera_handles.read()
            if fn == pressNum:
                img = image
                # cv2.imshow('foreground', image)
                # cv2.waitKey(2000)
                break
            fn += 1

        RGBdata = [[] for _ in range(4)]
        for person in range(4):
            pts, jac = cv2.projectPoints(np.float32(C3D[person]), rvec, tvec, mtx, dist)
            pts = np.int32(pts)
            pixels = []
            for j in range(len(pts)):
                pixels.append(img[pts[j][0][1]][pts[j][0][0]].tolist())
                cv2.circle(img, tuple([pts[j][0][0], pts[j][0][1]]), 2, img[pts[j][0][1]][pts[j][0][0]].tolist(), -1)
            RGBdata[person] = pixels
            # cv2.imshow('img', img)
            # cv2.waitKey(1000)

        # return colorModel(RGBdata, frame)     # this no longer use

        # create the GMM models
        components = 3
        gmm0 = GaussianMixture(components)
        gmm1 = GaussianMixture(components)
        gmm2 = GaussianMixture(components)
        gmm3 = GaussianMixture(components)

        # fit the GMM models to the training data
        gmm0.fit(RGBdata[0])
        gmm1.fit(RGBdata[1])
        gmm2.fit(RGBdata[2])
        gmm3.fit(RGBdata[3])

        # store the trained GMM models
        trainedGMMs.append([gmm0, gmm1, gmm2, gmm3])
    return trainedGMMs

# load coord from storage
def loadCoord(pressNum):
    newfile = f'./4persons/video/voxelCoords{pressNum}.xml'
    #print(newfile)
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_READ)
    coord = fs.getNode('coord').mat()
    label = fs.getNode('label').mat()
    center = fs.getNode('center').mat()
    return coord, label, center

# save label and center to storage
def saveLabel(labels, centers, pressNum):
    newfile = f'./4persons/video/voxelCoords{pressNum}.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_APPEND)
    fs.write("label", labels)
    fs.write("center", centers)
    fs.release()

# knn cluster
def knn(pressNum):
    global paths_ex, path_video
    coordOrigin, _, _ = loadCoord(pressNum)
    coord = coordOrigin[:, [0, 1]]    # Remove the height of the voxel
    numClusters = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(coord), numClusters, None, criteria, 10, flags)

    saveLabel(labels, centers, pressNum)

    ##### testing project result
    # labels = np.squeeze(labels)
    # C0 = coordOrigin[labels == 0]
    # C1 = coordOrigin[labels == 1]
    # C2 = coordOrigin[labels == 2]
    # C3 = coordOrigin[labels == 3]
    # C3D = [C0, C1, C2, C3]
    #
    # ### test cluster result:
    # # colors = np.zeros((len(labels), 3))
    # # colors[labels == 0] = [255, 0, 0]   # red for label 0
    # # colors[labels == 1] = [0, 255, 0]   # green for label 1
    # # colors[labels == 2] = [0, 0, 255]   # blue for label 2
    # # colors[labels == 3] = [255, 255, 0]  # yellow for label 3
    #
    # RGBdata = []
    # for i in range(4):
    #     fs = cv2.FileStorage(paths_ex[i], cv2.FILE_STORAGE_READ)
    #     mtx = fs.getNode('mtx').mat()
    #     dist = fs.getNode('dist').mat()
    #     rvec = fs.getNode('rvec').mat()
    #     tvec = fs.getNode('tvec').mat()
    #
    #     camera_handles = cv2.VideoCapture(path_video[i])
    #     fn = 0
    #     while True:
    #         ret, image = camera_handles.read()
    #         if fn == pressNum:
    #             img = image
    #             break
    #         fn += 1
    #
    #     ### test cluster result:
    #     # pts, jac = cv2.projectPoints(np.float32(coordOrigin), rvec, tvec, mtx, dist)
    #     # pts = np.int32(pts)
    #     # for j in range(len(pts)):
    #     #     cv2.circle(img, tuple([pts[j][0][0], pts[j][0][1]]), 2, colors[j], -1)
    #     # cv2.imshow('img', img)
    #     # cv2.waitKey(0)
    #     for person in range(4):
    #         pts, jac = cv2.projectPoints(np.float32(C3D[person]), rvec, tvec, mtx, dist)
    #         pts = np.int32(pts)
    #         pixels = []
    #         for j in range(len(pts)):
    #             pixels.append(img[pts[j][0][1]][pts[j][0][0]].tolist())
    #             cv2.circle(img, tuple([pts[j][0][0], pts[j][0][1]]), 2, img[pts[j][0][1]][pts[j][0][0]].tolist(), -1)
    #         RGBdata.append(pixels)
    #         # cv2.imshow('img', img)
    #         # cv2.waitKey(1000)

    # color(RGBdata, pressNum)  # this no longer use
    cv2.destroyAllWindows()

"""backup, tries didn't work"""
#tring1:
    # 1.1 Euclidean distance of meanValue, no good results
    # GMMs = p3.trainGMM(pressNum)
    # for i in range(len(GMMs)):
    #     distances = [np.linalg.norm(GMMs[i] - mean) for mean in
    #                  [trainedGMMs[0], trainedGMMs[1], trainedGMMs[2], trainedGMMs[3]]]
    #     closest_mean_dis = np.argmin(distances)
    #     print("New data", GMMs[i], "is closest to mean value", closest_mean_dis)
    #
    # 1.2 predict each pixel, no good results
    # personLabel = []
    # for i in range(4):
    #     personLabel = []
    #     for C in C2D[i]:
    #         likelihoods = np.zeros(4)
    #         likelihoods[0] = trainedGMMs[0].predict_proba(C.reshape(1, -1))[0, 1]  # likelihood of belonging to person 0
    #         likelihoods[1] = trainedGMMs[1].predict_proba(C.reshape(1, -1))[0, 1]  # likelihood of belonging to person 1
    #         likelihoods[2] = trainedGMMs[2].predict_proba(C.reshape(1, -1))[0, 1]  # likelihood of belonging to person 2
    #         likelihoods[3] = trainedGMMs[3].predict_proba(C.reshape(1, -1))[0, 1]  # likelihood of belonging to person 3
    #         #personLabel.append(np.argmax(likelihoods))    # the person with the highest likelihood for this pixel value
    #         personLabel.append(likelihoods)
    #     row_means = np.mean(np.array(personLabel), axis=0)
    #     # mean_value = sum(personLabel) / len(personLabel)  # Calculate mean value
    #     # rounded_mean = round(mean_value)  # Round mean value to nearest integer
    #     # closest_number = min([0, 1, 2, 3], key=lambda x: abs(x - rounded_mean))
    #     print(row_means)

#tring2: try to predict each pixel, four camera
        #
        # personLabel0 = [[[] for i in range(len(Cood0))] for j in range(4)]
        # personLabel1 = [[[] for i in range(len(Cood1))] for j in range(4)]
        # personLabel2 = [[[] for i in range(len(Cood2))] for j in range(4)]
        # personLabel3 = [[[] for i in range(len(Cood3))] for j in range(4)]
        # print(len(personLabel0[0]), len(personLabel1[0]), len(personLabel2[0]), len(personLabel3[0]))
        #
        # for cluster in range(4):
        #     for C in range(len(C2D[cluster])):
        #         likelihoods = np.zeros(4)
        #         likelihoods[0] = trainedGMMs[i][0].predict_proba(C2D[cluster][C].reshape(1, -1))[0, 1]  # likelihood of belonging to person 0
        #         likelihoods[1] = trainedGMMs[i][1].predict_proba(C2D[cluster][C].reshape(1, -1))[0, 1]  # likelihood of belonging to person 1
        #         likelihoods[2] = trainedGMMs[i][2].predict_proba(C2D[cluster][C].reshape(1, -1))[0, 1]  # likelihood of belonging to person 2
        #         likelihoods[3] = trainedGMMs[i][3].predict_proba(C2D[cluster][C].reshape(1, -1))[0, 1]  # likelihood of belonging to person 3
        #         if cluster == 0:
        #             personLabel0[i][C] = likelihoods
        #         elif cluster == 1:
        #             personLabel1[i][C] = likelihoods
        #         elif cluster == 2:
        #             personLabel2[i][C] = likelihoods
        #         elif cluster == 3:
        #             personLabel3[i][C] = likelihoods
        # personLabel0 = np.array(personLabel0)
        # personLabel1 = np.array(personLabel1)
        # personLabel2 = np.array(personLabel2)
        # personLabel3 = np.array(personLabel3)
        #
        # final_label = []
        # result0 = np.zeros((len(personLabel0[0]), 4))  # create a len(pixels)x4 array
        # for a in range(len(personLabel0[0])):
        #     for b in range(4):
        #         for c in range(4):
        #                 result0[a, b] += personLabel0[c, a, b]  # add the value at the current index
        # r0 =  np.sum(np.array(result0), axis=0)
        # final_label.append(np.argmax(r0))
        # print(np.argmax(r0))
        # result1 = np.zeros((len(personLabel1[0]), 4))  # create a len(pixels)x4 array
        # for a in range(len(personLabel1[0])):
        #     for b in range(4):
        #         for c in range(4):
        #             result1[a, b] += personLabel1[c, a, b]  # add the value at the current index
        # r1 = np.sum(np.array(result1), axis=0)
        # final_label.append(np.argmax(r1))
        # print(np.argmax(r1))
        # result2 = np.zeros((len(personLabel2[0]), 4))  # create a len(pixels)x4 array
        # for a in range(len(personLabel2[0])):
        #     for b in range(4):
        #         for c in range(4):
        #             result2[a, b] += personLabel2[c, a, b]  # add the value at the current index
        # r2 = np.sum(np.array(result2), axis=0)
        # final_label.append(np.argmax(r2))
        # print(np.argmax(r2))
        # result3 = np.zeros((len(personLabel3[0]), 4))  # create a len(pixels)x4 array
        # for a in range(len(personLabel3[0])):
        #     for b in range(4):
        #         for c in range(4):
        #             result3[a, b] += personLabel3[c, a, b]  # add the value at the current index
        # r3 = np.sum(np.array(result3), axis=0)
        # final_label.append(np.argmax(r3))
        # print(final_label)

#trying3: average likelihood for each camera
# if final_label == []:
#     predicted_likelihoods_4cam = np.array(predicted_likelihoods_4cam)
#     result = np.zeros((4, 4))  # create a 4x4 array to hold the results
#     for a in range(4):
#         for b in range(4):
#             for c in range(4):
#                 result[a, b] += predicted_likelihoods_4cam[c, a, b]  # add the value at the current index
#     label = []
#     for n in range(4):
#         resultList = result[n].tolist()
#         label.append(resultList.index(max(resultList)))
#     final_label = label

"""this no longer use"""
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

"""this no longer use"""
def loadTable():
    newfile = './4persons/video/lookupTable.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_READ)
    flag = fs.getNode('flag').mat()
    coord = fs.getNode('coord').mat()

"""this no longer use"""
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

"""this no longer use"""
def colorModel(coord, frame):
    newfile = f'./4persons/video/colorModel{frame}.xml'
    fs = cv2.FileStorage(newfile, cv2.FILE_STORAGE_WRITE)
    means = []
    for i in range(4) :
        gmm = cv2.ml.EM_create()
        gmm.setClustersNumber(3)
        gmm.trainEM(np.array(coord[i]))
        means.append(gmm.getMeans())
        # means = gmm.getMeans()
        # covs = gmm.getCovs()
        # weights = gmm.getWeights()
        #
        fs.write(f"means{i}", np.array(means))
        # fs.write(f"covs{i}", np.array(covs))
        # fs.write(f"weights{i}", np.array(weights))

    fs.release()
    return means


if __name__ == "__main__":
    paths1 = ["./4persons/extrinsics/Take25.54389819.20141124164119.avi",
             "./4persons/extrinsics/Take25.59624062.20141124164119.avi",
             "./4persons/extrinsics/Take25.60703227.20141124164119.avi",
             "./4persons/extrinsics/Take25.62474905.20141124164119.avi"]
    paths2 = ["./4persons/extrinsics/Take25.54389819.imageCorners.xml",
              "./4persons/extrinsics/Take25.59624062.imageCorners.xml",
              "./4persons/extrinsics/Take25.60703227.imageCorners.xml",
              "./4persons/extrinsics/Take25.62474905.imageCorners.xml"]

    """the following are testing"""
    #for i in range(4):
        # assignment.getExtrinsics(paths1[i], paths2[i])    # Calibrate the extrinsics

    # assignment.set_voxel_positions(1, 1, 1, 200)
    # knn(140)
    trainedGMMs = trainGMM(0)
    update(300,trainedGMMs)
    NUMBERS = [140,150,160,170,180,190,
                   200,210,220,260,270,280,290,
                   300,310,320,330,340,350,360,360,380,390,
                   400,410,420,430,440,450,490,
                   500,510,530,530,540,550,560,570,580,590,
                   610,620,630,640,650,660,670,680,690,
                   700,710,720,740,750,780,790,
                   800,810,820,830,840,850,860,870,880,890,
                   900,910,920,930,940,950,960,970,990,
                   1000]

    # for n in NUMBERS:
    #     update(n,trainedGMMs)

