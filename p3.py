import cv2
import assignment

def videoFrame(path, frameNum):
    cap = cv2.VideoCapture(path)
    frameIndex = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameIndex == frameNum:
            newfile = path[:-18] + 'video.jpg'
            cv2.imwrite(newfile, frame)
        frameIndex += 1
    cap.release()
    cv2.destroyAllWindows()

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

    # assignment.backgroundSub(path1[0], path2[0], 11.58)
    # assignment.backgroundSub(path1[1], path2[1], 0)
    # assignment.backgroundSub(path1[2], path2[2], 9.58)
    # assignment.backgroundSub(path1[3], path2[3], 18.15)
    # for i in range(4):
    #     assignment.backgroundSub(path1[i], path2[i], 0)

    # videoFrame(path2,0)
    path3 = './4persons/video/Take30.59624062.video.jpg'

    #assignment.backgroundSub2(path1, path3, 110, 180, 40)