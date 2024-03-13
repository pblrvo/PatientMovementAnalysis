import os
import cv2

protoFile   = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = os.path.join("model", "pose_iter_160000.caffemodel")

nPoints = 1
POSE_PAIRS = [
    [0]
]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

cap = cv2.VideoCapture('media/walking_sideview.mp4')

while True:
    ret, frame = cap.read()

    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    if not ret:
        break

    inWidth  = frame.shape[1]
    inHeight = frame.shape[0]

    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)


    output = net.forward()

    scaleX = inWidth  / output.shape[3]
    scaleY = inHeight / output.shape[2]

    points = []

    threshold = 0.1

    for i in range(nPoints):
        # Probability map
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold:
            # Add the point to the list if the probability is greater thanthreshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    imPoints = frame.copy()

    for i, p in enumerate(points):
        cv2.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
    
    cv2.imshow('Pose Estimation', imPoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()