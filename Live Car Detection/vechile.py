import cv2
import numpy as np

cap = cv2.VideoCapture(r'video.mp4')
count_line = 550
mw, mh = 80, 80
alg = cv2.bgsegm.createBackgroundSubtractorMOG()

def centerhand(x, y, h, w):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

detect = []
offset = 6
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    imgsub = alg.apply(blur)
    dilat = cv2.dilate(imgsub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, count_line), (1200, count_line), (255, 127, 0), 3)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        validCount = (w >= mw) and (h >= mh)
        if not validCount:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = centerhand(x, y, h, w)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (cx, cy) in detect:
            if (count_line - offset) < cy < (count_line + offset): 
                counter += 1

                detect.remove((cx, cy))  
                print("Vehicle counter: " + str(counter))



    cv2.putText(frame,"Vechile counter:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
    
    cv2.imshow('video Original', frame)

    if cv2.waitKey(1) == 13:  
        break

cap.release()
cv2.destroyAllWindows()
