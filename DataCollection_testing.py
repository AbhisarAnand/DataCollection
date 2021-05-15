import cv2
import numpy
import time

vs = cv2.VideoCapture(0)

i=0
while (vs.isOpened()):
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=450)
    
    cv2.imshow("Frame", frame)
    if ret == False:
        break
    if i !=500:
        a = cv2.imwrite('Training_Image'+str(i)+'.jpg',frame)
        i+=1
    else:
        break
    

vs.release()
cv2.destroyAllWindows()
