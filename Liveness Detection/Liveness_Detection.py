import cv2
import keras
import time
import numpy as np
import threading
import random
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from DR import DoubleReaction
from LRS import LongRepString
from LBP import LBP

turnon = True
image = 0
putWord = "wait"
putWord2 = "wait"
ear = 0
lrscounter = 0
drcounter = 0
lbpcounter = 0
randomseed = True
drseed = 3
starttime = 0

#--Load Model--#
doubleReaction = DoubleReaction()
lrs = LongRepString()
lbp = LBP()

def cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture('test_video.mp4')
    fps_time = 0
    act = ''
    global turnon, image, putWord, putWord2, ear

    while (turnon):
        success, img = cap.read()
        image = img
        cv2.rectangle(img, (0, 0), (240, 90), (204, 255, 255), -1)
        cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "LBP Status: %s" % putWord, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Reaction: %s" % putWord2, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "EAR: %.2f" % ear, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if((time.time() - starttime)<10 and starttime != 0):
            if(drseed == 0):
                act = 'RAISE YOUR LEFTHAND'
            elif(drseed == 1):
                act = 'RAISE YOUR RIGHTHAND'
            elif(drseed == 2):
                act = 'STAND UP'
            cv2.putText(img, "PLEASE " + act, (90, 230), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)            

        if((time.time() - starttime)>10 and starttime != 0):
            cv2.putText(img, "FAKE DETECTED", (200, 230), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imshow("Liveness Detection", img)
        fps_time = time.time()       

        if ((cv2.waitKey(1) == ord('q')) | (cv2.waitKey(1) == 27)):
            turnon = False
    
    cap.release()
    cv2.destroyAllWindows()

def livenessDetection(flat):
    global image, putWord, putWord2, ear, lrscounter, drcounter, lbpcounter, randomseed, drseed, starttime
    putWord, lbppr = lbp.prediction(image)
    repblink, ear = lrs.prediction(image)
    putWord2, drpr = doubleReaction.prediction(flat)

    #--To count the continuous frame that LBP result keeps being fake--#
    if(lbppr == 0):
        lbpcounter = 0
    else:
        lbpcounter += 1

    #--To count the times that LRS predict as repeated blink string--#
    if(repblink == True):
        lrscounter += 1

    #--The condition to start DR--#
    if(lrscounter >= 2 or lbpcounter >= 100):
        #--Randomly choose the action to do--#
        if(randomseed):
            drseed = random.randint(0, 2)
            print(drseed)
            starttime = time.time()
            randomseed = False        
        #--To count the frames that user do the same action as asked--#
        if(drpr == drseed):
            drcounter += 1    
        #--The condition to pass the DR and initialize the parameters--#
        if(drcounter>=10):
            lbpcounter = 0
            lrscounter = 0
            drcounter = 0
            randomseed = True
            drseed = 3
            starttime = 0

if __name__ == '__main__':
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    t = threading.Thread(target=cam)
    t.start()

    time.sleep(2)
    while(turnon):       
        humans = e.inference(image, resize_to_default=True, upsample_size=2.0)
        img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        livenessDetection(flat)

    t.join()    