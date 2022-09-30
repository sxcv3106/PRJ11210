import cv2
import time
import threading
import random
import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from TF import TF
from MD import MD
from DR import DoubleReaction
from LRS import LongRepString
from LBP import LBP
from iottalk import DAN

#--Prepare for global variables--#
graph = tf.get_default_graph()
turnon = True
image = 0
putWord = "wait"
putWord2 = "wait"
putWord3 = "wait"
putWord4 = "wait"
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

#--open webcam--#
def cam(): 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fps_time = 0
    act = ''
    global turnon, image, putWord, putWord2, putWord3, putWord4, ear

    while (turnon):
        success, img = cap.read()
        image = img
        cv2.rectangle(img, (0, 0), (240, 130), (204, 255, 255), -1)
        cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Pose: %s" % putWord, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Phone: %s" % putWord2, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "LBP Status: %s" % putWord3, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Reaction: %s" % putWord4, (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "EAR: %.2f" % ear, (10, 120), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
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

#--define liveness Detection system--#
def livenessDetection(flat):
    global image, putWord3, putWord4, ear, lrscounter, drcounter, lbpcounter, randomseed, drseed, starttime, lbp, lrs, doubleReaction
    
    with graph.as_default():
        putWord3, lbppr = lbp.prediction(image)
    
    with graph.as_default():
        repblink, ear = lrs.prediction(image)
    
    with graph.as_default():
        putWord4, drpr = doubleReaction.prediction(flat)
        
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
            
#--open liveness Detection system--#            
def run_LD(): 
    global image, turnon
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    time.sleep(1)
    while(turnon):       
        humans = e.inference(image, resize_to_default=True, upsample_size=2.0)
        img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        livenessDetection(flat)

#--Main function--#        
if __name__ == '__main__':
    try:
        pf = TF()
        iot = MD()
        t = threading.Thread(target=cam)
        t2 = threading.Thread(target=run_LD)
        sum = 0
        count = 0
        t.start()
        MD.prepare(iot)
        t2.start()
        while(turnon):
            # if(t2.isAlive()):
                # print("\nt2 alive\n")
            # else:
                # print("\nt2 alive\n")
                
            Predict_Result = TF.predict(pf, image)
            Predict_iot = MD.predict(iot)
            
            if(Predict_Result==0):
                putWord = "Focus"
            elif(Predict_Result==1):
                putWord = "Normal"
            else:
                putWord = "Not Focus"
            
            if(Predict_iot==0):
                putWord2 = "Focus"
            elif(Predict_iot==1):
                putWord2 = "Normal"
            elif(Predict_iot==2):
                putWord2 = "Not Focus"
            else:
                putWord2 = "Search"    
                
            if (Predict_Result==0)and(Predict_iot==0):
                sum += 4
            elif (Predict_Result==0)and(Predict_iot==1):
                sum += 3
            elif (Predict_Result==1)and((Predict_iot==0)or(Predict_iot==1)):
                sum += 2
            elif (Predict_iot==3)and((Predict_Result==0)or(Predict_Result==1)):
                sum += 1
            else:
                sum = sum
            count += 1   
        t.join()
        t2.join()
        DAN.deregister()
        print('%.2f' % (sum*100/(count*4)))
    except:
        DAN.deregister()  