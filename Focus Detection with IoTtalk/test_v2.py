import cv2
import time
import threading
from iottalk import DAN
from TF import TF
from MD import MD

turnon = True
image = 0
putWord = "wait"
putWord2 = "wait"

def cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fps_time = 0
    global turnon, image, putWord, putWord2

    while (turnon):
        success, img = cap.read()
        image = img
        cv2.rectangle(img, (0, 0), (240, 80), (204, 255, 255), -1)
        cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Pose: %s" % putWord, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Phone: %s" % putWord2, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Elearning", img)
        fps_time = time.time()
        if ((cv2.waitKey(1) == ord('q')) | (cv2.waitKey(1) == 27)):
            turnon = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        tf = TF()
        iot = MD()
        t = threading.Thread(target=cam)  
        sum = 0
        count = 0
        t.start()
        MD.prepare(iot)

        while(turnon):
            Predict_Result = TF.predict(tf, image)
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
        DAN.deregister()
        print('%.2f' % (sum*100/(count*4)))
    except:
        DAN.deregister()