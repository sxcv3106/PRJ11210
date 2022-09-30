import cv2
import keras
import numpy as np

def olbp(src):
    dst = np.zeros(src.shape,dtype=src.dtype)
    for i in range(1,src.shape[0]-1):
        for j in range(1,src.shape[1]-1):
            pass
            center = src[i][j]
            code = 0;
            code |= (src[i-1][j-1] >= center) << 7;
            code |= (src[i-1][j  ] >= center) << 6;
            code |= (src[i-1][j+1] >= center) << 5;
            code |= (src[i  ][j+1] >= center) << 4;
            code |= (src[i+1][j+1] >= center) << 3;
            code |= (src[i+1][j  ] >= center) << 2;  
            code |= (src[i+1][j-1] >= center) << 1; 
            code |= (src[i  ][j-1] >= center) << 0;             
            dst[i-1][j-1]= code;
    return dst

class LBP:
    def __init__(self):
        self.lbpmodel = keras.models.load_model('D:\\tf-pose-estimation-master\\models\\Model_3types_LBP_CNN.h5')

    def prediction(self, image):
        putWord = ''

        #--Image Reshape and LBP Prediction--#
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray,(64,48))
        image = olbp(gray)
        image = image.reshape(1, 48, 64, 1)
        Predict_Result = self.lbpmodel.predict_classes(image)
        Predict_Result = int(Predict_Result)

        #--LBP String--#
        if (Predict_Result == 0):
            putWord = "True"
        elif (Predict_Result == 1):
            putWord = "Phone"
        else:
            putWord = "Paper"

        return putWord, Predict_Result