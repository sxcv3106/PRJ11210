import keras
import numpy as np

class DoubleReaction:
    def __init__(self):
        self.drmodel = keras.models.load_model('D:\\tf-pose-estimation-master\\models\\Model_4types_DR_LSTM.h5')

    def prediction(self, flat):
        putWord = ''

        #--Double Reaction Prediction--#
        Predict = np.reshape(flat[0:36], (1,36,1))
        Predict_Result = self.drmodel.predict_classes(Predict)
        Predict_Result = int(Predict_Result)

        #--Double Reaction String--#
        if(Predict_Result==0):
            putWord = "LeftHand"
        elif(Predict_Result==1):
            putWord = "RightHand"
        elif(Predict_Result==2):
            putWord = "Stand"
        else:
            putWord = "Normal"

        return putWord, Predict_Result