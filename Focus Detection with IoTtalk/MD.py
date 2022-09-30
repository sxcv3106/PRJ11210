import keras
import numpy as np
from iottalk import DAI, DAN

class MD:
    
    def __init__(self):
        self.md = keras.models.load_model('./models/Model_4types_IOT_LSTM.h5')
        self.iotin = []
        self.input2 = []
        DAI.register_to_iottalk()
    
    def prepare(self):
        while True:
            if (len(self.iotin) < 30):
                self.iotin = DAI.dai(self.iotin)
            else:
                self.input2 = np.array(self.iotin)
                self.input2 = self.input2.reshape(1,30,1)
                break
    
    def predict(self):
        Pd = self.md.predict_classes(self.input2)
        Pd = int(Pd)
        del self.iotin[0]
        del self.iotin[1]
        del self.iotin[2]
        self.iotin = DAI.dai(self.iotin)
        self.input2 = np.array(self.iotin)
        self.input2 = self.input2.reshape(1,30,1)
        return Pd