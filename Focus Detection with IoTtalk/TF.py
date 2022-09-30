import keras
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

class TF:
    
    def __init__(self):
        self.md = keras.models.load_model('./models/Model_4type_OPFC_LSTM.h5')
        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    
    def predict(self, image):
        humans = self.e.inference(image, resize_to_default=True, upsample_size=2.0)
        img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        Predict = np.reshape(flat[0:36], (1,36,1))
        Pd = self.md.predict_classes(Predict)
        Pd = int(Pd)
        return Pd