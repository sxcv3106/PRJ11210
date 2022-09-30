import dlib
import cv2
from scipy.spatial import distance 
from imutils import face_utils
import joblib
from Rep_String_API import longestDupSubstring

#--EAR Define--#
VECTOR_SIZE = 7
EYE_AR_CONSEC_FRAMES = 3 #how many consecutive frames would be defined as blink when EAR is predicted as eye closed
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5]) #垂直距離1
    B = distance.euclidean(eye[2], eye[4]) #垂直距離2
    C = distance.euclidean(eye[0], eye[3]) #水平距離
    ear = (A + B) / (2.0 * C)
    return ear

class LongRepString:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "C:\\Users\\user\\anaconda3\\envs\\test\\Lib\\site-packages\\dlib-19.7.0.dist-info\\shape_predictor_68_face_landmarks.dat"
        )
        #--Import Model--#
        self.clf = joblib.load("D:\\tf-pose-estimation-master\\models\\Model_EAR_SVM.m")      

        self.frame_counter = 0
        self.ear_vector = []
        self.EYE_STATE = ''

    def prediction(self, image):
        repblink = False
        ear = 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            #--Get EAR--#
            shape = self.predictor(gray, rect)
            points = face_utils.shape_to_np(shape) #convert the facial landmark (x,y)-coordinates to a NumPy array
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0           

            #--Draw the eye shape on the frame--#
            #leftEyeHull = cv2.convexHull(leftEye)
            #rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

            #--Put the collected EAR into the vector and detect whether the tester is blinking--#
            ret, self.ear_vector = queue_in(self.ear_vector, ear)
            if(len(self.ear_vector) == VECTOR_SIZE):
                input_vector = []
                input_vector.append(self.ear_vector)
                res = self.clf.predict(input_vector)

                if res == 1:
                    self.EYE_STATE += '0'
                    self.frame_counter += 1
                else:
                    if self.frame_counter >= EYE_AR_CONSEC_FRAMES:
                        self.EYE_STATE += '1'
                    else:
                        self.EYE_STATE += '0'
                    self.frame_counter = 0
                
                Get_Rep_Object = longestDupSubstring(self.EYE_STATE)   
                self.EYE_STATE, repblink = Get_Rep_Object.Rep_Detect_Check()
        
        return repblink, ear
