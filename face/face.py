#!/usr/bin/env python

from .detect_face import *
from .face_rec import *
import dlib, os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = 'models'

def bbox_to_dlib_rect(box):
    """
    Convert a box in (left, top, right, bottom) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    x1, y1 = int(box['topleft']['x']), int(box['topleft']['y'])
    x2, y2 = int(box['bottomright']['x']), int(box['bottomright']['y'])
    return dlib.rectangle(x1, y1, x2, y2)

def size_bbox(box):
    w = box['bottomright']['x'] - box['topleft']['x']
    h = box['bottomright']['y'] - box['topleft']['y']
    return w, h, w*h


class Face(object):
    """
    A face interface class implementing different face detection, alignment and recognition algorithms.
    Arguments:
        method: name of the algorithm to identify face (default: opencv)
        model: external model path to use for certain algorithms (default: included models)
    Methods:
        detect()
            imgcv = grayscale or colored numpy image 
            kwargs = optional argumentts for some algorithms e.g. upscale=1  
    """
    def __init__(self, detector_method='opencv', detector_model=None, predictor_model='small', recognition_method='dlib', recognition_model=None):
        if detector_method == 'dlib':
            self._detector = FaceDetectorDlib()
        elif detector_method == 'cnn':
            self._detector = FaceDetectorCNN(detector_model)
        elif detector_method == 'yolo':
            self._detector = FaceDetectorYolo()
        else:
            self._detector = FaceDetectorOpenCV(detector_model)

        if predictor_model == 'large':
            pose_predictor = os.path.join(WORK_DIR, MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
        else:
            pose_predictor = os.path.join(WORK_DIR, MODEL_DIR, 'shape_predictor_5_face_landmarks.dat')
        self._predictor = dlib.shape_predictor(pose_predictor)

        self._recognizer = FaceRecDlib(recognition_model)


    def detect(self, imgcv, **kwargs):
        return self._detector.detect(imgcv, **kwargs)


    def detect_raw(self, imgcv, **kwargs):
        return self._detector.detect_raw(imgcv, **kwargs)


    def detect_largest(self, imgcv, **kwargs):
        faces = self.detect(imgcv, **kwargs)
        if len(faces) > 0:
            return max(faces, key=lambda rect: size_bbox(rect['box'])[-1])


    def get_landmarks(self, imgcv, face_locations=None):
        if face_locations is None:
            face_locations = self.detect(imgcv)
        face_locations = [bbox_to_dlib_rect(face_location['box']) for face_location in face_locations]
        # print len(face_locations)
        return [self._predictor(imgcv, face_location) for face_location in face_locations]


    def get_encodings(self, imgcv, landmarks, **kwargs):
        num_samples = kwargs.get('num_samples', 1)
        return self._recognizer.face_encodings(imgcv, landmarks, num_samples)


    def get_distance(self, face_encodings, face_to_compare):
        return self._recognizer.face_distance(face_encodings, face_to_compare)


    def compare(self, face_image1, face_image2, tolerance=0.6):
        face1 = self.detect_largest(face_image1)
        face2 = self.detect_largest(face_image2)
        results = {
                    "face1":face1,
                    "face2":face2,
                    "threshold":tolerance
                    }
        if face1 is not None and face2 is not None:
            encodings = self.get_encodings(face_image1, self.get_landmarks(face_image1, [face1]))
            encoding = self.get_encodings(face_image2, self.get_landmarks(face_image2, [face2]))
            dist = self.get_distance(encodings, encoding[0])[0]
            results["isMatched"] = dist <= tolerance
            results["matchingConfidence"] = (tolerance-dist/2)*100/tolerance if dist<2*tolerance else 0.0
        return results


if __name__ == '__main__':
    import sys, pprint

    facedemo = Face(detector_method='dlib')
    if len(sys.argv) < 3:
        print "Give two images as arguments"
    else:
        image_url1 = sys.argv[1]
        image_url2 = sys.argv[2]

    imgcv1 = cv2.imread(image_url1)
    imgcv2 = cv2.imread(image_url2)

    if imgcv1 is not None and imgcv2 is not None:
        results = facedemo.compare(imgcv1, imgcv2)
        print results
        cv2.imshow('Face1', imgcv1)
        cv2.imshow('Face2', imgcv2)
        cv2.waitKey(0)
    else:
        print("Could not read image: {}".format(image_url1))