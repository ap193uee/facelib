#!/usr/bin/env python

import cv2

class FaceDetector(object):
    """
    A face detector interface class implementing different face detection algorithms.
    """
    def __init__(self, method='dlib'):
        if method == 'dlib':
            self._detector = FaceDetectorDlib()

    def run(self, imgcv):
        return self._detector.run(imgcv)

class FaceDetectorDlib(object):
    """
    A face detector based on dlib library.
    """
    def __init__(self):
        import dlib
        self._detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor(model_name)

    def run(self, imgcv):
        gray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, 1)
        return self._format_result(faces)

    # Format the results
    def _format_result(self, result):
        out_list = []
        for res in result:
            formatted_res = dict()
            formatted_res["class"] = 'face'
            formatted_res["prob"] = 0.99
            formatted_res["box"] = {
                                    "topleft":{'x':res.left(),'y':res.top()},
                                    "bottomright":{'x':res.right(),'y':res.bottom()}
                                    }
            out_list.append(formatted_res)
        return out_list

if __name__ == '__main__':
    print FaceDetector.__doc__
    import sys, pprint

    detector = FaceDetector()
    image_url = 'test.png' if len(sys.argv) < 2 else sys.argv[1]
    imgcv = cv2.imread(image_url)
    if imgcv is not None:
        results = detector.run(imgcv)
        pprint.pprint(results)
    else:
        print("Could not read image: {}".format(image_url))