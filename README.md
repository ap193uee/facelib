# README #

A face interface class implementing different face detection, alignment and recognition algorithms.

### Algorithms Implemented ###

* Dlib Face Detector (detector_method='dlib')
* Dlib CNN Face Detector (detector_method='cnn')
* OpenCV Face Detector (detector_method='opencv')
* Dlib Face Recognition (recognition_method='dlib')

### Requirements ###

* dlib
* opencv
* numpy
* cudnn (for gpu supoort for cnn methods)

### How to use ###

    import face, cv2
    facedemo = Face(detector_method='dlib')

    image_url1 = 'test.png'
    image_url2 = 'test2.png'
    
    imgcv1 = cv2.imread(image_url1)
    imgcv2 = cv2.imread(image_url2)

    if imgcv1 is not None and imgcv2 is not None:
        results = facedemo.compare(imgcv1, imgcv2)
        print results