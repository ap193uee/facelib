# README #

A face detector interface class implementing different face detection algorithms.

### Algorithms Implemented ###

* Dlib Face Detector (method='dlib')
* Dlib CNN Face Detector (method='cnn')
* OpenCV Face Detector (method='opencv')
* Neural Network

### Requirements ###

* dlib
* opencv

### How to use ###

    import face, cv2
    detector = face.FaceDetector(method='dlib')
    image_url = 'test.png'
    imgcv = cv2.imread(image_url)
    if imgcv is not None:
        results = detector.run(imgcv)
        print results