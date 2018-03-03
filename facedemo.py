#!/usr/bin/env python

from face import Face
import common, cv2
from multitracker import MultiTracker, CorrelationTracker

# url = 'rtsp://admin:admin@192.168.0.100/live'
# url = '/home/aestaq/Videos/qb.mp4'
url = "/home/aestaq/Videos/test.avi"
# url = 1

DETECTOR = 'opencv'
removalConfig = {
    'invisible_count':35,       # Remove if no detection found on tracker in 35 countinously tracked frames
    'overlap_thresh':0.9,       # Remove if no detection found on tracker in overlap_invisible_count countinously tracked frames and trackers bbox is overlaping by 90% (intersect/union)
    'overlap_invisible_count':5,
    'corner_percentage':0.1,    # Remove if no detection found on tracker in corner_invisible_count countinously tracked frames and any portion of trackers bbox lies in 10% border oof image 
    'corner_invisible_count':5
    }

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def to_cvbox(detections):
    return [(det['box']['topleft']['x'], det['box']['topleft']['y'],
            det['box']['bottomright']['x']-det['box']['topleft']['x'], det['box']['bottomright']['y']-det['box']['topleft']['y']) for det in detections]

def to_bbox(detections):
    return [{
            "class": 'face',
            "prob": 0.99,
            "box": {
                "topleft":{'x':x, 'y':y},
                "bottomright":{'x':(x+w), 'y':(y+h)}
                },  
            } for (x,y,w,h) in detections]

def draw_faces(img, faces, names):
    """ Draws bounding boxes of objects detected on given image """
    h, w = img.shape[:2]
    for face, tid in zip(faces, names):
        # draw rectangle
        x1, y1, x2, y2 = face['box']['topleft']['x'], face['box']['topleft']['y'], face['box']['bottomright']['x'], face['box']['bottomright']['y']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        text = "%d"%(tid)
        img = common.drawLabel(img, text, (x1, y1))

    return img

def demo_video(video_file):
    import time
    facedemo = Face(detector_method=DETECTOR)
    mtracker = MultiTracker(SingleTrackerType = CorrelationTracker) #, removalConfig=removalConfig)
    # mtracker = MultiTracker(SingleTrackerType = cv2.TrackerKCF_create)   #initialise Multitracker
    # mtracker = CorrelationTracker()
    cap = common.VideoStream(video_file, queueSize=4).start()
    time.sleep(1)

    while not cap.stopped:
        t = clock()
        total_t, counter = 0, 0

        imgcv = cap.read()
        if imgcv is not None:
            counter += 1
            detections = facedemo.detect(imgcv, upsamples=0)
            temp = mtracker.update(imgcv, to_cvbox(detections))
            cvboxes, ids = [], []
            for tid,tracker in mtracker.trackers.items():
                if tracker.visible_count > 3 and tracker.consecutive_invisible_count<10:
                    cvboxes.append(tracker.bbox)
                    ids.append(tid)
            detections = to_bbox(cvboxes)
            print detections, 
            common.showImage(draw_faces(imgcv, detections, ids))

        key = cv2.waitKey(1) & 0xFF 
        if key == 27:
            break

        t1 = clock()
        dt = t1-t
        t = t1
        total_t += dt
        print counter/total_t

if __name__ == '__main__':
    demo_video(url)