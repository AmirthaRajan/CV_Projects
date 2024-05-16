import cv2

#img = cv2.imread('lena.png')
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

class_names: list

with open('coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight_path = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weight_path, config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            print(classId, confidence, box)
            if classId != 0:
                cv2.rectangle(img=img, rec=box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, class_names[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                cv2.putText(img, str(int(confidence*100))+'%', (box[0]+10, box[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
