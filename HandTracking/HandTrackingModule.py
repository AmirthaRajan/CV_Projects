import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img
        # for id, lm in enumerate(hand.landmark):
        #     h , w, c = img.shape
        #     cx, cy = int(lm.x*w), int(lm.y*h)
        #     if id == 0:
        #         cv2.circle(img, (cx,cy), 15, (0,0,255) , cv2.FILLED)


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(1)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
