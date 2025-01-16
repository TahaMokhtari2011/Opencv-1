import cv2 as cv
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

FM = FaceMeshDetector()
HD = HandDetector()
FD = FaceDetector()
PD = PoseDetector()
cam = cv.VideoCapture(0)
while True:
    ret, frame = cam.read()
    # Hand
    frame, hands = HD.findHands(frame, draw=True)
    # Face Mesh
    frame, Mfaces = FM.findFaceMesh(frame, draw=True)
    # Face
    frame, faces = FD.findFaces(frame, draw=True)
    # Pose
    frame = PD.findPose(frame, draw=True)
    cv.imshow('webcam', frame)
    cv.waitKey(0)
