import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
def landmarks(cap):
   

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:\mix\cv2\models\shape_predictor_68_face_landmarks.dat")

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            
            landmarks = predictor(gray, face)
            if len(faces) == 0:
                
                print("Masked..")
            else:
                print("No-Mask")

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (1, 120, 0), -1)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

landmarks(cap)






        
