import cv2 as cv
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rgray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(rgray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(rgray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
