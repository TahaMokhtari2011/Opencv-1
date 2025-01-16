import cv2 as cv

cap = cv.VideoCapture(0)
FG = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = FG.apply(frame)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour) > 500:  # فیلتر کردن اشیاء کوچک
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('Motion Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
