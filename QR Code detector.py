import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    qrcodes = decode(frame)
    for qr in qrcodes:
        points = qr.polygon
        if len(points) == 4:
            pt = points
        else:
            pt = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
        cv2.polylines(frame, [np.int32(pt)], isClosed=True, color=(0, 255, 0), thickness=5)
        qr_data = qr.data.decode('utf-8')
        x, y, w, h = qr.rect
        cv2.putText(frame, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('QR Code', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
