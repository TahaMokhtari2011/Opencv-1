import cv2 as cv

image = cv.imread('Images/Pelak_img.jpg')
image = cv.resize(image, (800, 600))  # تشخیص بهتر
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edge = cv.Canny(blur, 50, 150)
contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
rectangles = []
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    ratio = w / float(h)
    if 2 < ratio < 5 and 300 > w > 100 > h > 30:
        rectangles.append((x, y, w, h))
rectangles, weghits = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.2)
for (x, y, w, h) in rectangles:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv.imshow('image_detected', image)
cv.waitKey(0)
cv.destroyAllWindows()
