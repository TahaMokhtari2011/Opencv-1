import cv2 as cv

image = cv.imread('image.jpg', 0)
template = cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)
result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
minval, maxval, minloc, maxloc = cv.minMaxLoc(result)
top_left = maxloc
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
cv.imshow('Image', image)
cv.waitKey(0)
cv.destroyAllWindows()
