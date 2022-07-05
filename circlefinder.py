import cv2
import numpy as np

image = cv2.imread(input("Path to image file:"))
image = (cv2.convertScaleAbs(image, alpha=6, beta=0))
output = image.copy()
img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 30, 10000)

if circles is not None:
    pass
else:
    print('No circles found.')
    exit()

x, y, r = circles[0, 0, :]
print(f'The circle is centered around ({x}, {y}) with a radius of {r} pixels')

# If some circle is found
if circles is not None:
   # Get the (x, y, r) as integers
   circles = np.round(circles[0, :]).astype("int")
   print(circles)
   # loop over the circles
   for (x, y, r) in circles:
      cv2.circle(output, (x, y), r, (0, 255, 0), 2)
# show the output image
cv2.imshow("circle",output)
cv2.waitKey(0)