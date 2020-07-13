from window_cap import WindowCapture
import cv2

cv2.imshow('thing', WindowCapture('League of Legends').get_screenshot())
cv2.waitKey()