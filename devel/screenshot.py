import numpy as np
import pyautogui
import imutils
import cv2
from time import time, sleep

#jsh2201

###########################################################
# screenshot.py
#	Simple script for iteratively taking screenshots
###########################################################

T = 3
SNAPSHOTS = 10

# Give time to open video
print('Preparing to take screensots...')
sleep(3)

for _ in range(SNAPSHOTS):
	pyautogui.screenshot("./snapshots/{}.png".format(round(time())))
	print("Taking snapshot...")
	sleep(T)
