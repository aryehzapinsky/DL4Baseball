import time

import numpy as np
import cv2
import pytesseract
from mss import mss, tools
from PIL import Image

class VideoSource():
    def __init__(self):
        self.video_boundary = None
        self.player_name_boundary = None

    def set_player_name(self, scene_title, original_scene, monitor, screen_x=1080, screen_y = 500):
        self.player_reference_points = list()
        reference_points = self.player_reference_points
        scene = np.array(original_scene)

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                reference_points.append((x, y))

            if len(reference_points) == 2:
                cv2.rectangle(scene, reference_points[0], reference_points[1], (0, 255, 0), 2)
                cv2.imshow(scene_title, scene)

        selection_disp = cv2.namedWindow(scene_title, cv2.WINDOW_KEEPRATIO)
        # Move window
        cv2.moveWindow(scene_title, screen_x, screen_y)
        cv2.setMouseCallback(scene_title, click)
        cv2.imshow(scene_title, scene)



    def screen_cap(self):
        disp = cv2.namedWindow('game display', cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('game display', 1080, 125)
        monitor = {'top': 0, 'left': 0, 'width': 840, 'height': 540}
        name_monitor = None

        key = cv2.waitKey(1) & 0xFF

        with mss() as sct:

            # while 'Screen Initialization':
            #     img = np.array(sct.grab(monitor))
            #     cv2.imshow('game display', img)

            #     cv2.waitKey(0)

            # Part of screen to capture
            # Hard coded location of image
            monitor = {'top': 176, 'left': 100, 'width': 640, 'height': 360}
            output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)
            name_monitor = None

            while 'Screen capturing':
                # Raw pixels as Numpy array
                img = np.array(sct.grab(monitor))

                # Display the picture
                cv2.imshow('game display', img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif key == ord('w'):
                    self.set_video_window()
                elif key == ord('n'):
                    self.set_player_name('player', img, self.player_name_boundary, 1080, 500)
                elif key == ord('s'):
                    reference_points = self.player_reference_points
                    self.player_name_boundary = {'top': reference_points[0][0],
                           'left': reference_points[0][1],
                           'width': reference_points[1][0] - reference_points[0][0],
                           'height': reference_points[1][1] - reference_points[1][0]}
                    cv2.destroyWindow('player')

    def process_player_name(self):
        start = time.time()
        #hard coded image path for testing
        cv2.namedWindow('name', cv2.WINDOW_KEEPRATIO)
        image = cv2.imread("red name copy.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray,170,255,cv2.THRESH_TOZERO)[1]
        cv2.imwrite('red erode copy.png', gray)


        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        #erosion = cv2.erode(gray,kernel,iterations = 1)

        cv2.imshow('name', erosion)
        cv2.waitKey(0)

        cv2.imwrite("erosion red name copy copy.png", erosion)
        text = pytesseract.image_to_string(Image.open("erosion red name copy copy.png"))
        dur = time.time()-start
        print(dur)
        print(text)
'''
    sct_img = sct.grab(monitor)

    #img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

    tools.to_png(sct_img.rgb, sct_img.size, output=output)
    #img.save("mod"+output)
    print(output)
'''

if __name__ == "__main__":
    vid = VideoSource()
    vid.screen_cap()
    print(vid.player_name_boundary)
    #vid.process_player_name()
