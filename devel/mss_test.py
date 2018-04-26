import time

import numpy as np
import cv2
import pytesseract
from mss import mss, tools
from PIL import Image

#ayz2103
class VideoSource():
    def __init__(self):
        self.video_boundary = None
        self.player_name_boundary = None
        self.full_monitor = {'top': 0, 'left': 0, 'width': 1680, 'height': 1050}
        self.monitor = {'top': 176, 'left': 100, 'width': 640, 'height': 360}
        self.player_name_boundary = {'top': 435, 'left': 294, 'width': 300, 'height': 25}

    def set_region(self, scene_title, original_scene, reference_points, screen_x=1080, screen_y = 500):
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

    def init_screen(self):
        disp = cv2.namedWindow('game display', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('game display', 480, 300)
        cv2.moveWindow('game display', 1080, 125)

        SCALE = 1.6

        key = cv2.waitKey(1) & 0xFF

        with mss() as sct:
            # Part of screen to capture
            # Hard coded location of image
            #monitor = {'top': 176, 'left': 100, 'width': 640, 'height': 360}
            #output = 'sct-{top}x{left}_{width}x{height}.png'.format(**self.full_monitor)

            while 'Screen capturing':
                # Raw pixels as Numpy array
                img = np.array(sct.grab(self.monitor))

                # Display the picture
                cv2.imshow('game display', img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif key == ord('w'):
                    self.video_window_points = list()
                    self.set_region('input video', img, self.video_window_points, 1080, 700)
                elif key == ord('c'):
                    reference_points = self.video_window_points
                    print(reference_points)
                    self.monitor = {'top': reference_points[0][0] / SCALE,
                           'left': reference_points[0][1] / SCALE,
                           'width': (reference_points[1][0] - reference_points[0][0]) / SCALE,
                    'height': (reference_points[1][1] - reference_points[0][1]) / SCALE}
                    cv2.destroyWindow('input video')
                elif key == ord('r'):
                    self.monitor = self.full_monitor
                elif key == ord('n'):
                    self.player_reference_points = list()
                    self.set_region('player', img, self.player_reference_points, 1080, 500)
                elif key == ord('s'):
                    reference_points = self.player_reference_points
                    self.player_name_boundary = {'top': reference_points[0][0],
                           'left': reference_points[0][1],
                           'width': reference_points[1][0] - reference_points[0][0],
                           'height': reference_points[1][1] - reference_points[0][1]}
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
    vid.init_screen()
    print(vid.monitor)
    print(vid.player_name_boundary)
    #vid.process_player_name()
