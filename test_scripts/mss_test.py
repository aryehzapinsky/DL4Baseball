import time

import numpy as np
import cv2
from mss import mss, tools
from PIL import Image

class VideoSource():
    def __init__(self):
        self.video_boundary = None
        self.player_name_boundary = None

    @staticmethod
    def select_region(scene_title, original_scene, monitor, screen_x=1080, screen_y = 500):
        reference_points = list()
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

        if len(reference_points) == 2:
            key = cv2.waitKey(1) & 0xFF
            # if key == ord('r'):
            #     scene = np.array(original_scene)
            #     reference_points = list()
            if key == ord('f'):
                monitor = {'top': reference_points[0][0],
                           'left': reference_points[0][1],
                           'width': reference_points[1][0] - reference_points[0][0],
                           'height': reference_points[1][1] - reference_points[1][0]}

    def screen_cap(self):
        disp = cv2.namedWindow('game display', cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('game display', 1080, 125)
        monitor = {'top': 0, 'left': 0, 'width': 640, 'height': 640}
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
                elif key == ord('s'):
                    VideoSource.select_region('player', img, self.player_name_boundary, 1080, 500)

                # name_monitor = {'top': bounds[0][0],
                #                 'left': bounds[0][1],
                #                 'width': bounds[1][0] - bounds[0][0],
                #                 'height': bounds[1][1] - bounds[1][0]}

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
