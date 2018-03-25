import time

import numpy as np
import cv2
from mss import mss, tools
from PIL import Image

def select_region(scene_title, scene, screen_x=1080, screen_y = 500):
    reference_points = list()
    cropping = False
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            reference_points.append([(x, y)])
            cropping = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            reference_points.append([(x, y)])
            cropping = False

        cv2.rectangle(scene, reference_points[0], reference_points[1], (0, 255, 0), 2)

    selection_disp = cv2.namedWindow(scene_title, cv2.WINDOW_KEEPRATIO)
    # Move window
    cv2.moveWindow(scene_title, screen_x, screen_y)
    cv2.setMouseCallback(scene_title, click)
    cv2.imshow(scene_title, scene)

    return selection_disp


def screen_cap():
    disp = cv2.namedWindow('game display', cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow('game display', 1080, 125)

    with mss() as sct:
        # Part of screen to capture
        monitor = {'top': 176, 'left': 100, 'width': 640, 'height': 360}
        output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)

        while 'Screen capturing':
            last_time = time.time()

            # Raw pixels as Numpy array
            img = np.array(sct.grab(monitor))

            # Display the picture
            cv2.imshow('game display', img)

            #print('fps: {0}'.format(1 / (time.time()-last_time)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                name_selection = select_region('player', img, 1080, 500)



'''
    sct_img = sct.grab(monitor)

    #img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

    tools.to_png(sct_img.rgb, sct_img.size, output=output)
    #img.save("mod"+output)
    print(output)
'''

if __name__ == "__main__":
    screen_cap()
