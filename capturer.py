import time
from datetime import datetime

import pathlib

import numpy as np
import cv2

from mss import mss, tools

def capture_frame(thumbnail_size, samples, delay, base_folder):
    monitor = {'top': 176, 'left': 100, 'width': 640, 'height': 360}

    with mss() as sct:
        for _ in range(samples):
            img = np.array(sct.grab(monitor))

            resized = cv2.resize(img, thumbnail_size)
            filename = '{}/{}.png'.format(base_folder, datetime.now().strftime('%m_%d-%H-%M-%S'))
            #print(filename)
            cv2.imwrite(filename, resized)

            time.sleep(delay)


if __name__ == "__main__":
    top_level = pathlib.Path('./data_samples/').mkdir(exist_ok=True)

    current_run = datetime.now().strftime('%H-%M-%S')
    current_dump = pathlib.Path('./data_samples/{}'.format(current_run))
    current_dump.mkdir()
    capture_frame((300,300), 7200, 1, './'+str(current_dump))
