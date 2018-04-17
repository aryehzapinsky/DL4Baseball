import time
from datetime import datetime

import pathlib
import queue
import concurrent.futures

import numpy as np
import cv2
import keras

from mss import mss, tools


SCREEN_POSITIONS = dict(
    monitor_top_left = {'top': 176, 'left': 100, 'width': 640, 'height': 360},
    player_name_yes = {'top': 458, 'left': 308, 'width': 100, 'height': 30},
    player_name_espn = {'top': 440, 'left': 260, 'width': 200, 'height': 35},
    player_name_nesn = {'top': 454, 'left': 304, 'width': 90, 'height': 45},
    logo_nesn = {'top': 225, 'left': 145, 'width': 35, 'height': 10},
    logo_espn = {'top': 190, 'left': 160, 'width': 37, 'height': 10}
    )

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

def collect_first_dataset():
    top_level = pathlib.Path('./data_samples/').mkdir(exist_ok=True)

    current_run = datetime.now().strftime('%H-%M-%S')
    current_dump = pathlib.Path('./data_samples/{}'.format(current_run))
    current_dump.mkdir()
    capture_frame((300,300), 7200, 1, './'+str(current_dump))

def collect_second_dataset():
    top_level = pathlib.Path('./data_samples_second/').mkdir(exist_ok=True)

    current_run = datetime.now().strftime('%H-%M-%S')
    current_dump = pathlib.Path('./data_samples_second/{}'.format(current_run))
    current_dump.mkdir()
    capture_frame((300,300), 7200, 1, './'+str(current_dump))

def capture_image(screen_postion, delay, thumbnail_size, name_q, at_bat_q):
    while True:
        with mss() as sct:
            img = np.array(sct.grab(screen_postion))

            resized = cv2.resize(img, thumbnail_size)
            timestamp = datetime.now().strftime('%m_%d-%H-%M-%S-%f')

            name_q.put((resized, timestamp))
            at_bat_q.put((resized, timestamp))

        print(timestamp)
        time.sleep(delay)

def parser(model_path, image_q, title_q):
    print("Loading model {} : {}".format(model_path, datetime.now().strftime('%H:%M:%S')))
    model = keras.models.load_model(model_path)
    print("Loaded model {} : {}".format(model_path, datetime.now().strftime('%H:%M:%S')))

    while True:
         (img, timestamp) = image_q.get()
         prediction = model.predict_classes(img)

         if prediction == 0:
             filename = '{}{}.png'.format(base_folder, timestamp)
             cv2.imwrite(filename, img)
             title_q.put(timestamp)

def record_mapping(output, at_bat_title_q, name_title_q):

    with open(output, "a") as f:
        player_name = name_title_q.get()
        while True:
            if not name_title_q.empty():
                player_name = name_title_q.get()

                if not at_bat_title_q.empty():
                    at_bat_img_title = at_bat_title_q.get()
                    f.write("{},{}\n".format(player_name, at_bat_img_title))


def start_pipeline(delay, thumbnail_size):

    name_image_q = queue.Queue()
    at_bat_image_q = queue.Queue()
    name_title_q = queue.Queue()
    at_bat_title_q = queue.Queue()

    at_bat_model = "./models/at_bat_net.hdf5"
    name_model = "./models/namenet.hdf5"

    pathlib.Path('./matchups/').mkdir(exist_ok=True)
    pathlib.Path('./matchups/batters/').mkdir(exist_ok=True)
    pathlib.Path('./matchups/names/').mkdir(exist_ok=True)
    base_folder_batter = "./matchups/batters/"
    base_folder_batter = "./matchups/batters/"

    record_mapping = "./record.csv"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        at_bat_executor = executor.submit(parser, at_bat_model, at_bat_image_q, at_bat_title_q)
        name_executor = executor.submit(parser, name_model, name_image_q, name_title_q)
        recorder_executor = executor.submit(record_mapping, at_bat_title_q, name_title_q)

        """
        image_capture_executor = executor.submit(capture_image,
                                                 SCREEN_POSITIONS.get('monitor_top_left'),
                                                 delay,
                                                 thumbnail_size,
                                                 name_image_q,
                                                 at_bat_image_q)
        """
        executor.shutdown()


if __name__ == "__main__":
    at_bat_model = "./models/at_bat_net.hdf5"
    model = keras.models.load_model(at_bat_model)
#    start_pipeline(0.4, (300, 300))
