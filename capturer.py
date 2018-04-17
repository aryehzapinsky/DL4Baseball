import time
from datetime import datetime

import pathlib
import queue
import concurrent.futures

import numpy as np
import cv2

from mss import mss, tools

import json

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

def capture_image(screen_postion, delay, samples):
    with mss() as sct:
        img = np.array(sct.grab(screen_postion))

        resized = cv2.resize(img, thumbnail_size)
        filename = '{}/{}.png'.format(base_folder, datetime.now().strftime('%m_%d-%H-%M-%S'))


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

def parser(model_path, base_folder, image_q, title_q):
    model = load_model(model_path)

    while True:
         (img, timestamp) = image_q.get()
         prediction = model.predict_classes(img)

         if prediction == 0:
             filename = '{}/{}.png'.format(base_folder, timestamp)
             cv2.imwrite(filename, img)
             title_q.put(timestamp)

def record_mapping(output, at_bat_title_q, name_title_q):

    player_name = name_title_q.get()

    while True:
        player_name = name_title_q.get() if not name_title_q.empty() else player_name

        at_bat_img_title = at_bat_title_q.get() if not



def start_pipeline(thumbnail_size, samples, base_folder):

    name_image_q = queue.Queue()
    at_bat_image_q = queue.Queue()
    name_title_q = queue.Queue()
    at_bat_title_q = queue.Queue()

    pathlib.Path('./matchups/').mkdir(exist_ok=True)
    pathlib.Path('./matchups/batters/').mkdir(exist_ok=True)
    pathlib.Path('./matchups/names/').mkdir(exist_ok=True)
    base_folder_batter = "./matchups/batters/"
    base_folder_batter = "./matchups/batters/"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        at_bat_executor = executor.submit(parser, at_bat_model, at_bat_image_q, at_bat_title_q)
        name_executor = executor.submit(parse_name, name_image_q, name_title_q)
        recorder_executor = executor.submit(record_mapping, at_bat_title_q, name_title_q)


if __name__ == "__main__":
