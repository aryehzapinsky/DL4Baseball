#!/usr/bin/env python3

import time
from datetime import datetime

import pathlib
import queue
import concurrent.futures
import threading

import numpy as np
import cv2
import keras
from keras import backend as K
import tensorflow as tf

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
    # Delayed start up
    time.sleep(5)

    while True:
        with mss() as sct:
            img = np.array(sct.grab(screen_postion))

            resized = cv2.resize(img, thumbnail_size)
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
            resized = np.reshape(resized, (1, 300, 300, 3))
            timestamp = datetime.now().strftime('%m_%d-%H-%M-%S-%f')

            name_q.put((resized, timestamp))
            at_bat_q.put((resized, timestamp))

        print("capture image thread {}- name_q: {}".format(timestamp, name_q.qsize()))
        time.sleep(delay)

def parser(model_path, base_folder, image_q, title_q):
    print("Loading model {} : {}".format(model_path, datetime.now().strftime('%H:%M:%S')))
    graph = tf.Graph()
    with graph.as_default():
        session = tf.Session()
        with session.as_default():
            model = keras.models.load_model(model_path)
            print("Loaded model {} : {}".format(model_path, datetime.now().strftime('%H:%M:%S')))

            while True:
                print("Model Path {} - Q Size{}".format(model_path, image_q.qsize()))
                (img, timestamp) = image_q.get()
                print("Setting up prediction || {} - {}".format(model_path, timestamp))
                prediction = model.predict_classes(img)[0][0]
                print("Predicted || {} - {} ~~~~~~~~~~ Predicted class: {}".format(model_path, timestamp, prediction))

                if prediction == 0:
                    filename = '{}{}.png'.format(base_folder, timestamp)
                    print("FILENAME: {}<<<<<<<<<".format(filename))
                    cv2.imwrite(filename, img[0])
                    title_q.put(timestamp)

                print("*******************")

def record_mapping(output, at_bat_title_q, name_title_q):

    print("Opening recorder file")

    with open(output, "a+") as f:
        while True:
            if not name_title_q.empty():
                player_name = name_title_q.get()

                if not at_bat_title_q.empty():
                    at_bat_img_title = at_bat_title_q.get()
                    f.write("{},{}\n".format(player_name, at_bat_img_title))
                    print("~~~~~~~~~saved~~~~~~~~~~~")


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
    base_folder_names = "./matchups/names/"

    record_file = "./record.csv"

    at_bat_thread = threading.Thread(target=parser, name="at_bat",
                                     args=(at_bat_model, base_folder_batter,
                                           at_bat_image_q, at_bat_title_q))
    name_thread = threading.Thread(target=parser, name="player_name",
                                   args=(name_model, base_folder_names,
                                         name_image_q, name_title_q))
    recorder_thread = threading.Thread(target=record_mapping, name="recorder",
                                       args=(record_file, at_bat_title_q, name_title_q))
    capture_thread = threading.Thread(target=capture_image, name="capture",
                                      args=(SCREEN_POSITIONS.get('monitor_top_left'), delay, thumbnail_size, name_image_q, at_bat_image_q))

    at_bat_thread.start()
    time.sleep(5)
    name_thread.start()
    time.sleep(5)
    capture_thread.start()
    recorder_thread.start()


if __name__ == "__main__":
    start_pipeline(4, (300, 300))
