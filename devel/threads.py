from keras.models import load_model
from PIL import Image
import numpy as np

#jsh2201

############################################################################################
#  Jonathan Herman jsh2201
#  4/12/18
#
#  threads.py
#      This script will get spun up twice by the main thread: once for the Name ConvNet
#	(which passes through only frames that contain batters' names) and once for the
#	AB ConvNet (which passes through only images of batters at bat.
#
#      Input: (img, timestamp) tuples queued by main thread
#      Output: (img, timestamp) tuples written to output queue (to be processed in
#		main thread)
#
############################################################################################

def main(model_path='./models/vgg_16.best.hdf5'):
    print('\n\nLoading model...\n\n')
    model = load_model(model_path)
    print('\n'*2, 'Loaded model...', '\n'*2)

    while(1):
        # global queue
        if len(queue) > 0:
        # if not q_in.empty:
            (img, timestamp) = queue[0]
            # (img, timestamp) = q.get()
            prediction = model.predict_classes(img)
            # q_out.put(prediction) # save to output queue
            queue = []
            print(prediction)
            q_in.task_done()
        else:
            print('Queue is empty... exiting')
            break # take this out

if __name__ == '__main__':
	model_path = './models/vgg_16.best.hdf5'
	main(model_path)
