from threads import main
from Queue import Queue

#jsh2201

######## Create demo queue containing tuples of (img, timestamp)
# Read in image
img_path = './data/train/AB/03_29-18-35-09.png'
img = Image.open(img_path)

# Convert image to np.array [1x300x300x3] (1 image, 300x300 pixels, 3 color channels)
img = img.convert(mode='RGB')
X = np.asarray( img, dtype="int32" )
X = np.reshape(X, (1, 300, 300, 3))

entry0 = (X, '3_29-18-35-09')
queue = [entry0]

# Initialize queues for passing data between threads
q_in = queue.Queue()
q_out = queue.Queue()

# Keep track of threads
threads = []

# Start thread
model_path = './models/vgg_16.best.hdf5'
t = threading.Thread(target=threads.main)
t.start()
threads.append(t)

# put some images on the queue...
