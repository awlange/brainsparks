# BrainSparks imports
from brainsparks.src.calrissian.particle_network import ParticleNetwork
from brainsparks.src.calrissian.layers.particle import Particle
from brainsparks.src.calrissian.layers.particle import ParticleInput
from brainsparks.src.calrissian.optimizers.particle_sgd import ParticleSGD
from brainsparks.src.calrissian.optimizers.particle_rprop import ParticleRPROP
from brainsparks.src.calrissian.regularization.particle_regularize_l2 import ParticleRegularizeL2
from brainsparks.src.calrissian.regularization.particle_regularize_distance import ParticleRegularizeDistance
from brainsparks.src.calrissian.regularization.particle_regularize_l2plus import ParticleRegularizeL2Plus
from brainsparks.src.calrissian.regularization.particle_regularize_orthogonal import ParticleRegularizeOrthogonal


from brainsparks.src.calrissian.network import Network
from brainsparks.src.calrissian.layers.dense import Dense
from brainsparks.src.calrissian.optimizers.sgd import SGD

import numpy as np
import socket
import time
import json

# Global-ish variables
X_train = None
Y_train = None
indexes = None
shuffle_X = None
shuffle_Y = None
prev_opt_state = None


def read_data():
    """
    Read data on available to this worker
    """
    global X_train
    global Y_train
    global indexes

    train_data_filename = "/home/pi/data/mnist/mnist_train.csv"

    X_train = []
    Y_train = []

    ts = time.time()

    with open(train_data_filename) as f:
        print("Worker is reading training data...")
        for line in f:
            tmp = [int(s) for s in line.split(",")]
            X_train.append([float(x) / 255.0 for x in tmp[1:]])
            ytmp = [0.0 for _ in range(10)]
            ytmp[tmp[0]] = 1.0
            Y_train.append(ytmp)

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    indexes = np.arange(len(X_train))

    print("Worker data preparation time: {}".format(time.time() - ts))


def shuffle_data():
    global indexes
    global shuffle_X
    global shuffle_Y

    np.random.shuffle(indexes)  # in-place shuffle
    shuffle_X = np.asarray([X_train[i] for i in indexes])
    shuffle_Y = np.asarray([Y_train[i] for i in indexes])


def run():
    # Set up the server
    server = socket.socket()  # Create a socket object
    host = socket.gethostname()  # Get local machine name
    port = 8000  # Reserve a port for your service.
    server.bind((host, port))  # Bind to the port

    server.listen(5)  # Now wait for client connection.
    while True:
        c, addr = server.accept()  # Establish connection with client.
        print('Got connection from ', addr)
        data = bytes.decode(buffer_recv(c))
        c.close()  # Close the connection

        # Do something with the data
        network_json, worker, opt_state_json, _ = data.split("---")
        json_gradient = json.dumps(compute_network_gradient(network_json, opt_state_json))
        json_gradient += "---" + worker + "---:::"

        # Send the response (just an echo for now)
        d = socket.socket()
        d.connect(("nebula0", 8100))
        d.send(str.encode(json_gradient))
        d.close()


def buffer_recv(socket):
    """
    Keep receiving data into buffer until we get a triple colon (:::)
    """

    buffer = socket.recv(4096)
    buffering = True
    while buffering:
        buffer_decoded = bytes.decode(buffer)
        if ":::" in buffer_decoded:
            return buffer
        else:
            more = socket.recv(4096)
            if not more:
                buffering = False
            else:
                buffer += more
    if buffer:
        return buffer


def compute_network_gradient(network_json, opt_state_json):
    global shuffle_X
    global shuffle_Y
    global prev_opt_state

    opt_state = json.loads(opt_state_json)
    mb = opt_state.get("mb")

    # Need to determine if this is needed based on a new epoch
    if prev_opt_state is None or prev_opt_state.get("n") < opt_state.get("n"):
        shuffle_data()
    size = len(shuffle_X) // opt_state.get("mbpe")
    mini_X = shuffle_X[mb*size:(mb+1)*size]
    mini_Y = shuffle_Y[mb*size:(mb+1)*size]

    print("Computing gradient for epoch {} mini-batch {}".format(opt_state.get("n"), mb))

    net = ParticleNetwork.read_from_json(None, network_json)
    dc_db, dc_dq, dc_dr, dc_dt = net.cost_gradient(mini_X, mini_Y)

    # De-mean-ify the gradient
    scale = float(size)

    # Convert gradients to dict of lists for JSON serialization
    dc_dt[0] = list(dc_dt[0] * scale)
    dc_dr[0][0] = list(dc_dr[0][0] * scale)
    dc_dr[0][1] = list(dc_dr[0][1] * scale)
    dc_dr[0][2] = list(dc_dr[0][2] * scale)
    for l, layer in enumerate(net.layers):
        dc_db[l] = list(dc_db[l])
        dc_db[l][0] = list(dc_db[l][0] * scale)
        dc_dq[l] = list(dc_dq[l] * scale)
        dc_dt[l+1] = list(dc_dt[l+1] * scale)
        dc_dr[l+1][0] = list(dc_dr[l+1][0] * scale)
        dc_dr[l+1][1] = list(dc_dr[l+1][1] * scale)
        dc_dr[l+1][2] = list(dc_dr[l+1][2] * scale)

    gradient = dict()
    gradient["dc_db"] = dc_db
    gradient["dc_dq"] = dc_dq
    gradient["dc_dr"] = dc_dr
    gradient["dc_dt"] = dc_dt

    # Compute the cost every ten mini batches
    cost = 0.0
    if (mb % 10) == 0:
        cost = net.cost(shuffle_X, shuffle_Y) * len(shuffle_X)
    gradient["cost"] = cost

    prev_opt_state = opt_state

    return gradient

if __name__ == '__main__':
    read_data()
    run()
