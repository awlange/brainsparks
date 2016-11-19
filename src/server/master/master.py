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
import json
import time
import socket

# Seed random for reproducibility
n_seed = 50000
np.random.seed(n_seed)
state = np.random.get_state()

# Global-ish variables
X_train = None
Y_train = None
X_test = None
Y_test = None


def main():
    print("Master setting up particle network")

    # To ensure always same...
    np.random.set_state(state)

    phase = True
    s = 0.5
    t = None
    q = None
    b = None

    net = ParticleNetwork(cost="categorical_cross_entropy",
                          particle_input=ParticleInput(784, s=s, t=t, phase_enabled=phase))
    net.append(Particle(784, 32, activation="sigmoid", s=s, t=t, q=q, b=b, phase_enabled=phase))
    net.append(Particle(32, 10, activation="softmax", s=s, t=t, q=q, b=b, phase_enabled=phase))

    cost_acc_list = []

    mbs = 200
    nt = 4
    cs = np.min((mbs // nt, 500))
    sgd = ParticleSGD(n_epochs=1, mini_batch_size=mbs, verbosity=2, weight_update="rmsprop",
                      beta=0.95, gamma=0.99, cost_freq=5, alpha=0.01,
                      n_threads=nt, chunk_size=cs)

    n_epochs = 1
    worker_list = ["nebula0", "nebula1", "nebula2", "nebula3"]

    # Set up server
    print("Master setting up server")
    server = socket.socket()
    host = socket.gethostname()
    port = 8100
    server.bind((host, port))
    server.listen(5)

    mini_batches_per_epoch = 60000 // mbs
    opt_state = {
        "n": 0,
        "mb": 0,
        "mbpe": mini_batches_per_epoch
    }

    for n in range(n_epochs):
        opt_state[n] = n

        for mb in range(mini_batches_per_epoch):
            opt_state["mb"] = mb

            print("Epoch: {} Mini-batch: {}".format(n, mb))

            # JSON-ify the net
            net_json = net.write_to_json(None)

            # Broadcast the net for the gradient
            for nw in ["---".join((net_json, w, json.dumps(opt_state), ":::")) for w in worker_list]:
                broadcast(nw)

            # Wait for responses from workers
            gradient_data = {}
            while len(gradient_data) < len(worker_list):
                client, addr = server.accept()  # Establish connection with client.
                # print('Got connection from ', addr)
                recv_data = bytes.decode(buffer_recv(client))
                json_data, worker, _ = recv_data.split("---")
                gradient_data[worker] = json.loads(json_data)  # deserialize grads
                client.close()  # Close the connection

            # print("Received all gradients")

            # Aggregate gradient over pool
            dc_db, dc_dq, dc_dr, dc_dt, cost = aggregate_gradient(gradient_data, mbs, net)

            # Update network according to SGD method
            sgd.dc_db = dc_db
            sgd.dc_dq = dc_dq
            sgd.dc_dt = dc_dt
            sgd.dc_dr = dc_dr
            sgd.weight_update_func(net)

            # Report cost
            if cost > 0.0:
                print("Cost at epoch {} mini-batch {}: {:g}".format(n, mb, cost))

        stats = classification_stats(net.predict(X_test).argmax(axis=1), Y_test.argmax(axis=1))
        print("Test accuracy for epoch: {}".format(stats.get("total_accuracy")))


def aggregate_gradient(gradient_data, mini_batch_size, net):
    dc_db = []
    dc_dq = []
    dc_dr_x = [np.zeros(net.particle_input.rx.shape)]
    dc_dr_y = [np.zeros(net.particle_input.ry.shape)]
    dc_dr_z = [np.zeros(net.particle_input.rz.shape)]
    dc_dt = [np.zeros(net.particle_input.theta.shape)]
    for l, layer in enumerate(net.layers):
        dc_db.append(np.zeros(layer.b.shape))
        dc_dq.append(np.zeros(layer.q.shape))
        dc_dr_x.append(np.zeros(layer.rx.shape))
        dc_dr_y.append(np.zeros(layer.ry.shape))
        dc_dr_z.append(np.zeros(layer.rz.shape))
        dc_dt.append(np.zeros(layer.theta.shape))

    # Sum over all workers
    cost = 0.0
    for worker, grad in gradient_data.items():
        wk_dc_db = grad.get("dc_db")
        wk_dc_dq = grad.get("dc_dq")
        wk_dc_dr = grad.get("dc_dr")
        wk_dc_dt = grad.get("dc_dt")

        cost += grad.get("cost")

        dc_dr_x[0] += np.asarray(wk_dc_dr[0][0])
        dc_dr_y[0] += np.asarray(wk_dc_dr[1][0])
        dc_dr_z[0] += np.asarray(wk_dc_dr[2][0])
        dc_dt[0] += np.asarray(wk_dc_dt[0])
        for l, layer in enumerate(net.layers):
            dc_dr_x[l+1] += np.asarray(wk_dc_dr[0][l+1])
            dc_dr_y[l+1] += np.asarray(wk_dc_dr[1][l+1])
            dc_dr_z[l+1] += np.asarray(wk_dc_dr[2][l+1])
            dc_dt[l+1] += np.asarray(wk_dc_dt[l+1])
            dc_db[l] += np.asarray(wk_dc_db[l])
            dc_dq[l] += np.asarray(wk_dc_dq[l])

    # Divide all gradients for mean
    dc_dr_x[0] /= mini_batch_size
    dc_dr_y[0] /= mini_batch_size
    dc_dr_z[0] /= mini_batch_size
    dc_dt[0] /= mini_batch_size
    for l, layer in enumerate(net.layers):
        dc_dr_x[l + 1] /= mini_batch_size
        dc_dr_y[l + 1] /= mini_batch_size
        dc_dr_z[l + 1] /= mini_batch_size
        dc_dt[l + 1] /= mini_batch_size
        dc_db[l] /= mini_batch_size
        dc_dq[l] /= mini_batch_size

    return dc_db, dc_dq, (dc_dr_x, dc_dr_y, dc_dr_z), dc_dt, cost / 60000


def broadcast(inp):
    # Worker computes gradient, returns gradient
    net_json, worker, _, __ = inp.split("---")
    c = socket.socket()
    c.connect((worker, 8000))
    c.send(str.encode(inp))
    c.close()


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


def read_data():
    """
    Read test data on available to the master
    """
    global X_train
    global Y_train
    global X_test
    global Y_test

    # train_data_filename = "/home/pi/data/mnist/mnist_train_full.csv"
    test_data_filename = "/home/pi/data/mnist/mnist_test.csv"

    X_train = []
    Y_train = []

    ts = time.time()

    # with open(train_data_filename) as f:
    #     print("Master is reading training data...")
    #     for line in f:
    #         tmp = [int(s) for s in line.split(",")]
    #         X_train.append([float(x) / 255.0 for x in tmp[1:]])
    #         ytmp = [0.0 for _ in range(10)]
    #         ytmp[tmp[0]] = 1.0
    #         Y_train.append(ytmp)
    #
    # X_train = np.asarray(X_train)
    # Y_train = np.asarray(Y_train)

    X_test = []
    Y_test = []

    with open(test_data_filename) as f:
        print("Master is reading testing data...")
        for line in f:
            tmp = [int(s) for s in line.split(",")]
            X_test.append([float(x) / 255.0 for x in tmp[1:]])
            ytmp = [0.0 for _ in range(10)]
            ytmp[tmp[0]] = 1.0
            Y_test.append(ytmp)

    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    print("Master data preparation time: {}".format(time.time() - ts))


def classification_stats(y, yhat):
    mat = np.zeros((10, 10))
    for i, yy in enumerate(y):
        yyhat = yhat[i]
        mat[yyhat][yy] += 1

    tmp = dict()
    for val in yhat:
        tmp[val] = tmp.get(val, 0) + 1

    total_accuracy = np.trace(mat) / len(y)
    accuracy = np.zeros(10)
    for i in range(10):
        accuracy[i] = mat[i][i] / tmp.get(i)

    precision = np.zeros(10)
    recall = np.zeros(10)
    for i in range(10):
        precision[i] = mat[i][i] / sum(mat[i])
        recall[i] = mat[i][i] / sum(mat.transpose()[i])

    f1 = 2.0 / ((1.0 / precision) + 1.0 / recall)

    return {
        "total_accuracy": total_accuracy,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mat": mat
    }


if __name__ == '__main__':
    read_data()
    main()
