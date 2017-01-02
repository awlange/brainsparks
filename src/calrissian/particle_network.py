from .cost import Cost

from .layers.particle import Particle
from .layers.particle import ParticleInput
from .regularization.particle_regularize_l2 import ParticleRegularizeL2
from .regularization.particle_regularize_l2plus import ParticleRegularizeL2Plus
from .regularization.particle_regularize_orthogonal import ParticleRegularizeOrthogonal

import numpy as np
import json


class ParticleNetwork(object):

    def __init__(self, particle_input=None, cost="mse", regularizer=None):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer
        self.particle_input = particle_input

    def append(self, layer):
        """
        Appends a layer to the network

        :param layer:
        :return:
        """
        self.layers.append(layer)

    def build(self):
        """
        Handle networks layer dimensions checks, other possible initializations

        Release build lock

        :return:
        """
        # TODO

        self.lock_built = True

    def predict_single(self, data_X):
        """
        Same as predict, but only one sample
        """
        return self.predict(data_X.reshape((1, len(data_X))))

    def predict(self, data_X):
        """
        Pass given input through network to compute the output prediction

        :param data_X:
        :return:
        """
        a, r = self.particle_input.feed_forward(data_X)
        for layer in self.layers:
            a, r = layer.feed_forward(a, r)
        return a

    def feed_to_layer(self, data_X, end_layer=0):
        """
        Feed data forward until given end layer. Return the resulting activation

        :param data_X: input data
        :param end_layer: the index of the ending layer
        :return: resulting activation at end layer
        """
        if len(self.layers) <= end_layer < 0:
            return None

        a, r = self.particle_input.feed_forward(data_X)
        for l, layer in enumerate(self.layers):
            a, r = layer.feed_forward(a, r)
            if l == end_layer:
                return a

        return None

    def cost(self, data_X, data_Y):
        """
        Compute the cost for all input data corresponding to expected output

        :param data_X:
        :param data_Y:
        :return:
        """
        c = self.cost_function(data_Y, self.predict(data_X))

        if self.regularizer is not None:
            c += self.regularizer.cost(self.particle_input, self.layers)

        return c

    def cost_gradient_thread(self, data_XYt):
        """
        Wrapper for multithreaded call
        :param data_XY:
        :return:
        """
        return self.cost_gradient(data_XYt[0], data_XYt[1], thread_scale=data_XYt[2])

    def cost_gradient(self, data_X, data_Y, thread_scale=1):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        :param data_X:
        :param data_Y:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dq = []
        dc_dr_x = [np.zeros(self.particle_input.output_size)]
        dc_dr_y = [np.zeros(self.particle_input.output_size)]
        dc_dr_z = [np.zeros(self.particle_input.output_size)]
        dc_dt = [np.zeros(self.particle_input.output_size)]
        dc_dt_in = [np.zeros(self.particle_input.output_size)]  # not used but just keep for ease
        dc_dzeta = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr_x.append(np.zeros(len(layer.q)))
            dc_dr_y.append(np.zeros(len(layer.q)))
            dc_dr_z.append(np.zeros(len(layer.q)))
            dc_dt.append(np.zeros(layer.theta.shape))
            dc_dt_in.append(np.zeros(layer.theta.shape))
            dc_dzeta.append(np.zeros(layer.zeta.shape))

        # Regularization options
        l2 = self.regularizer is not None and isinstance(self.regularizer, ParticleRegularizeL2)
        # L2plus
        l2plus = self.regularizer is not None and isinstance(self.regularizer, ParticleRegularizeL2Plus)
        # Ortho
        ortho = self.regularizer is not None and isinstance(self.regularizer, ParticleRegularizeOrthogonal)

        sigma_Z = []
        A_scaled, _ = self.particle_input.feed_forward(data_X)
        A = [A_scaled]  # Note: A has one more element than sigma_Z
        prev_layer_rr = self.particle_input.get_rxyz()
        for l, layer in enumerate(self.layers):
            if l2plus:
                layer.compute_w(prev_layer_rr)
            z = layer.compute_z(A[l], prev_layer_rr)
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))
            prev_layer_rr = layer.get_rxyz()

        delta_L = self.cost_d_function(data_Y, A[-1], sigma_Z[-1])

        # IMPORTANT:
        # For threaded calls, we need to divide the cost gradient by the number threads to account for the mean being
        # taken in the cost function. When data is split, the mean is off by a factor of the number of threads.
        if thread_scale > 1:
            delta_L /= thread_scale

        # For each piece of data
        for di, data in enumerate(data_X):
            dc_db[-1] += delta_L[di]

        # Reshape positions
        self.particle_input.rx = self.particle_input.rx.reshape((self.particle_input.output_size, 1))
        self.particle_input.ry = self.particle_input.ry.reshape((self.particle_input.output_size, 1))
        self.particle_input.rz = self.particle_input.rz.reshape((self.particle_input.output_size, 1))
        self.particle_input.zeta = self.particle_input.zeta.reshape((self.particle_input.output_size, 1))
        self.particle_input.theta = self.particle_input.theta.reshape((self.particle_input.output_size, 1))
        for layer in self.layers:
            layer.rx = layer.rx.reshape((layer.output_size, 1))
            layer.ry = layer.ry.reshape((layer.output_size, 1))
            layer.rz = layer.rz.reshape((layer.output_size, 1))
            layer.zeta = layer.zeta.reshape((layer.output_size, 1))
            layer.theta = layer.theta.reshape((layer.output_size, 1))
            layer.theta_in = layer.theta_in.reshape((layer.output_size, 1))

        l = -1
        layer = self.layers[l]
        prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((len(prev_layer.rx), len(data_X)))

        # Ortho help
        wt = None
        dd = None
        if ortho:
            wt = layer.w.transpose().copy()
            dd = np.zeros(layer.output_size)
            for j in range(layer.output_size):
                dd[j] = np.sqrt(wt[j].dot(wt[j]))
                wt[j] = wt[j] / dd[j]

        # Position gradient
        for j in range(layer.output_size):
            qj = layer.q[j]
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            dx = (prev_layer.rx - layer.rx[j])
            dy = (prev_layer.ry - layer.ry[j])
            dz = (prev_layer.rz - layer.rz[j])
            d2 = dx**2 + dy**2 + dz**2
            # zeta_i = prev_layer.zeta
            # zeta_ij = np.sqrt(zeta_i**2 * layer.zeta[j]**2)
            # exp_dij = np.exp(-zeta_ij * d2)
            exp_dij = np.exp(-d2)

            dt = 0.0
            if layer.phase_enabled and prev_layer.phase_enabled:
                dt = (prev_layer.theta - layer.theta[j])
                # dt = (prev_layer.theta - layer.theta_in[j])
                exp_dij *= np.cos(dt)

            # Next delta
            next_delta += (qj * trans_delta_L_j) * exp_dij * trans_sigma_Z_l

            # Charge gradient
            dq = exp_dij * Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq)

            # Position gradient
            # tmp = 2.0 * zeta_ij * qj * dq
            tmp = 2.0 * qj * dq
            tx = dx * tmp
            ty = dy * tmp
            tz = dz * tmp

            dc_dr_x[l][j] += np.sum(tx)
            dc_dr_y[l][j] += np.sum(ty)
            dc_dr_z[l][j] += np.sum(tz)

            dc_dr_x[l-1] -= np.sum(tx, axis=1)
            dc_dr_y[l-1] -= np.sum(ty, axis=1)
            dc_dr_z[l-1] -= np.sum(tz, axis=1)

            # # Width gradient
            # tmp = -qj * dq * d2
            # dc_dzeta[l][j] += np.sum(tmp * zeta_i)
            # dc_dzeta[l-1] += np.sum(tmp * layer.zeta[j], axis=1)

            if layer.phase_enabled and prev_layer.phase_enabled:
                # Phase gradient
                # dq *= -np.sin(dt) / np.cos(dt)  # could use tan but being explicit here
                # dq *= -np.tan(dt)
                tmp = -qj * dq * np.tan(dt)
                dc_dt[l][j] -= np.sum(tmp)
                # dc_dt_in[l][j] -= np.sum(tmp)
                dc_dt[l-1] += np.sum(tmp, axis=1)

            # Ortho help
            s = None
            if ortho:
                s = 2 * self.regularizer.coeff_lambda * (np.eye(len(wt[j])) - np.outer(wt[j], wt[j])) / dd[j]

            # ----- L2 regularized w_ij by position
            if l2plus:
                coeff_lambda = self.regularizer.coeff_lambda
                # Should be computed from before
                wt = layer.w.transpose()

                for kk in range(layer.output_size):
                    if j == kk:
                        continue

                    s = np.sign(wt[j].dot(wt[kk]))
                    # s = 2 * wt[j].dot(wt[kk])
                    dq = 2 * coeff_lambda * s * wt[kk].reshape((prev_layer.output_size, 1)) * exp_dij

                    # Charge
                    dc_dq[l][j] += np.sum(dq)

                    # Position
                    tmp = 2.0 * qj * dq
                    tx = dx * tmp
                    ty = dy * tmp
                    tz = dz * tmp

                    dc_dr_x[l][j] += np.sum(tx)
                    dc_dr_y[l][j] += np.sum(ty)
                    dc_dr_z[l][j] += np.sum(tz)

                    dc_dr_x[l-1] -= np.sum(tx, axis=1)
                    dc_dr_y[l-1] -= np.sum(ty, axis=1)
                    dc_dr_z[l-1] -= np.sum(tz, axis=1)

                    # Phase
                    if layer.phase_enabled and prev_layer.phase_enabled:
                        dq *= -np.tan(dt)
                        tmp = qj * dq
                        dc_dt[l][j] -= np.sum(tmp)
                        dc_dt[l - 1] += np.sum(tmp, axis=1)

            elif ortho:
                coeff_lambda = self.regularizer.coeff_lambda

                # Should be computed from before
                # wt = layer.w.transpose()

                for kk in range(layer.output_size):
                    if j == kk:
                        continue

                    # dj = np.sqrt(wt[j].dot(wt[j]))
                    # wtj = wt[j] / dj
                    # s = 2 * coeff_lambda * (np.eye(len(wtj)) - np.outer(wtj, wtj)) / dj
                    #
                    # dk = np.sqrt(wt[kk].dot(wt[kk]))
                    # wtk = wt[kk] / dk
                    # dq = (wtk.dot(s) * np.sign(wtj.dot(wtk))).reshape((prev_layer.output_size, 1)) * exp_dij

                    dq = (wt[kk].dot(s) * np.sign(wt[j].dot(wt[kk]))).reshape((prev_layer.output_size, 1)) * exp_dij

                    # Charge
                    dc_dq[l][j] += np.sum(dq)

                    # Position
                    tmp = 2.0 * qj * dq
                    tx = dx * tmp
                    ty = dy * tmp
                    tz = dz * tmp

                    dc_dr_x[l][j] += np.sum(tx)
                    dc_dr_y[l][j] += np.sum(ty)
                    dc_dr_z[l][j] += np.sum(tz)

                    dc_dr_x[l-1] -= np.sum(tx, axis=1)
                    dc_dr_y[l-1] -= np.sum(ty, axis=1)
                    dc_dr_z[l-1] -= np.sum(tz, axis=1)

                    # Phase
                    if layer.phase_enabled and prev_layer.phase_enabled:
                        dq *= -np.tan(dt)
                        tmp = qj * dq
                        dc_dt[l][j] -= np.sum(tmp)
                        dc_dt[l - 1] += np.sum(tmp, axis=1)

            elif l2:
                coeff_lambda = self.regularizer.coeff_lambda
                w_ij = qj * exp_dij

                # Charge gradient
                dq = 2 * coeff_lambda * w_ij * exp_dij
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                tmp = 2.0 * qj * dq
                tx = dx * tmp
                ty = dy * tmp
                tz = dz * tmp

                dc_dr_x[l][j] += np.sum(tx)
                dc_dr_y[l][j] += np.sum(ty)
                dc_dr_z[l][j] += np.sum(tz)

                dc_dr_x[l-1] -= np.sum(tx, axis=1)
                dc_dr_y[l-1] -= np.sum(ty, axis=1)
                dc_dr_z[l-1] -= np.sum(tz, axis=1)

                # Phase
                if layer.phase_enabled and prev_layer.phase_enabled:
                    dq *= -np.tan(dt)
                    tmp = qj * dq
                    dc_dt[l][j] -= np.sum(tmp)
                    dc_dt[l-1] += np.sum(tmp, axis=1)

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            next_delta = np.zeros((prev_layer.output_size, len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            # Ortho help
            wt = None
            dd = None
            if ortho:
                wt = layer.w.transpose().copy()
                dd = np.zeros(layer.output_size)
                for j in range(layer.output_size):
                    dd[j] = np.sqrt(wt[j].dot(wt[j]))
                    wt[j] = wt[j] / dd[j]

            # Position gradient
            for j in range(layer.output_size):
                qj = layer.q[j]
                this_delta_j = this_delta[j]

                dx = (prev_layer.rx - layer.rx[j])
                dy = (prev_layer.ry - layer.ry[j])
                dz = (prev_layer.rz - layer.rz[j])
                d2 = dx**2 + dy**2 + dz**2
                # zeta_i = prev_layer.zeta
                # zeta_ij = np.sqrt(zeta_i**2 * layer.zeta[j]**2)
                # exp_dij = np.exp(-zeta_ij * d2)
                exp_dij = np.exp(-d2)

                dt = 0.0
                if layer.phase_enabled and prev_layer.phase_enabled:
                    dt = (prev_layer.theta - layer.theta[j])
                    # dt = (prev_layer.theta - layer.theta_in[j])
                    exp_dij *= np.cos(dt)

                # Next delta
                next_delta += (qj * this_delta_j) * exp_dij * trans_sigma_Z_l

                # Charge gradient
                dq = exp_dij * Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                # tmp = 2.0 * zeta_ij * qj * dq
                tmp = 2.0 * qj * dq
                tx = dx * tmp
                ty = dy * tmp
                tz = dz * tmp

                dc_dr_x[l][j] += np.sum(tx)
                dc_dr_y[l][j] += np.sum(ty)
                dc_dr_z[l][j] += np.sum(tz)

                dc_dr_x[l-1] -= np.sum(tx, axis=1)
                dc_dr_y[l-1] -= np.sum(ty, axis=1)
                dc_dr_z[l-1] -= np.sum(tz, axis=1)

                # # Width gradient
                # tmp = -qj * dq * d2
                # dc_dzeta[l][j] += np.sum(tmp * zeta_i)
                # dc_dzeta[l-1] += np.sum(tmp * layer.zeta[j], axis=1)

                # Phase gradient
                if layer.phase_enabled and prev_layer.phase_enabled:
                    # dq *= -np.sin(dt) / np.cos(dt)  # could use tan but being explicit here
                    # dq *= -np.tan(dt)
                    tmp = -qj * dq * np.tan(dt)
                    dc_dt[l][j] -= np.sum(tmp)
                    # dc_dt_in[l][j] -= np.sum(tmp)
                    dc_dt[l - 1] += np.sum(tmp, axis=1)

                # Ortho help
                s = None
                if ortho:
                    s = 2 * self.regularizer.coeff_lambda * (np.eye(len(wt[j])) - np.outer(wt[j], wt[j])) / dd[j]

                # ----- L2 regularized w_ij by position
                if l2plus:
                    coeff_lambda = self.regularizer.coeff_lambda
                    wt = layer.w.transpose()
                    for kk in range(layer.output_size):
                        if j == kk:
                            continue

                        s = np.sign(wt[j].dot(wt[kk]))
                        # s = 2 * wt[j].dot(wt[kk])
                        dq = 2 * coeff_lambda * s * wt[kk].reshape((prev_layer.output_size, 1)) * exp_dij

                        # Charge
                        dc_dq[l][j] += np.sum(dq)

                        # Position
                        tmp = 2.0 * qj * dq
                        tx = dx * tmp
                        ty = dy * tmp
                        tz = dz * tmp

                        dc_dr_x[l][j] += np.sum(tx)
                        dc_dr_y[l][j] += np.sum(ty)
                        dc_dr_z[l][j] += np.sum(tz)

                        dc_dr_x[l - 1] -= np.sum(tx, axis=1)
                        dc_dr_y[l - 1] -= np.sum(ty, axis=1)
                        dc_dr_z[l - 1] -= np.sum(tz, axis=1)

                        # Phase
                        if layer.phase_enabled and prev_layer.phase_enabled:
                            dq *= -np.tan(dt)
                            tmp = qj * dq
                            dc_dt[l][j] -= np.sum(tmp)
                            dc_dt[l - 1] += np.sum(tmp, axis=1)

                elif ortho:
                    coeff_lambda = self.regularizer.coeff_lambda

                    # Should be computed from before
                    # wt = layer.w.transpose()

                    for kk in range(layer.output_size):
                        if j == kk:
                            continue

                        dq = (wt[kk].dot(s) * np.sign(wt[j].dot(wt[kk]))).reshape((prev_layer.output_size, 1)) * exp_dij

                        # Charge
                        dc_dq[l][j] += np.sum(dq)

                        # Position
                        tmp = 2.0 * qj * dq
                        tx = dx * tmp
                        ty = dy * tmp
                        tz = dz * tmp

                        dc_dr_x[l][j] += np.sum(tx)
                        dc_dr_y[l][j] += np.sum(ty)
                        dc_dr_z[l][j] += np.sum(tz)

                        dc_dr_x[l - 1] -= np.sum(tx, axis=1)
                        dc_dr_y[l - 1] -= np.sum(ty, axis=1)
                        dc_dr_z[l - 1] -= np.sum(tz, axis=1)

                        # Phase
                        if layer.phase_enabled and prev_layer.phase_enabled:
                            dq *= -np.tan(dt)
                            tmp = qj * dq
                            dc_dt[l][j] -= np.sum(tmp)
                            dc_dt[l - 1] += np.sum(tmp, axis=1)

                elif l2:
                    coeff_lambda = self.regularizer.coeff_lambda
                    w_ij = qj * exp_dij

                    # Charge gradient
                    dq = 2 * coeff_lambda * w_ij * exp_dij
                    dc_dq[l][j] += np.sum(dq)

                    # Position gradient
                    tmp = 2.0 * qj * dq
                    tx = dx * tmp
                    ty = dy * tmp
                    tz = dz * tmp

                    dc_dr_x[l][j] += np.sum(tx)
                    dc_dr_y[l][j] += np.sum(ty)
                    dc_dr_z[l][j] += np.sum(tz)

                    dc_dr_x[l-1] -= np.sum(tx, axis=1)
                    dc_dr_y[l-1] -= np.sum(ty, axis=1)
                    dc_dr_z[l-1] -= np.sum(tz, axis=1)

                    # Phase
                    if layer.phase_enabled and prev_layer.phase_enabled:
                        dq *= -np.tan(dt)
                        tmp = qj * dq
                        dc_dt[l][j] -= np.sum(tmp)
                        dc_dt[l-1] += np.sum(tmp, axis=1)

        # Position gradient list
        dc_dr = (dc_dr_x, dc_dr_y, dc_dr_z)

        # Restore shapes
        self.particle_input.rx = self.particle_input.rx.reshape((self.particle_input.output_size, ))
        self.particle_input.ry = self.particle_input.ry.reshape((self.particle_input.output_size, ))
        self.particle_input.rz = self.particle_input.rz.reshape((self.particle_input.output_size, ))
        self.particle_input.zeta = self.particle_input.zeta.reshape((self.particle_input.output_size, ))
        self.particle_input.theta = self.particle_input.theta.reshape((self.particle_input.output_size, ))
        self.particle_input.theta_in = self.particle_input.theta_in.reshape((self.particle_input.output_size, ))
        for layer in self.layers:
            layer.rx = layer.rx.reshape((layer.output_size, ))
            layer.ry = layer.ry.reshape((layer.output_size, ))
            layer.rz = layer.rz.reshape((layer.output_size, ))
            layer.zeta = layer.zeta.reshape((layer.output_size, ))
            layer.theta = layer.theta.reshape((layer.output_size, ))
            layer.theta_in = layer.theta_in.reshape((layer.output_size, ))

        # Perform charge regularization if needed
        if self.regularizer is not None:
            dc_dq, dc_db, dc_dr = self.regularizer.cost_gradient(self.particle_input, self.layers, dc_dq, dc_db, dc_dr)

        return dc_db, dc_dq, dc_dr, dc_dt, dc_dt_in

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

    def write_to_json(self, file=None):
        """
        Write network data to file in JSON format
        :param file: a file open for writing
        :return:
        """

        network = {"particle_input": {}, "layers": [], "cost_name": self.cost_name}

        p_inp = {"rx": [], "ry": [], "rz": [], "theta": []}
        for i in range(self.particle_input.output_size):
            p_inp["rx"].append(self.particle_input.rx[i])
            p_inp["ry"].append(self.particle_input.ry[i])
            p_inp["rz"].append(self.particle_input.rz[i])
            p_inp["theta"].append(self.particle_input.theta[i])
        network["particle_input"] = p_inp

        for layer in self.layers:
            l_data = {"rx": [], "ry": [], "rz": [], "q": [], "b": [], "theta": [],
                      "activation_name": layer.activation_name}
            for i in range(layer.output_size):
                l_data["q"].append(layer.q[i])
                l_data["b"].append(layer.b[0][i])
                l_data["rx"].append(layer.rx[i])
                l_data["ry"].append(layer.ry[i])
                l_data["rz"].append(layer.rz[i])
                l_data["theta"].append(layer.theta[i])
            network["layers"].append(l_data)

        if file is not None:
            json.dump(network, file)
        else:
            return json.dumps(network)

    @staticmethod
    def read_from_json(file, from_string=None):
        """
        Read network data from file in JSON format, return new ParticleNetwork
        :param file: a file open for reading
        :return:
        """

        data = None
        if from_string is None:
            data = json.load(file)
        else:
            data = json.loads(from_string)

        network = ParticleNetwork(cost=data.get("cost_name"))

        data_p_inp = data.get("particle_input")
        particle_input = ParticleInput(len(data_p_inp.get("rx")))
        for i, r in enumerate(data_p_inp.get("rx")):
            particle_input.rx[i] = r
        for i, r in enumerate(data_p_inp.get("ry")):
            particle_input.ry[i] = r
        for i, r in enumerate(data_p_inp.get("rz")):
            particle_input.rz[i] = r
        for i, t in enumerate(data_p_inp.get("theta")):
            particle_input.theta[i] = t
        network.particle_input = particle_input

        data_layers = data.get("layers")
        n_input = len(data_p_inp.get("rx"))
        for d_layer in data_layers:
            particle = Particle(input_size=n_input, output_size=len(d_layer.get("rx")),
                                activation=d_layer.get("activation_name"))
            for j, _ in enumerate(d_layer.get("rx")):
                particle.q[j] = d_layer.get("q")[j]
                particle.b[0][j] = d_layer.get("b")[j]
                for i, r in enumerate(d_layer.get("rx")):
                    particle.rx[i] = r
                for i, r in enumerate(d_layer.get("ry")):
                    particle.ry[i] = r
                for i, r in enumerate(d_layer.get("rz")):
                    particle.rz[i] = r
                for i, t in enumerate(d_layer.get("theta")):
                    particle.theta[i] = t
            network.layers.append(particle)
            n_input = len(d_layer.get("rx"))

        return network
