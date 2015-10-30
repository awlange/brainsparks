from .layer import Layer
from ..activation import Activation

import numpy as np


class Convolution1D(Layer):
    """
    TODO: refactor everything to take advantage of numpy's convolve function
    """

    def __init__(self, input_size=0, n_filters=0, filter_size=0, activation="sigmoid", stride_length=1,
                 flatten_output=False, max_pool=False, pool_size=0, pool_stride_length=1):
        super().__init__("Convolution1D", True)
        self.input_size = input_size
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride_length = stride_length
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.flatten_output = flatten_output
        self.max_pool = max_pool
        self.pool_size = pool_size
        self.pool_stride_length = pool_stride_length

        # Number of fields in convolution
        self.n_fields = 1 + (self.input_size - self.filter_size) // self.stride_length
        self.n_pool_fields = 1 + (self.n_fields - self.pool_size) // self.pool_stride_length

        # Params
        # self.b = np.asarray([[1.05*(o+1) for o in range(n_filters)]])
        # self.w = np.transpose(np.asarray([[0.01*(i+o+1) for i in range(filter_size)] for o in range(n_filters)]))

        self.b = np.asarray([[0.5 for o in range(n_filters)]])
        self.w = np.transpose(np.asarray([[0.1*(i+1) for i in range(filter_size)] for o in range(n_filters)]))

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        # TODO: Seems like we should be able to optimize this better to take advantage of numpy better
        # Perhaps move the loop over filters to outer most loop
        z = []
        for i_data in range(len(a_in)):
            # For each filter, convolve the input left to right
            convolution = []
            i = 0
            for field in range(self.n_fields):
                convolution.append((a_in[i_data][i:(i+self.filter_size)].dot(self.w) + self.b)[0])
                i += self.stride_length
            z.append(np.transpose(convolution))
        return np.asarray(z)

    def compute_a(self, z):
        a = self.activation(z)
        if self.max_pool:
            a = self.apply_max_pool(a)

        if self.flatten_output:
            return Convolution1D.flatten(a)
        return a

    def compute_da(self, z, a=None):
        d_a = self.d_activation(z)

        if self.max_pool and a is not None:
            # Compute non-pooled activations (some redundant work, I know...)
            non_pooled_a = self.activation(self.compute_z(a))
            # Get argmax indexes
            argmaxes = self.apply_max_pool(non_pooled_a, argmax=True)
            # Pluck out only the argmax gradients
            result = []
            for i_data in range(len(z)):
                result_filter = []
                for filter in range(self.n_filters):
                    result_field = []
                    for field in range(self.n_pool_fields):
                        result_field.append(d_a[i_data][filter][argmaxes[i_data][filter][field]])
                    result_filter.append(result_field)
                result.append(result_filter)
            d_a = np.asarray(result)

        if self.flatten_output:
            return Convolution1D.flatten(d_a)
        return d_a

    def compute_gradient(self, prev_delta, A, sigma_Z=None):
        # TODO: For max pooling, we'll only need to compute the gradient w/r to the max field

        if self.max_pool:
            return self.compute_gradient_max_pool(prev_delta, A)

        # Ensure shape
        prev_delta = prev_delta.reshape((self.n_filters, -1))

        dc_db = prev_delta if sigma_Z is None else self.w.dot(prev_delta) * sigma_Z

        # For each filter, convolve the input left to right
        convolution = []
        i = 0
        for field in range(self.n_fields):
            convolution.append(np.outer(A[i:(i+self.filter_size)], dc_db.transpose()[field]))
            i += self.stride_length
        dc_dw = np.transpose(convolution)

        return dc_db, dc_dw

    def compute_gradient_max_pool(self, prev_delta, A, sigma_Z=None):
        # Ensure shape
        prev_delta = prev_delta.reshape((self.n_filters, -1))

        dc_db = prev_delta if sigma_Z is None else self.w.dot(prev_delta) * sigma_Z

        # Get argmax indexes from A
        argmaxes = self.convolve_max_pool(A, argmax=True, by_filter=True)

        delta_b = np.zeros_like(self.b)
        delta_w = np.zeros_like(self.w)

        for filter in range(self.n_filters):
            for field in range(self.n_pool_fields):
                delta_b[0][filter] += dc_db[0][argmaxes[filter][field]]

        # For each filter, convolve the input left to right
        convolution = []
        i = 0
        for field in range(self.n_fields):
            convolution.append(np.outer(A[i:(i+self.filter_size)], delta_b.transpose()[field]))
            i += self.stride_length
        dc_dw = np.transpose(convolution)

        return delta_b, dc_dw

    def compute_gradient_update(self, dc_db, dc_dw, A=None, convolve=True):
        """
        Need to take sum across fields

        For max pooling, we'll probably need to do something else, like only use the max one

        :param dc_db:
        :param dc_dw:
        :return:
        """
        delta_b = None
        delta_w = None

        if self.max_pool:
            delta_b, delta_w = self.gradient_update_max_pool(
                dc_db.reshape((self.n_filters, self.n_fields)), dc_dw, A, convolve)
        else:
            # Sum across the fields
            delta_b = np.sum(dc_db.reshape((self.n_filters, self.n_fields)), axis=1)
            if not convolve:
                delta_w = np.sum(dc_dw.reshape((self.n_filters, self.filter_size, self.n_fields)), axis=2)
            else:
                # Need to convolve the gradients per field for each filter
                dc_dw_t = dc_dw.transpose()
                delta_w = np.zeros_like(self.w.transpose())
                for filter in range(self.n_filters):
                    i = 0
                    for field in range(self.n_fields):
                        delta_w[filter] += dc_dw_t[field + filter*self.n_fields][i:(i+self.filter_size)]
                        i += self.stride_length

        return delta_b, delta_w.transpose()

    def gradient_update_max_pool(self, dc_db, dc_dw, A, convolve):
        """
        For max pooling, just pluck out the argmax field gradients

        :param dc_db:
        :param dc_dw:
        :param convolve:
        :return:
        """

        # Get argmax indexes from A
        argmaxes = self.convolve_max_pool(A, True)

        delta_b = np.zeros_like(self.b)
        delta_w = np.zeros_like(self.w)

        for filter in range(self.n_filters):
            for field in range(self.n_pool_fields):
                delta_b[0][filter] += dc_db[0][argmaxes[filter][field]]

        # if not convolve:
        #     delta_w = np.sum(dc_dw.reshape((self.n_filters, self.filter_size, self.n_fields)), axis=2)
        # else:
        #     dc_dw_t = dc_dw.transpose()
        #     delta_w = np.zeros_like(self.w.transpose())
        #     for filter in range(self.n_filters):
        #         i = 0
        #         for field in range(self.n_fields):
        #             delta_w[filter] += dc_dw_t[field + filter*self.n_fields][i:(i+self.filter_size)]
        #             i += self.stride_length

        return delta_b, delta_w

    def apply_max_pool(self, a, argmax=False):
        """
        Given activation, max pool it
        :param a:
        :return:
        """
        a_pool = []
        for i_data in range(len(a)):
            a_pool.append(self.convolve_max_pool(a[i_data], argmax))
        return np.asarray(a_pool)

    def convolve_max_pool(self, a, argmax=False, by_filter=True):
        # Choose if we want the values or the indexes
        max_func = np.argmax if argmax else np.max

        if by_filter:
            a_filter = []
            for filter in range(self.n_filters):
                convolution = []
                i = 0
                for field in range(self.n_pool_fields):
                    convolution.append(max_func(a[filter][i:(i+self.pool_size)]))
                    i += self.pool_stride_length
                a_filter.append(convolution)
            return a_filter
        else:
            convolution = []
            i = 0
            for field in range(self.n_pool_fields):
                convolution.append(max_func(a[i:(i+self.pool_size)]))
                i += self.pool_stride_length
            return convolution

    @staticmethod
    def flatten(x):
        result = []
        for i, x in enumerate(x):
            result.append(x.flatten())
        return np.asarray(result)
