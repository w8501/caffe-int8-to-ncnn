# -*- coding: utf-8 -*-

# BUG1989 is pleased to support the open source community by supporting ncnn available.
#
# Copyright (C) 2019 BUG1989. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""
Quantization module for generating the calibration tables will be used by
quantized (INT8) models from FP32 models.with bucket split,[k, k, cin, cout]
cut into "cout" buckets.
This tool is based on Caffe Framework.
"""
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import sys, os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import time
import datetime
from google.protobuf import text_format
from scipy import stats

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='find the pretrained caffe models int8 quantize scale value')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--mean', dest='mean',
                        help='value of mean', type=float, nargs=3)
    parser.add_argument('--norm', dest='norm',
                        help='value of normalize', type=float, nargs=1, default=1.0)
    parser.add_argument('--images', dest='images',
                        help='path to calibration images', type=str)
    parser.add_argument('--output_param', dest='output_param',
                        help='path to output quantize param file', type=str, default='quanzation-dev.param')
    parser.add_argument('--output_bin', dest='output_bin',
                        help='path to output quantize bin file', type=str, default='quanzation-dev.bin')
    parser.add_argument('--group', dest='group',
                        help='enable the group scale', type=int, default=1)
    parser.add_argument('--gpu', dest='gpu',
                        help='use gpu to forward', type=int, default=0)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()

# global params
QUANTIZE_NUM = 127
QUANTIZE_WINOGRAND_NUM = 31
STATISTIC = 1
INTERVAL_NUM = 8001


# ugly global params
quantize_layer_lists = []
# rename mapping for identical bottom top style
blob_name_decorated = {}
# 8001
def mxnet_quantize_blob(hist, hist_edges):
    """
    Return the best threshold value.
    Ref: https://github.com/apache/incubator-mxnet/blob/master/src/operator/quantization/calibrate.cc
    Args:
        hist: list, activations has been processed by histogram and normalize,size is 8001
        hist_edges: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL
    """
    num_quantized_bins = 255
    num_bins = len(hist)
    m_hist = np.array(hist).copy()
    look_index = m_hist > 0
    look_hist = m_hist[look_index]
    m_hist_edges = np.array(hist_edges).copy()
    zero_bin_idx = int(num_bins / 2)
    num_half_quantized_bins =  int(num_quantized_bins / 2)
    thresholds = np.zeros(int(num_bins / 2 + 1 - num_quantized_bins / 2), dtype=np.float64)
    divergence = np.zeros(thresholds.size, dtype=np.float64)
    for i in range(int(num_quantized_bins / 2) , zero_bin_idx + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]

        sliced_nd_hist = np.zeros(p_bin_idx_stop - p_bin_idx_start, dtype=np.int32)
        p = np.zeros(p_bin_idx_stop - p_bin_idx_start, dtype=np.float64)
        p[0] = 0
        p[-1] = 0
        for j in range(0, num_bins):
            if j <= p_bin_idx_start:
                p[0] += hist[j]
            elif j >= p_bin_idx_stop:
                p[-1] += hist[j]
            else:
                sliced_nd_hist[j - p_bin_idx_start] = hist[j]
                p[j - p_bin_idx_start] = hist[j]

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = int(sliced_nd_hist.size / num_quantized_bins)
        quantized_bins = np.zeros(num_quantized_bins, dtype=np.float64)

        for j in range(0, num_quantized_bins):
            start = j * num_merged_bins
            stop = (j + 1) * num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()

        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (sliced_nd_hist != 0).astype(np.int64)
        for j in range(0, num_quantized_bins):
            start = j * num_merged_bins
            stop = q.size if j == num_quantized_bins - 1 else (j + 1) * num_merged_bins
            is_nonzeros = (sliced_nd_hist[start:stop] != 0).astype(np.int64)
            norm = is_nonzeros.sum()
            if norm != 0:
                for k in range(start, stop):
                    if p[k] > 0:
                        q[k] = float(quantized_bins[j]) / float(norm)

        is_neg = (p < 0).astype(np.float32)
        is_neg1 = (q < 0).astype(np.float32)
        is_sum = is_neg.sum()
        is_sum1 = is_neg1.sum()
        if is_sum > 0 or is_sum1:
            a = 1
            b = 2
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        # calculate kl_divergence between q and p
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)
        # print(stats.entropy(p, q))
    min_kl_divergence = np.argmin(divergence)
    threshold_value = thresholds[min_kl_divergence]

    return threshold_value, min_kl_divergence


class QuantizeLayer:
    def __init__(self, name, blob_name, group_num):
        self.name = name
        self.blob_name = blob_name
        self.group_num = group_num
        self.weight_scale = np.zeros(group_num)
        self.blob_max = 0.0
        self.blob_distubution_interval = 0.0
        self.blob_distubution = np.zeros(INTERVAL_NUM)
        self.blob_edge = np.zeros(INTERVAL_NUM)
        self.blob_threshold = 0
        self.blob_scale = 1.0
        self.group_zero = np.zeros(group_num)

    def quantize_weight(self, weight_data, flag):
        # spilt the weight data by cout num
        blob_group_data = np.array_split(weight_data, self.group_num)
        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            threshold = max(abs(max_val), abs(min_val))
            if threshold < 0.0001:
                self.weight_scale[i] = 0
                self.group_zero[i] = 1
            else:
                if (flag == True):
                    self.weight_scale[i] = QUANTIZE_WINOGRAND_NUM / threshold
                else:
                    self.weight_scale[i] = QUANTIZE_NUM / threshold
            print("%-20s group : %-5d max_val : %-10f scale_val : %-10f" % (
            self.name + "_param0", i, threshold, self.weight_scale[i]))

    def initial_blob_max(self, blob_data):
        # get the max value of blob
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        self.blob_distubution_interval = STATISTIC * self.blob_max / INTERVAL_NUM
        print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (
        self.name, self.blob_max, self.blob_distubution_interval))

    def initial_histograms(self, blob_data):
        # collect histogram of every group channel blob
        th = self.blob_max
        hist, hist_edge = np.histogram(blob_data, bins=INTERVAL_NUM, range=(-th, th))
        self.blob_edge = hist_edge
        self.blob_distubution += hist

    def quantize_blob(self):
        # calculate threshold
        distribution = np.array(self.blob_distubution)
        mxnet_threshold, num =  mxnet_quantize_blob(distribution, self.blob_edge)
        self.blob_scale = 127 / abs(mxnet_threshold)
        print("%-20s bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (
        self.name, num, mxnet_threshold, self.blob_distubution_interval, self.blob_scale))

        # pick threshold which minimizes KL divergence
        # threshold_bin = threshold_distribution(distribution)
        # self.blob_threshold = threshold_bin
        # threshold = (threshold_bin + 0.5) * self.blob_distubution_interval
        # # get the activation calibration value
        # self.blob_scale = QUANTIZE_NUM / threshold
        # print("%-20s bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (
        # self.name, threshold_bin, threshold, self.blob_distubution_interval, self.blob_scale))



def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value.
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL
    """
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        #
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many b0ins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin

        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p) # with some bugs, need to fix
        q = _smooth_distribution(q)
        # p[p == 0] = 0.0001
        # q[q == 0] = 0.0001

        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


def net_forward(net, image_path, transformer):
    """
    network inference and statistics the cost time
    Args:
        net: the instance of Caffe inference
        image_path: a image need to be inference
        transformer:
    Returns:
        none
    """
    # load image
    img = caffe.io.load_image(image_path)

    # transformer.preprocess the image
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    # net forward
    output = net.forward()


def file_name(file_dir):
    """
    Find the all file path with the directory
    Args:
        file_dir: The source file directory
    Returns:
        files_path: all the file path into a list
    """
    files_path = []

    for root, dir, files in os.walk(file_dir):
        for name in files:
            file_path = root + "/" + name
            print(file_path)
            files_path.append(file_path)

    return files_path


def network_prepare(net, mean, norm):
    """
    instance the prepare process param of caffe network inference
    Args:
        net: the instance of Caffe inference
        mean: the value of mean
        norm: the value of normalize
    Returns:
        none
    """
    print("Network initial")

    img_mean = np.array(mean)

    # initial transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # convert hwc to cwh
    transformer.set_transpose('data', (2, 0, 1))
    # load meanfile
    transformer.set_mean('data', img_mean)
    # resize image data from [0,1] to [0,255]
    transformer.set_raw_scale('data', 255)
    # convert RGB -> BGR
    transformer.set_channel_swap('data', (2, 1, 0))
    # normalize
    transformer.set_input_scale('data', norm)

    return transformer


def weight_quantize(net, net_file, group_on):
    """
    CaffeModel convolution weight blob Int8 quantize
    Args:
        net: the instance of Caffe inference
        net_file: deploy caffe prototxt
    Returns:
        none
    """
    print("\nQuantize the kernel weight:")
    print(len(net._layer_names))
    for name in net._layer_names:
        print(name)
    print('type:')
    for layer in net.layers:
        print(layer.type)

    # parse the net param from deploy prototxt
    params = caffe_pb2.NetParameter()
    with open(net_file) as f:
        text_format.Merge(f.read(), params)

    for i, layer in enumerate(params.layer):

        # find the convolution layers to get out the weight_scale
        if (layer.type == "Convolution" or layer.type == "ConvolutionDepthwise" or layer.type == "InnerProduct"):
            weight_blob = net.params[layer.name][0].data
             # initial the instance of QuantizeLayer Class lists,you can use enable group quantize to generate int8 scale for each group layer.convolution_param.group
            if (group_on == 1):
                if (layer.type == "Convolution" or layer.type == "ConvolutionDepthwise"):
                    quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], layer.convolution_param.num_output)
                elif (layer.type == "InnerProduct"):
                    quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], layer.inner_product_param.num_output)
            else:
                quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], 1)

            # quantize the weight value using 6bit for conv3x3s1 layer to winograd F(4,3)
            # if (layer.type == "Convolution" and layer.convolution_param.kernel_size[0] == 3 and (
            #         (len(layer.convolution_param.stride) == 0) or layer.convolution_param.stride[0] == 1)):
            #     if (layer.convolution_param.group != layer.convolution_param.num_output):
            #         quanitze_layer.quantize_weight(weight_blob, True)
            #     else:
            #         quanitze_layer.quantize_weight(weight_blob, False)
            # # quantize the weight value using 8bit for another conv layers
            # else:
            # quanitze_layer.quantize_weight(weight_blob, False)
            if (layer.type == "Convolution" or layer.type == "InnerProduct"):
                quanitze_layer.quantize_weight(weight_blob, False)
            quantize_layer_lists.append(quanitze_layer)
            # add the quantize_layer into the save list
    return None


def activation_quantize(net, transformer, images_files):
    """
    Activation Int8 quantize, optimaize threshold selection with KL divergence,
    given a dataset, find the optimal threshold for quantizing it.
    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    Args:
        net: the instance of Caffe inference
        transformer:
        images_files: calibration dataset
    Returns:
        none
    """
    print("\nQuantize the Activation:")
    # run float32 inference on calibration dataset to find the activations range
    for i, image in enumerate(images_files):
        # inference
        net_forward(net, image, transformer)
        # find max threshold
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_blob_max(blob)
        if i % 100 == 0:
            print("loop stage 1 : %d/%d" % (i, len(images_files)))

    # calculate statistic blob scope and interval distribution
    for layer in quantize_layer_lists:
        layer.initial_blob_distubution_interval()

        # for each layers
    # collect histograms of activations
    print("\nCollect histograms of activations:")
    for i, image in enumerate(images_files):
        net_forward(net, image, transformer)
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_histograms(blob)
        if i % 100 == 0:
            print("loop stage 2 : %d/%d" % (i, len(images_files)))

            # calculate threshold with KL divergence
    for layer in quantize_layer_lists:
        layer.quantize_blob()

    return None





def get_blob_count(net):
# bottom blob reference
bottom_reference = {}
    blob_names = []
    for layer_name in net._layer_names:
        for blob_name in net.bottom_names[layer_name]:
            if blob_name in blob_name_decorated.keys():
                blob_name = blob_name_decorated[blob_name]

            blob_names.append(blob_name)
            if blob_name not in bottom_reference.keys():
                bottom_reference[blob_name] = 1
            else:
                bottom_reference[blob_name] = bottom_reference[blob_name] + 1

        if len(net.bottom_names[layer_name]) == 1 and len(net.top_names[layer_name]) == 1 and \
                net.bottom_names[layer_name][0] == net.top_names[layer_name][0]:
            blob_name = net.top_names[layer_name][0] + '_' + layer_name
            blob_name_decorated[net.top_names[layer_name][0]] = blob_name
            blob_names.append(blob_name)
        else:
            for blob_name in net.top_names[layer_name]:
                blob_names.append(blob_name)

    blob_names = set(blob_names)
    return len(blob_names)

def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python caffe-int8-scale-tools-dev.py -h")

def write_ncnn_model(proto_path, bin_path, params, net, int8_scale_term, quantize_layer_list):
    proto_file = open(proto_path, 'wt')
    bin_file = open(bin_path, 'wb')
    # magic key
    proto_file.write('7767517\n')
    blob_count = get_blob_count(net)
    proto_file.write('%lu %lu\n' % (len(net._layer_names), blob_count))
    for layer, layer_name in zip(net.layers, net._layer_names):
        # find blob binary by layer name
        netidx = 0
        for netidx in range(0, len(params.layer)):
            if params.layer[netidx].name == layer_name:
                break


        if layer.type == 'Convolution':
            layer_param = params.layer[netidx]
            if layer_param.convolution_param.group != 1:
                proto_file.write('%-16s' % ("ConvolutionDepthWise"))
            else:
                proto_file.write('%-16s' % ("Convolution"))
        elif layer.type == 'Input':
            proto_file.write("%-16s" % (layer.type))
        else:
            proto_file.write("%-16s" % (layer.type))

        proto_file.write(
            "%-16s %d %d" % (layer_name, len(net.bottom_names[layer_name]), len(net.top_names[layer_name])))
        for bottom_name in net.bottom_names[layer_name]:
            proto_file.write(" %s" % (bottom_name))

        if len(net.bottom_names[layer_name]) == 1 and len(net.top_names[layer_name]) == 1 and \
                net.bottom_names[layer_name][0] == net.top_names[layer_name][0]:
            blob_name = net.top_names[layer_name][0] + '_' + layer_name
            blob_name_decorated[net.top_names[layer_name][0]] = blob_name

            proto_file.write(" %s" % (blob_name))
        else:
            for top_name in net.top_names[layer_name]:
                proto_file.write(" %s" % (top_name))

        if layer.type == "Convolution" or layer.type == "ConvolutionDepthwise" or layer.type == "DepthwiseConvolution":
            layer_param = params.layer[netidx]
            weight_blob = net.params[layer_name][0].data
            bias_blob = net.params[layer_name][1].data
            proto_file.write(" 0=%d" % (layer_param.convolution_param.num_output))
            if layer_param.convolution_param.kernel_h and layer_param.convolution_param.kernel_w:
                proto_file.write(" 1=%d" % (layer_param.convolution_param.kernel_w))
                proto_file.write(" 11=%d" % (layer_param.convolution_param.kernel_h))
            else:
                proto_file.write(" 1=%d" % (layer_param.convolution_param.kernel_size[0]))

            if len(layer_param.convolution_param.dilation) != 0:
                proto_file.write(" 2=%d" % (len(layer_param.convolution_param.dilation)))
            else:
                proto_file.write(" 2=%d" % (1))

            if layer_param.convolution_param.stride_h and layer_param.convolution_param.stride_w:
                proto_file.write(" 3=%d" % (layer_param.convolution_param.stride_w))
                proto_file.write(" 13=%d" % (layer_param.convolution_param.stride_h))
            else:
                proto_file.write(" 3=%d" % (layer_param.convolution_param.stride[0]))

            if layer_param.convolution_param.pad_h and layer_param.convolution_param.pad_w:
                proto_file.write(" 4=%d" % (layer_param.convolution_param.pad_w))
                proto_file.write(" 14=%d" % (layer_param.convolution_param.pad_h))
            else:
                if len(layer_param.convolution_param.pad) == 0:
                    proto_file.write(" 4=0")
                else:
                    proto_file.write(" 4=%d" % layer_param.convolution_param.pad[0])

            proto_file.write(" 5=%d" % (int(layer_param.convolution_param.bias_term)))
            proto_file.write(" 6=%d" % (weight_blob.size))

            num_group = 1
            if layer.type == "ConvolutionDepthwise" or layer.type == "DepthwiseConvolution":
                num_group = layer_param.convolution_param.num_output
            else:
                num_group = layer_param.convolution_param.group

            if num_group != 1:
                proto_file.write(" 7=%d" % (num_group))
            q_layer_index = 0
            quantize_tag = 0
            if int8_scale_term:

                for q_layer in quantize_layer_list:
                    if q_layer.name == layer_name:
                        break
                    q_layer_index = q_layer_index + 1

                if len(quantize_layer_list[q_layer_index].weight_scale) != num_group:
                    proto_file.write(" 8=1")
                else:
                    proto_file.write(" 8=2")
                quantize_tag = 871224

                for i in range(0, layer_param.convolution_param.num_output):
                    weight_blob[i] = np.round(weight_blob[i] * quantize_layer_list[q_layer_index].weight_scale[i])

                weight_blob = weight_blob.astype(np.int8)

            bin_file.write(np.array(quantize_tag).tobytes())
            bin_file.write(np.array(weight_blob).tobytes())
            # sz = weight_blob.size
            # sz = (sz + 4 - 1) & -4;
            # print(sz)
            # if sz != weight_blob.size:
            #     pad_num  = sz - weight_blob.size
            #     for i in range(0, pad_num):
            #         p = np.array(0)
            #         p = p.astype(np.int8)
            #         bin_file.write(p.tobytes())
            bin_file.write(np.array(bias_blob).tobytes())
            if int8_scale_term:
                q_w = np.array(quantize_layer_list[q_layer_index].weight_scale)
                q_w = q_w.astype(np.float32)

                q_a =np.array(np.float32(quantize_layer_list[q_layer_index].blob_scale))
                bin_file.write(q_w.tobytes())
                bin_file.write(q_a.tobytes())
        elif layer.type == "Input":
            input_layer = params.layer[netidx]
            blob_shape = input_layer.input_param.shape[0]
            if len(blob_shape.dim) == 4:
                proto_file.write(" 0=%ld" % (blob_shape.dim[3]))
                proto_file.write(" 1=%ld" % (blob_shape.dim[2]))
                proto_file.write(" 2=%ld" % (blob_shape.dim[1]))
            elif len(blob_shape.dim) == 3:
                proto_file.write(" 0=%ld" % (blob_shape.dim[3]))
                proto_file.write(" 1=%ld" % (blob_shape.dim[2]))
                proto_file.write(" 2=-233")
            elif len(blob_shape.dim) == 2:
                proto_file.write(" 0=%ld" % (blob_shape.dim[3]))
                proto_file.write(" 1=-233")
                proto_file.write(" 2=-233")
        elif layer.type == "InnerProduct":
            fc_layer =  params.layer[netidx]
            weight_blob = net.params[layer_name][0].data
            bias_blob = net.params[layer_name][1].data
            proto_file.write(" 0=%d" %(fc_layer.inner_product_param.num_output))
            if fc_layer.inner_product_param.bias_term:
                proto_file.write(" 1=1")
            else:
                proto_file.write(" 1=0")
            proto_file.write(" 2=%d" % (weight_blob.size))

            if int8_scale_term:
                proto_file.write(" 8=1")


            q_layer_index = 0
            quantize_tag = 0
            if int8_scale_term:
                quantize_tag = 871224
                for i in range(0, fc_layer.inner_product_param.num_output):
                    weight_blob[i] = np.round(weight_blob[i] * quantize_layer_list[q_layer_index].weight_scale[i])
                weight_blob = weight_blob.astype(np.int8)



            bin_file.write(np.array(quantize_tag).tobytes())
            bin_file.write(np.array(weight_blob).tobytes())
            # sz = weight_blob.size
            # sz = (sz + 4 - 1) & -4;
            # print(sz)
            # if sz != weight_blob.size:
            #     pad_num  = sz - weight_blob.size
            #     for i in range(0, pad_num):
            #         p = np.array(0)
            #         p = p.astype(np.int8)
            #         bin_file.write(p.tobytes())
            bin_file.write(np.array(bias_blob).tobytes())
            if int8_scale_term:
                q_w = np.array(quantize_layer_list[q_layer_index].weight_scale)
                q_w = q_w.astype(np.float32)

                q_a =np.array(np.float32(quantize_layer_list[q_layer_index].blob_scale))
                bin_file.write(q_w.tobytes())
                bin_file.write(q_a.tobytes())

            if fc_layer.inner_product_param.num_output == 2:
                tmp_blob = np.array((1,1), dtype=np.float32)

                bin_file.write(tmp_blob.tobytes())

        elif layer.type == "PReLU":
            slope_blob = net.params[layer_name][0].data
            proto_file.write(" 0=%d" % (len(slope_blob)))
            bin_file.write(np.array(slope_blob).tobytes())
        elif layer.type == "Pooling":
            pooling_layer = params.layer[netidx]
            proto_file.write(" 0=%d" % (pooling_layer.pooling_param.pool))
            if pooling_layer.pooling_param.kernel_w and pooling_layer.pooling_param.kernel_h:
                proto_file.write(" 1=%d" % (pooling_layer.pooling_param.kernel_w))
                proto_file.write(" 11=%d" % (pooling_layer.pooling_param.kernel_h))
            else:
                proto_file.write(" 1=%d" % (pooling_layer.pooling_param.kernel_size))

            if pooling_layer.pooling_param.stride_w and pooling_layer.pooling_param.stride_h:
                proto_file.write(" 2=%d" % (pooling_layer.pooling_param.stride_w))
                proto_file.write(" 12=%d" % (pooling_layer.pooling_param.stride_h))
            else:
                proto_file.write(" 2=%d" % (pooling_layer.pooling_param.stride))

            if pooling_layer.pooling_param.pad_w and pooling_layer.pooling_param.pad_h:
                proto_file.write(" 3=%d" % (pooling_layer.pooling_param.pad_w))
                proto_file.write(" 13=%d" % (pooling_layer.pooling_param.pad_h))
            else:
                proto_file.write(" 3=%d" % (pooling_layer.pooling_param.pad))
            if pooling_layer.pooling_param.global_pooling:
                proto_file.write(" 4=1")
            else:
                proto_file.write(" 4=0")

        elif layer.type == "Reshape":
            reshape_layer = params.layer[netidx]
            blob_shape = reshape_layer.reshape_param.shape
            if len(blob_shape.dim) == 1:
                proto_file.write(" 0=%ld 1=-233 2=-233" % (blob_shape.dim[0]))
            elif len(blob_shape.dim) == 2:
                proto_file.write(" 0=%ld 1=-233 2=-233" % (blob_shape.dim[1]))
            elif len(blob_shape.dim) == 3:
                proto_file.write(" 0=%ld 1=%ld 2=-233" % (blob_shape.dim[2], blob_shape.dim[1]))
            else:
                proto_file.write(" 0=%ld 1=%ld 2=%ld" % (blob_shape.dim[3], blob_shape.dim[2], blob_shape.dim[1]))
            proto_file.write(" 3=0")

        elif layer.type == "ReLU":
            global relu_size_index
            tmp_blob = np.zeros(relu_size[relu_size_index],dtype=np.float32)
            relu_size_index += 1
            bin_file.write(tmp_blob.tobytes())

        elif layer.type == "Softmax":
            softmax_layer = params.layer[netidx]
            dim = softmax_layer.softmax_param.axis - 1
            proto_file.write(" 0=%d" % (dim))
            proto_file.write(" 1=1")

        proto_file.write("\n")
    proto_file.close()
    bin_file.close()


def main():
    """
    main function
    """

    time_start = datetime.datetime.now()

    print(args)

    
    if args.proto == None or args.model == None or args.mean == None or args.images == None:
        usage_info()
        return None
    
    # deploy caffe prototxt path
    net_file = args.proto

    # trained caffemodel path
    caffe_model = args.model

    # mean value
    mean = args.mean

    # norm value
    if args.norm != 1.0:
        norm = args.norm

    # calibration dataset
    images_path = args.images

    # the output calibration file
    calibration_path = args.output

    # enable the group scale
    group_on = args.group

    # default use CPU to forwark
    if args.gpu != 0:
        caffe.set_device(0)
        caffe.set_mode_gpu()

    # initial caffe net and the forword model(GPU or CPU)
    net = caffe.Net(net_file, caffe_model, caffe.TEST)

    # prepare the cnn network
    transformer = living_network_prepare(net)

    # get the calibration datasets images files path
    images_files = file_name(images_path)
    get_blob_count(net)
    # quanitze kernel weight of the caffemodel to find it's calibration table
    weight_quantize(net, net_file, group_on)
    for q_layer in quantize_layer_lists:
        print(q_layer.name)

    # quantize activation value of the caffemodel to find it's calibration table
    params = caffe_pb2.NetParameter()

    with open(net_file) as f:
        text_format.Merge(f.read(), params)
        
    activation_quantize(net, transformer, images_files)
    write_ncnn_model("F:\\model4_q_NEW.param", "F:\\model4_q_1205.bin", params, net, 1, quantize_layer_lists)
    # time end

    time_end = datetime.datetime.now()

    print(
        "\nCaffeModel Int8 to NCNN_Net create success, it's cost %s, best wish for your INT8 inference has a low accuracy loss...\(^▽^)/...2333..." % (
                    time_end - time_start))



if __name__ == "__main__":
    main()
