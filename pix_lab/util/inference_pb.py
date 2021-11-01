from __future__ import print_function, division

import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import misc
from pix_lab.util.util import load_graph

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, image, scale=1.0):
        self.graph = load_graph(path_to_pb)
        self.image = image
        self.scale = scale

    def inference(self, gpu_device="0"):
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0') 
            batch_x = self.load_img(self.image, self.scale)
            pred = sess.run(predictor, feed_dict={x: batch_x})
            return pred[0,:, :,2]

    def load_img(self, image, scale):
        sImg = misc.imresize(image, scale, interp='bicubic')
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)
        fImg = np.expand_dims(fImg,0)
        return fImg

