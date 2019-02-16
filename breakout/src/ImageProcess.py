
import matplotlib
matplotlib.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageProcess():

    def colormat2bin(self, state, config):

        height, width, channle = state.shape
        s_height = int(height * 0.5)
        s_width = config.getint('agent', 'cnn_input_width')

        # plt.imshow(state)
        # plt.show()
        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # print(state.shape)
        # plt.imshow(state_gray, cmap='gray')
        # plt.show()
        # _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)
        #
        # state_gray_small = cv2.resize(state_gray, ( s_width, s_height), interpolation=cv2.INTER_AREA)
        # cnn_input_image = state_binary_small[25:, :]
        # cnn_input_image = state_gray_small.reshape((config.getint('agent', 'cnn_input_width'),
        #                                       config.getint('agent', 'cnn_input_height')))
        cnn_input_image = state_gray / 255
        # print(state_gray.shape)
        # print(cnn_input_image.shape)
        # plt.imshow(cnn_input_image, cmap='gray')
        # plt.show()
        # exit(0)
        # print(cnn_input_image[80:120,20:40])
        # print(np.max(cnn_input_image))
        # exit(0)
        return cnn_input_image

