

import cv2

class ImageProcess():

    def colormat2bin(self, state, config):

        height, width, channle = state.shape
        s_height = int(height * 0.5)
        s_width = config.cnn_input_witdh

        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)
        state_binary_small = cv2.resize(state_binary, ( s_width, s_height), interpolation=cv2.INTER_AREA)
        cnn_input_image = state_binary_small[25:, :]

        cnn_inputImg = cnn_input_image.reshape((config.cnn_input_witdh, config.cnn_input_height))

        return cnn_input_image

