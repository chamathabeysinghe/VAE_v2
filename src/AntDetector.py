from processing.VideoFileProcessor import VideoFileProcessor
from predict.ModelPipeline import ModelPipeline
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import visdom
from PIL import Image
vis = visdom.Visdom(env='main')

# 1. Get images from videos (bw, colored)
# 2. Split images to parts
# 3. Run bw image portions through the model
# 3. Combine image parts


class FrameHandler:
    def __init__(self, modelPipeline, input_shape=(75, 75), crop_shape=(300, 300)):
        self.modelPipeline = modelPipeline
        self.splits = []
        self.image = None
        self.input_shape = input_shape
        self.crop_shape = crop_shape

    @staticmethod
    def resize_image(img, resize_shape):
        res = cv2.resize(img, dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
        return res

    @staticmethod
    def split_image(image, window):
        plt.imshow(image, cmap='gray')
        plt.show()
        splits = []
        for x in range(0, image.shape[0] - window[0], window[0]):
            for y in range(0, image.shape[1] - window[1], window[1]):
                crop = image[x: x + window[0], y: y + window[1]]
                splits.append(FrameHandler.resize_image(crop, (75, 75)))
                # plt.imshow(crop, cmap='gray')
                # plt.show()
        return splits

    @staticmethod
    def merge_image(splits, window, original_shape):
        image = np.zeros((original_shape[0], original_shape[1], 3))
        count = 0
        for x in range(0, original_shape[0] - window[0], window[0]):
            for y in range(0, original_shape[1] - window[1], window[1]):
                image[x: x + window[0], y: y + window[1], :] = splits[count]
                count += 1
        plt.imshow(image)
        plt.show()

        return image

    def initialize_image(self, image):
        self.image = image
        splits = FrameHandler.split_image(image, self.crop_shape)
        predictions = []
        for count, split in enumerate(splits):
            print(count)
            network_img = np.copy(split)
            network_img = network_img.astype(np.float32)
            network_img = network_img / 255.0
            # plt.imshow(network_img, cmap='gray')
            # plt.show()
            # vis.image(split.reshape(split.shape + (3,)).transpose((2, 0, 1)))
            selected_boxes, img_w_box, prediction_str = self.modelPipeline.run_test(network_img)
            predictions.append(img_w_box)
            # vis.image(img_w_box.transpose((2, 0, 1)))
            # plt.imshow(img_w_box)
            # plt.show()
            # predictions.append(split)
        FrameHandler.merge_image(predictions, self.input_shape, (image.shape[0] // 4, image.shape[1] // 4))


class AntDetector:

    def __init__(self, video_file_path, max_length=math.inf, cropping=None):
        self.video_file_path = video_file_path
        self.video_frames = None
        self.video_frame_masks = None
        self.videoFileProcessor = VideoFileProcessor(video_file_path, cropping=cropping, max_length=max_length)

    def extract_video(self):
        masks = self.videoFileProcessor.process()
        frames = self.videoFileProcessor.get_video_frames()
        self.video_frame_masks = masks
        self.video_frames = frames

    def predict_ants(self, frame_index):
        load_path = '/Users/chamathabeysinghe/Projects/monash/VAE_v2/checkpoints/model-size-75-3ants.ckpt'
        modelPipeline = ModelPipeline(load_path, repeat_count=50)
        frameHandler = FrameHandler(modelPipeline)
        frameHandler.initialize_image(self.video_frame_masks[frame_index])


antDetector = AntDetector('/Users/chamathabeysinghe/Projects/monash/Dataset/Tagless_ant_tracking'
                          '/10 ants, filmed from above, unattenuated light.MOV', max_length=100)

antDetector.extract_video()
antDetector.predict_ants(10)
# for i in range(50):
#     antDetector.predict_ants(i)
