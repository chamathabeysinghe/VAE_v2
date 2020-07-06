import cv2
import math
import matplotlib.pyplot as plt


class VideoFileProcessor:
    def __init__(self, video_file_path, cropping=None, verbose=False, max_length=math.inf):
        self.video_file_path = video_file_path
        self.verbose = verbose
        self.video_length = None
        self.video_frames = None
        self.max_length = max_length
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.video_frame_masks = None
        self.cropping = cropping

    def process(self):
        self.convert_video_to_frames()
        self.train_bg_removal()
        return self.predict_bg_removal()

    def convert_video_to_frames(self):
        if self.verbose:
            print('Converting video fiel: {}'.format(self.video_file_path))
        vidcap = cv2.VideoCapture(self.video_file_path)
        success, image = vidcap.read()
        count = 0
        frames = []
        while success:
            count += 1
            if self.cropping is not None:
                image = self.crop_image(image)
            frames.append(image)
            if count >= self.max_length:
                break
            success, image = vidcap.read()
        self.video_frames = frames
        self.video_length = len(frames)

    def crop_image(self, img):
        cropped_img = img[self.cropping['x_min']:self.cropping['x_max'], self.cropping['y_min']:self.cropping['y_max']]
        return cropped_img

    def train_bg_removal(self):
        if self.verbose:
            print('Training background removal')
        for i, frame in enumerate(self.video_frames):
            self.background_subtractor.apply(frame)
            if i > 100:
                break

    def predict_bg_removal(self):
        if self.verbose:
            print('Removing backgrounds from image')
        masks = []
        for i, frame in enumerate(self.video_frames):
            mask = self.background_subtractor.apply(frame)
            masks.append(mask)
        self.video_frame_masks = masks
        return masks

    def get_video_frames(self):
        return self.video_frames

    def get_masks(self):
        return self.video_frame_masks


# videoFileProcessor = VideoFileProcessor('/Users/chamathabeysinghe/Projects/monash/Dataset/Tagless_ant_tracking'
#                                         '/10 ants, filmed from above, unattenuated light.MOV', max_length=100,
#                                         verbose=True)
# masks = videoFileProcessor.process()
# frames = videoFileProcessor.get_video_frames()
# for i, mask in enumerate(masks):
#     plt.imshow(mask, cmap='gray')
#     plt.show()
#     plt.imshow(frames[i], cmap='gray')
#     plt.show()
#
# print("Done")