import argparse
import math
import os
import time
from functools import partial

import numpy as np
import torch
import visdom

import pyro
import pyro.contrib.examples.multi_mnist as multi_mnist
import pyro.optim as optim
import pyro.poutine as poutine
from components.AIR import AIR, latents_to_tensor
from pyro.contrib.examples.util import get_data_directory
from pyro.infer import SVI, JitTraceGraph_ELBO, TraceGraph_ELBO
from utils.viz import draw_many, tensor_to_objs
from utils.visualizer import plot_mnist_sample
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd


class ModelPipeline:

    def __init__(self, weights_path, repeat_count):
        air = AIR(
            num_steps=3,
            x_size=75,
            use_masking=True,
            use_baselines=True,
            z_what_size=50,
            use_cuda=False,
            window_size=28,
            rnn_hidden_size=256,
            encoder_net=[200],
            decoder_net=[200]
        )
        air.load_state_dict(torch.load(weights_path))
        self.air = air
        self.repeat_count = repeat_count

    @staticmethod
    def filter_by_bw_ratio(img, bboxes, box_stds):
        selected_bboxes = []
        selected_stds = []
        selected_bw_ratios = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x, y, w, h = bbox
            clipped = np.clip(img, 0, 1)
            crop_img = clipped[int(y):int(y) + int(h), int(x):int(x) + int(w)]
            white_count = np.count_nonzero(crop_img > 0.01)
            black_count = np.count_nonzero(crop_img < 0.01)
            if (black_count > 0 and white_count / black_count < 0.09):
                continue
            selected_bboxes.append(bboxes[i])
            selected_stds.append(box_stds[i])
            selected_bw_ratios.append(white_count / max(black_count, .00001))
        return selected_bboxes, selected_stds, selected_bw_ratios

    @staticmethod
    def bounding_box(z_where, x_size):
        """This doesn't take into account interpolation, but it's close
        enough to be usable."""
        w = x_size / z_where.s
        h = x_size / z_where.s
        xtrans = -z_where.x / z_where.s * x_size / 2.
        ytrans = -z_where.y / z_where.s * x_size / 2.
        x = (x_size - w) / 2 + xtrans  # origin is top left
        y = (x_size - h) / 2 + ytrans
        return (x, y), w, h

    @staticmethod
    def get_bounding_boxes_for_image(img, z_arr):
        bounding_boxes = []
        for k, z in enumerate(z_arr):
            if z.pres > 0:
                (x, y), w, h = ModelPipeline.bounding_box(z, img.shape[0])
                x, y, w, h = x.item(), y.item(), w.item(), h.item()
                bounding_boxes.append([x, y, w, h])
        return bounding_boxes

    def multiple_predict(self, img):
        img_array = np.asarray([np.copy(img) for _ in range(self.repeat_count)])
        img_array = torch.from_numpy(img_array)

        trace = poutine.trace(self.air.guide).get_trace(img_array, None)
        z, recons = poutine.replay(self.air.prior, trace=trace)(img_array.size(0))
        z_wheres = tensor_to_objs(latents_to_tensor(z))
        bboxes_frame = []
        for counter, z in enumerate(z_wheres):
            bboxes = ModelPipeline.get_bounding_boxes_for_image(img, z)
            # img2 = draw_bboxes_img(img, bboxes)
            bboxes_frame.append(bboxes)
        #         plt.imshow(img2)
        #         plt.show()
        return bboxes_frame

    def match_boxes(self, bboxes_frame):
        parents = [[-1, -1, -1] for _ in range(self.repeat_count)]
        original_boxes = []  # [{parent: [], childs: [[]]}]

        for frame_index in range(0, self.repeat_count):
            current_boxes = bboxes_frame[frame_index]

            for box_index in range(len(current_boxes)):
                if parents[frame_index][box_index] == -1:
                    original_boxes.append({'parent': current_boxes[box_index], 'childs': []})

            if frame_index + 1 >= self.repeat_count:
                break

            future_boxes = bboxes_frame[frame_index + 1]

            roi_f_box_orginal_box = [[0 for _ in range(len(original_boxes))] for _ in range(len(future_boxes))]
            for f_box_index in range(len(future_boxes)):
                for o_box_index in range(len(original_boxes)):
                    bb_f = future_boxes[f_box_index]
                    bb_o = original_boxes[o_box_index]['parent']

                    roi_f_box_orginal_box[f_box_index][o_box_index] = ModelPipeline.get_iou(
                        {'x1': bb_f[0], 'x2': bb_f[0] + bb_f[2], 'y1': bb_f[1], 'y2': bb_f[1] + bb_f[3]},
                        {'x1': bb_o[0], 'x2': bb_o[0] + bb_o[2], 'y1': bb_o[1], 'y2': bb_o[1] + bb_o[3]}
                    )

            #     for f_box_index in range(len(future_boxes)):
            #         print(roi_f_box_orginal_box[f_box_index])
            roi_f_box_orginal_box = np.asarray(roi_f_box_orginal_box) * -1
            row_ind, col_ind = linear_sum_assignment(roi_f_box_orginal_box)

            for i in range(len(row_ind)):
                f_f_index = row_ind[i]
                o_f_index = col_ind[i]
                parents[frame_index + 1][f_f_index] = o_f_index
                original_boxes[o_f_index]['childs'].append(future_boxes[f_f_index])

            return original_boxes

    @staticmethod
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """

        assert bb1['x1'] <= bb1['x2']
        assert bb1['y1'] <= bb1['y2']
        assert bb2['x1'] <= bb2['x2']
        assert bb2['y1'] <= bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

            # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box.
        # NOTE: We MUST ALWAYS add +1 to calculate area when working in
        # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
        # is the bottom right pixel. If we DON'T add +1, the result is wrong.
        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
        bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    @staticmethod
    def calculate_mean_std(original_boxes):
        box_means = []
        box_stds = []
        for i, item in enumerate(original_boxes):
            parent = item['parent']
            boxes = item['childs']
            boxes.append(parent)
            boxes = np.asarray(boxes)
            box_means.append(np.mean(boxes, axis=0))
            box_stds.append(np.std(boxes, axis=0))
        return box_means, box_stds

    @staticmethod
    def draw_bboxes_img(img, bboxes):
        img = Image.fromarray((img * 255).astype(np.uint8), mode='L').convert('RGB')
        draw = ImageDraw.Draw(img)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=color[i % 3])
        return np.asarray(img)

    def run_test(self, img):
        bboxes_frame = self.multiple_predict(img)
        bboxes_frame = list(filter(lambda e: len(e) > 0, bboxes_frame))
        if len(bboxes_frame) < self.repeat_count * .50:
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8), mode='L').convert('RGB'))
            return 1, img, 1
        original_boxes = self.match_boxes(bboxes_frame)
        selected_boxes, box_stds = ModelPipeline.calculate_mean_std(original_boxes)
        selected_boxes, box_stds, selected_bw_ratios = ModelPipeline.filter_by_bw_ratio(img, selected_boxes, box_stds)
        #         selected_boxes, box_stds = filter_by_std(selected_boxes, box_stds)
        img_w_box = ModelPipeline.draw_bboxes_img(img, selected_boxes)
        prediction_str = list(map(lambda x: list(x[0]) + [x[1]], zip(selected_boxes, selected_bw_ratios)))

        return selected_boxes, img_w_box, prediction_str


# load_path = '/Users/chamathabeysinghe/Projects/monash/VAE_v2/checkpoints/model-size-75-3ants.ckpt'
# modelPipeline = ModelPipeline(load_path, 50)
# path = "/Users/chamathabeysinghe/Projects/monash/VAE_v2/data/synthetic/complex_dataset/original:_300-resize:_75-2_ants/00040.png"
# img = np.asarray(Image.open(path))
# img = img.astype(np.float32)
# img = img / 255.0
# selected_boxes, img_w_box, prediction_str = modelPipeline.run_test(img)
# plt.imshow(img_w_box)
# plt.show()
# print(prediction_str)
#
