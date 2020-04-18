# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
import numpy as np

from PIL import Image, ImageColor, ImageDraw, ImageFont


def get_color(color):
    color = '#' + color[::-1]
    return ImageColor.getcolor(color, 'RGB')


STANDARD_COLORS = ['white', 'lime', 'magenta', 'teal', get_color('FFD800'), 'gray', get_color('7F3300')]


def draw_bounding_box_on_image(image,
                               y_min,
                               x_min,
                               y_max,
                               x_max,
                               color='white',
                               font_size=12,
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):

    img = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(img)
    im_width, im_height = img.size
    if use_normalized_coordinates is None:
        (left, right, top, bottom) = (0, im_width, 0, im_height)
    elif use_normalized_coordinates:
        (left, right, top, bottom) = (x_min * im_width, x_max * im_width,
                                      y_min * im_height, y_max * im_height)
    else:
        (left, right, top, bottom) = (x_min, x_max, y_min, y_max)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = bottom
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin
    np.copyto(image, np.array(img))


def get_boxes_and_labels(boxes,
                         classes,
                         scores,
                         category_index,
                         instance_masks=None,
                         min_score_thresh=.5,
                         agnostic_mode=False):

    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    detect = []
    percent = 0
    max_boxes_to_draw = 99
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    percent = int(100 * scores[i])
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS) - 1]
                detect.append({'id': classes[i], 'name': category_index[classes[i]]['name'], 'score': scores[i],
                               '%': percent, 'color': box_to_color_map[box]})

    return detect[0] if len(detect) > 0 else None


def draw(image,
         detect,
         use_normalized_coordinates=None,
         font_size=12,
         line_thickness=4):
    display_str = ['{}: {}%'.format(
        detect['name'],
        detect['%'])]

    draw_bounding_box_on_image(
        image,
        0,
        0,
        1,
        1,
        color=detect['color'],
        font_size=font_size,
        thickness=line_thickness,
        display_str_list=display_str,
        use_normalized_coordinates=use_normalized_coordinates)
