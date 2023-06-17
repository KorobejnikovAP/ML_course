"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2

from visual_api.handlers import SyncExecutor
from visual_api.models import Classification
import visual_api.launchers as launchers
from visual_api.common import NetworkInfo, open_images_capture, read_model_config, PerformanceMetrics
from visual_api.visualizers import ClassificationVisualizer

from time import perf_counter
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-c', '--config', required=True, type=Path, help='Required. Path to model config file')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-topk', help='Optional. Number of top results. Default value is 5. Must be from 1 to 10.', default=5,
                                   type=int, choices=range(1, 11))

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    delay = int(cap.get_type() in {'VIDEO', 'CAMERA'})

    # 1 create launcher
    model_configuration = read_model_config(args.config)
    launcher = launchers.create_launcher_by_model_path(model_configuration)
    log.info("Use {}".format(launcher.get_launcher_name()))

    # 2 create model
    user_config = {
        'mean_values':  args.mean_values,
        'scale_values': args.scale_values,
        'reverse_input_channels': args.reverse_input_channels,
        'topk': args.topk,
        'path_to_labels': args.labels
    }

    # merge config from cli and from file
    config = {**user_config, **model_configuration}
    model = Classification(NetworkInfo(launcher.get_input_layers(), launcher.get_output_layers()), config)
    model.log_layers_info()

    # 3 create handler-executor and visualizer
    executor = SyncExecutor(model, launcher)
    visualizer = ClassificationVisualizer()


    # 4 Inference part
    next_frame_id = 0
    video_writer = cv2.VideoWriter()
    ESC_KEY = 27
    key = -1
    # Metrics for inference part
    performance_metrics = PerformanceMetrics()
    while True:
        # Get new image/frame
        frame = cap.read()
        if frame is None:
            if next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            break
        if next_frame_id == 0:
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(), (frame.shape[1], frame.shape[0])):
                raise RuntimeError("Can't open video writer")

        # Inference current frame
        start_time = perf_counter()
        classifications, _ = executor.run(frame)
        if args.raw_output_message:
            visualizer.print_raw_results(classifications, next_frame_id)
        performance_metrics.update(start_time, frame)

        frame = visualizer.draw_labels(frame, classifications)
        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id <= args.output_limit-1):
            video_writer.write(frame)

        # Visualization
        if not args.no_show:
            cv2.imshow('Classification Results', frame)
            key = cv2.waitKey(1)
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break


        next_frame_id += 1

    performance_metrics.log_total()


if __name__ == '__main__':
    sys.exit(main() or 0)
