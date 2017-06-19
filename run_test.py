"""
MIT License

Copyright (c) 2017 Sameem Zahoor Taray

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python2.7
from    __future__ import print_function
from    math import ceil
from    scipy import misc

from flow_holder import FlowHolder
from flow_io import *

import os, sys, numpy as np
import argparse
import caffe
import tempfile
import cv2
import time
import matplotlib.pyplot as plt



if __name__ == '__main__':
    #Setting some parameters

    caffemodel  = '/home/taraysz/Desktop/flownet2-master/models/FlowNet2-css-ft-sd/FlowNet2-css-ft-sd_weights.caffemodel.h5'
    deployproto = '/home/taraysz/Desktop/flownet2-master/models/FlowNet2-css-ft-sd/FlowNet2-css-ft-sd_deploy.prototxt.template'
    gpu         = 0
    num_blobs = 2
    input_data = []
    webcam_frame_width = 640
    webcam_frame_height = 480
    caffe_logging = 0
    framenum = 0
    max_flow_size = 10
    flow_holder = FlowHolder(max_flow_size)
    if(not os.path.exists(caffemodel)): raise BaseException('caffemodel does not exist: ' + caffemodel)
    if(not os.path.exists(deployproto)): raise BaseException('deploy-proto does not exist: ' + deployproto)
    
    # Load the caffe model
    width = 640
    height = 480
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
    proto = open(deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))

        tmp.write(line)

    tmp.flush()

    if not caffe_logging:
        caffe.set_logging_disabled()

    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, caffemodel, caffe.TEST)

    # input_dict = {}
    # for blob_idx in range(num_blobs):
    #     input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    ## Setup the Plots
    # plt.ion()
    # fig = plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k',  tight_layout=True)
    # framerate_textbox = fig.text(0.15, 0.1,'', horizontalalignment='left', verticalalignment='center', fontsize=16, backgroundcolor=(0.85,0.85,0.85))
    # framerate_textbox_actual = fig.text(0.15, 0.05,'', horizontalalignment='left', verticalalignment='center', fontsize=16, backgroundcolor=(0.85,0.65,0.85))
    zero_img = np.zeros((480,640,3), dtype=np.uint8)
    # axis1 = plt.subplot(121)
    # axis1.set_title('Last frame for Change', fontsize=18)
    # axis1.set_axis_off()
    # ims1 = axis1.imshow(zero_img[:,:,0], vmin=0, vmax=255)
    

    # axis2 = plt.subplot(122)
    # ims2 = axis2.imshow(zero_img[:,:,0], vmin=0, vmax=255)
    # axis2.set_title('Segmentation Generated', fontsize=18)
    # axis2.set_axis_off()
    cam = cv2.VideoCapture(0)
    cam.set(4, webcam_frame_width) #set frame width to 640
    
    segmentation = zero_img[:,:,0]
    if cam.isOpened(): # try to get the first frame
        rval, prev = cam.read()
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        im1_g = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        im1_g = cv2.GaussianBlur(im1_g,(5,5),0)
    else:
        rval = False
    # Preallocate  input_data list and input_data dict for better performance
    input_data = []
    input_dict = {}
    input_data.append(prev[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    input_data.append(prev[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    tstart1 = time.time()
    for framenum in range(0,max_flow_size):
        input_data[0] = prev[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        rval, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im2_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        im2_g = cv2.GaussianBlur(im2_g,(5,5),0)
        input_data[1] = frame[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
        # Do forward pass
        net.forward(**input_dict)
        for name in net.blobs:
            blob = net.blobs[name]
        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        # change = flow_holder.processFlow(blob)
        change = flow_holder.processFlowResidue(im1_g, im2_g, blob)
    
    total_time = 0.0
    total_time_actual = 0.0
    framenum   = 0
    
    for temp in range(1, 1250):
        tstart = time.time()
        framenum += 1
        # Prepare input data in input_data and input_dict
        input_data[0] = prev[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        
        rval, frame = cam.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im2_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        im2_g = cv2.GaussianBlur(im2_g,(5,5),0)
        input_data[1] = frame[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
        # Do forward pass and save the current optical flow
        net.forward(**input_dict)
        for name in net.blobs:
            blob = net.blobs[name]
        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        curr_img = flow_to_color(blob)
        # cv2.imshow("Current Frame", curr_img)
        
        change = flow_holder.processFlowResidueMixed(im1_g, im2_g, blob)
        
        # Replace with if change
        if change:
            flow_holder.composeFlowResidue()    
            composed_flow   = flow_holder.getComposedFlow()
            # composed_residual = flow_holder.getComposedResidual()
            # print('Composed Residual mean = ', composed_residual.mean())
            composed_image  = flow_to_color(composed_flow)
            _segmentation   = segment_flow(composed_image)
            # change = flow_holder.likelyhoodTest(_segmentation)
            flow_holder.likelyhoodTestPropogated(_segmentation)
            if change:
                segmentation = _segmentation
                # cv2.imshow("Composed Flow", composed_residual)
                # tstart = time.time()
                # labels = flow_holder.segmentFlow(prev)
                # ax1.imshow(composed_image)
        
                # tactual           = time.time() - tstart  
                # total_time_actual += tactual

                # ims1.set_data(flow_holder.getResidual())
                # ims2.set_data(segmentation)
                # plt.pause(0.00001)
                # fig.canvas.flush_events()
                # print("Time for rendering = " + str(tend))
        tend                = time.time() - tstart
        total_time          += tend
        cv2.imshow("Segmentation", segmentation)
        cv2.imshow("Frame", frame)

        prev                = frame.copy()
        im1_g               = im2_g.copy()
        
        cv2.waitKey(1)
        
        # framerate_actual    = 1/total_time_actual * framenum
        framerate           = 1/total_time  * framenum
        print('Framerate =  %0.2f FPS' % framerate)
        # print(str(framerate) + " FPS")
                # framerate_textbox_actual.set_text('Framerate : ' + "{:.2f}".format(framerate)  + 'FPS')
                # framerate_textbox.set_text('Actual Framerate : ' + "{:.2f}".format(framerate_actual)  + 'FPS')

    # tend = time.time() - tstart1
    # framerate = 1/tend * framenum
    # print("Average Frame Rate = " + str(framerate))
    cv2.destroyAllWindows()