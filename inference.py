#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.net_plugin = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request_handle = None

    def load_model(self, model, CPU_EXTENSION, DEVICE):
        self.plugin = IECore()
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0]+".bin"
        self.net_plugin = IENetwork(model=model_xml, weights= model_bin)
        
        ### TODO: Check for supported layers ###
        all_supported,unsupported_layers = is_all_layers_supported(self.plugin, self.net_plugin)
        if not all_supported:
            print("Unsupported layers found: {}".format(unsupported_layers))
            self.plugin.add_extension(CPU_EXTENSION, DEVICE)
            
        self.exec_network = self.plugin.load_network(self.net_plugin, DEVICE)
        ### TODO: Return the loaded inference plugin ###
        self.input_blob = next(iter(self.net_plugin.inputs))        
        self.output_blob = next(iter(self.net_plugin.outputs))

     
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        #print(self.net_plugin.inputs['image_info'])
        # print(self.input_blob)
        #print(self.net_plugin.inputs[self.input_blob].shape)
        input_shapes = {}
        for inpt in self.net_plugin.inputs:
           # print(self.net_plugin.inputs[inpt].shape)
            input_shapes[inpt] = (self.net_plugin.inputs[inpt].shape)
        #print(input_shapes)
        
        return input_shapes

    def exec_net(self, net_input, request_id):
        ### TODO: Start an asynchronous request ###
        
        self.infer_request_handle = self.exec_network.start_async(request_id = request_id, inputs = net_input)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return 

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.infer_request_handle.wait()
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        output = self.infer_request_handle.outputs[self.output_blob]
        return output

def is_all_layers_supported(engine, network):
    ### TODO check if all layers are supported
    ### return True if all supported, False otherwise
    all_supported = True
    supported_layers = engine.query_network(network=network, device_name="CPU")
    unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        all_supported = False
   
    return all_supported, unsupported_layers