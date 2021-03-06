#This code is enhancement of Lesson 3 LAB code from Udacity Intel Edge-AI
# Foundation Course.
# The code has been customized a bit to run on RPi + NCS2.

'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"


        # Initialize the plugin
        #self.plugin = IECore()
        self.plugin = IEPlugin(device='MYRIAD')

        # Add a CPU extension, if applicable
        #if cpu_extension and "CPU" in device:
        #    self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
      
        # Load the IENetwork into the plugin
        #self.exec_network = self.plugin.load_network(self.network, device)
        self.exec_network = self.plugin.load(network=self.network, num_requests=2)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image, req_id=0):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        ### TODO: Start asynchronous inference
        self.exec_network.start_async(request_id=req_id,
            inputs={self.input_blob: image})
        return


    def wait(self, req_id=0):
        '''
        Checks the status of the inference request.
        '''
        ### TODO: Wait for the async request to be complete
        status = self.exec_network.requests[req_id].wait(-1)
        #print('status={s} for req_id={r}'.format(s=status, r=req_id))
        return status

    def wait_non_blocking(self, req_id=0):
        '''
        Checks the status of the inference request.
        '''
        ### TODO: Wait for the async request to be complete
        status = self.exec_network.requests[req_id].wait(0)
        return status


    def extract_output(self, req_id=0):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### TODO: Return the outputs of the network from the output_blob
        return self.exec_network.requests[req_id].outputs[self.output_blob]
