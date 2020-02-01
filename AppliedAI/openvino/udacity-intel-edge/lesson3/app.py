#This code is enhancement of Lesson 3 LAB code from Udacity Intel Edge-AI
# Foundation Course.
# The code has been customized a bit to run on RPi + NCS2.
# TODO: more enhancements needed to the code as some of it is written as
# experimental code.

import argparse
import cv2
from inference import Network
import numpy as np
from imutils.video import FPS
import threading
import time
from concurrent.futures import ThreadPoolExecutor


INPUT_STREAM = "/home/pi/mydemo/test_video.mp4"
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ###  Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ct_desc = "The confidence threshold to use with the bounding boxes"
    o_desc = "The output file path"
    t_desc = "The input type VIDEO/IMAGE"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='MYRIAD')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    optional.add_argument("-o", help=o_desc, default='out.h264')
    optional.add_argument("-t", help=t_desc, default='VIDEO')
    args = parser.parse_args()

    return args

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# classes for mobilenet
# TODO: not sure if these classes are correct.
# Need to externalize this to a config/mapping file
CLASSES = ["background", "person", "bicycle", "car", "boat",
        "aeroplane", "bus", "bird", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "bottle", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

# use this for person-vehicle-bike-detection-crossroad.xml model
#CLASSES = ["person", "bicycle", "car"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_boxes(frame, detections, args, w, h):
    '''
    Draw bounding boxes onto the frame.
    '''
    # loop over the detections
    # ex : 1x1x100x7 so detections.shape[2] will be 100
    # the class of the object, the confidence, and two corners (made of xmin, ymin, xmax, and ymax) that make up the bounding box, in that order.
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
	# the prediction
        confidence = detections[0, 0, i, 2] 
        # filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
        if confidence >= args.ct:
            # extract the index of the class label from the
            # detections, then compute the (x, y)-coordinates of
	    # the bounding box for the object
            # index 1 has the detected class
            idx = int(detections[0, 0, i, 1])
            # index 3 to 7 have the bounding box corners
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if idx > len(CLASSES):
                continue
            # draw the prediction on top of the frame
            #print('class ={}'.format(CLASSES[idx]));
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            #cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 4)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 4)
            # calculate the y-coordinate used to write the label on the
            # frame depending on the bounding box coordinate
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return frame


def preprocessing(input_image, height, width):
    '''
    Given an input image, network input size (height and width):
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    # change data layout from HxWxC to CxHxW
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def infer_on_image(args):
    print('INFER ON IMAGE')
    # Convert the args for confidence
    args.ct = float(args.ct)
    
    ### Initialize the Inference Engine
    plugin = Network()
    ### Load the network model into the IE
    plugin.load_model(args.m, args.d)
    net_input_shape = plugin.get_input_shape()
    # Read the input image
    image = cv2.imread(args.i)
    h, w = net_input_shape[2], net_input_shape[3] 

    ### Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    ###  Perform inference on the frame
    plugin.async_inference(preprocessed_image)
    ###  Get the output of inference
    if plugin.wait() == 0:
        output = plugin.extract_output()

    image = draw_boxes(image, output, args, w, h)
    cv2.imwrite(args.o, image)


def infer_on_video(args):
    print('INFER ON VIDEO')
    # Convert the args for confidence
    args.ct = float(args.ct)
    
    ###  Initialize the Inference Engine
    plugin = Network()
    ###  Load the network model into the IE
    plugin.load_model(args.m, args.d)
    net_input_shape = plugin.get_input_shape()
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    #out = cv2.VideoWriter('/home/pi/mydemo/out.mp4', 0x00000021, 30, (width,height))
    out_file = args.o
    #out = cv2.VideoWriter('out_file', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height))
    #out = cv2.VideoWriter('out_file', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('H','2','6','4')  , 30, (width,height))
    
    #start the FPS (frames per second recorder)
    fps = FPS().start()

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        #key_pressed = cv2.waitKey(60)

        ###  Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        ###  Perform inference on the frame
        plugin.async_inference(p_frame)
        ###  Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
        ###  Update the frame to include detected bounding boxes
        frame = draw_boxes(frame, result, args, width, height)
        # Write out the frame
        out.write(frame)

        #update the FPS counter
        fps.update()
        # Break if escape key pressed
        #if key_pressed == 27:
        #    break

    # Release the out writer, capture, and destroy any OpenCV windows
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def process_frame(condition, plugin, frame, net_input_shape, req_id, args, width, height):
    ###  Pre-process the frame
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    ###  submit frame to inference engine 
    plugin.async_inference(p_frame)
    ###  Get the output of inference
    condition.acquire()
    try:
        if plugin.wait(req_id) == 0:
            result = plugin.extract_output()
            condition.release()
            ###  Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            return frame
    except:    
        condition.release()
        return None

def infer_on_video_parallel(args):
    print('INFER ON VIDEO PARALLEL')
    # Convert the args for confidence
    args.ct = float(args.ct)
    
    executor = ThreadPoolExecutor(max_workers=2)
    condition = threading.Condition()
    NoneType = type(None)
    ###  Initialize the Inference Engine
    plugin = Network()
    ###  Load the network model into the IE
    plugin.load_model(args.m, args.d)
    net_input_shape = plugin.get_input_shape()
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    out_file = args.o
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('H','2','6','4')  , 30, (width,height))
    
    #start the FPS (frames per second recorder)
    fps = FPS().start()
    # Process frames until the video ends, or process is exited
    while cap.isOpened():

        #key_pressed = cv2.waitKey(60)

        # Read the next frame
        flag0, frame0 = cap.read()
        if not flag0:
            break
        future0 = executor.submit(process_frame, condition, plugin, frame0, net_input_shape, 0, args, width, height)

        flag1, frame1 = cap.read()
        if not flag1:
            break
        future1 = executor.submit(process_frame, condition, plugin, frame1, net_input_shape, 1, args, width, height)

        result = future0.result() 
        if type(result) == NoneType:
            print('none result frame0');
            break
        else:
            # Write out the frame
            fps.update()
            out.write(result)

        result = future1.result() 
        if type(result) == NoneType:
            print('none result frame1');
            break
        else:
            # Write out the frame
            fps.update()
            out.write(result)

        # Break if escape key pressed
        #if key_pressed == 27:
        #    break

    # Release the out writer, capture, and destroy any OpenCV windows
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    type = args.t
    if type == 'VIDEO':
        #infer_on_video(args)
        infer_on_video_parallel(args)
    else:
        infer_on_image(args)


if __name__ == "__main__":
    main()
