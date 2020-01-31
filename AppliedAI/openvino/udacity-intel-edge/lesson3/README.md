# assuming you have ssd mobilenet and classroom.mp4 downloaded you can run the
# code using the following command.

python3 app.py -m public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -o out-modified.h264 -i classroom.mp4

#Also run the following to get usage information

python3 app.py -h

---------------
usage: Run inference on an input video [-h] -m M [-i I] [-d D] [-ct CT] [-o O]
                                       [-t T]

required arguments:
  -m M    The location of the model XML file

optional arguments:
  -i I    The location of the input file
  -d D    The device name, if not 'CPU'
  -ct CT  The confidence threshold to use with the bounding boxes
  -o O    The output file path
  -t T    The input type VIDEO/IMAGE

---------------
