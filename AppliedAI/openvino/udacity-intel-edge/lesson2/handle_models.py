import cv2
import numpy as np

# print(output.keys(), output.shape)
def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # The input image shape is say (750, 1000, 3)
    # output is a dictionary with two items (blobs)"Mconv7_stage2_L1", "Mconv7_stage2_L2"
    # Extract only the second blob output (keypoint heatmaps)
    # which is of shape (1, 19, 32, 57)
    # 19 images of dimensions (32, 57). Each image corresponds to a keypoint heatmap.
    heatmaps = output['Mconv7_stage2_L2']
    # Resize the heatmap back to the size of the input
    # create a 19 x 750 x 1000 array
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    # Iterate through and re-size each of the heatmaps to size of the input image
    # also reverse the input H x W dimensions to W x H because cv2.resize
    # accepts horizontal(fx), vertical(fy)
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])
        
    return out_heatmap


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # input shape : input image in the format [BxCxHxW],
    # Extract only the first blob output (text/no text classification)
    #[1x2x192x320] - logits related to text/no-text classification for each pixel.
    text_classes = output['model/segm_logits/add']
    # TODO 2: Resize this output back to the size of the input
    # 2 x 192 x 320 -> 2 x H x W
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])
    return out_text


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # input :shape: [1x3x72x72] - An input image in following format [1xCxHxW]
    # Get the argmax of the "color" output
    #"color", shape: [1, 7, 1, 1] - Softmax output across seven color classes [white, gray, yellow, red, green, blue, black]
    color = output['color'].flatten()
    color_pred = np.argmax(color)
    # Get the argmax of the "type" output
    # "type", shape: [1, 4, 1, 1] - Softmax output across four type classes [car, bus, truck, van]
    car_type = output['type'].flatten()
    type_pred = np.argmax(car_type)
  
    return color_pred, type_pred


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
