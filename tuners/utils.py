import numpy as np
import cv2 as cv
import json

def plot(shape, imgs, title = ''):
    cv.namedWindow(title, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

    lines = []
    for i in range(shape[0]):
        lines.append(np.hstack(imgs[i * shape[0]:i * shape[0] + shape[1]]))

    img = np.vstack(lines)
    cv.imshow(title, img)


def convert_to_dict(obj):
    """
    A function takes in a custom object and returns a dictionary representation of the object.
    This dict representation includes meta data such as the object's module and class names.
    Original code: https://gist.github.com/ardenn/76aa5653245388519a2edb690d8ed7ba#file-json_convert_to_dict-py
    """

    #  Populate the dictionary with object meta data 
    obj_dict = {
        "__class__": obj.__class__.__name__,
        "__module__": obj.__module__
    }

    #  Populate the dictionary with object properties
    obj_dict.update(obj.__dict__)

    return obj_dict


def dict_to_obj(our_dict):
    """
    Function that takes in a dict and returns a custom object associated with the dict.
    This function makes use of the "__module__" and "__class__" metadata in the dictionary
    to know which object type to create.
    Original code: https://gist.github.com/ardenn/30f94f57876a70832a5c960fd4742d89#file-json_dict_to_obj-py
    """
    if "__class__" in our_dict:
        # Pop ensures we remove metadata from the dict to leave only the instance arguments
        class_name = our_dict.pop("__class__")

        # Get the module name from the dict and import it
        module_name = our_dict.pop("__module__")
        
        # We use the built in __import__ function since the module name is not yet known at runtime
        module = __import__(module_name)

        # Get the class from the module
        class_ = getattr(module,class_name)
        
        # Use dictionary unpacking to initialize the object
        obj = class_(**our_dict)
    else:
        obj = our_dict
    return obj

def load_cfg(file_path):
    try:
        with open(file_path, 'r') as cfg_file:
            return json.load(
                fp = cfg_file,
                object_hook = dict_to_obj,
            )
    except FileNotFoundError:
        return None

def save_cfg(file_path, obj):
    with open(file_path, 'w') as cfg_file:
        json.dump(
            obj = obj,
            fp = cfg_file,
            default = convert_to_dict,
        )