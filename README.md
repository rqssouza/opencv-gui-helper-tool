[//]: # (Image References)

[canny]: ./resources/canny.png
[road]: ./resources/road.png

# **canny-gui-parameter-tuner**

A GUI tool that helps finding parameters for computer vision algorithms

## **How to use**

There are two examples that shows how to use the tool.

### **Find Edges**

Tune parameters for the Canny edge finder algorithm

    python ./examples/canny_tuner.py <image-path>

Output:

    filterSize = 13
    threshold1 = 28
    threshold2 = 115

![alt text][canny]

### **Find Road Lanes**

Tune parameters for the road lines finder algorithm

    python ./examples/road_lines_tuner.py <image-path>

Output:

    top_cut = 435
    bottom_cut = 58
    base = 1095
    height = 336
    filterSize = 7
    threshold1 = 50
    threshold2 = 115
    points_ths = 10
    min_line_length = 6
    max_line_gap = 8

![alt text][road]

### **License**

Released under the [MIT License](LICENSE).
