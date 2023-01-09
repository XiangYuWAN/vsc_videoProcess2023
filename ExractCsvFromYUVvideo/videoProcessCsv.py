import numpy as np
import cv2 as cv
import pandas as pd
import yuvio


class VideoToCsv:
    videoPath = None
    videoW = None
    videoH = None
    yuvForm = None
