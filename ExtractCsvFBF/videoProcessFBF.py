import numpy as np
import cv2 as cv
import pandas as pd
import yuvio


class VideoToCsv:
    '''
    This is the video Process class to extract important inforamtion
    write to csv
    '''

    def __init__(self, videoPath, videoW, videoH, yuvForm, csvPath):
        '''
        get inisal information of video path, frames, bigest CU...
        '''
        self.videoPath = videoPath
        self.videoW = videoW
        self.videoH = videoH
        self.yuvForm = yuvForm
        self.subimg_h = 128
        self.subimg_w = 128
        self.csvPath = csvPath
        self.yuvFrames = yuvio.mimread(
            self.videoPath, self.videoW, self.videoH, self.yuvForm)

        CUnum = (int(videoH / 128))*(int(videoW / 128)) * \
            (self.yuvFrames.__len__())
        self.mssk = np.zeros((CUnum, 12))
        self.video_NormalArray = "No value now"
        self.video_SobelArray = "No value now"
        self.video_LapArray = "No value now"

        self.video_csv = "No value now"
        print("init sucess!")

    def imageSplit(self, image):
        '''
        Split a frame(image) to a 4D array
        '''
        img_h = self.videoH
        img_w = self.videoW
        subimg_h = self.subimg_h
        subimg_w = self.subimg_w

        array = np.lib.stride_tricks.as_strided(
            image,
            shape=(img_h // subimg_h, img_w // subimg_w,
                   subimg_h, subimg_w),  # rows, cols
            strides=image.itemsize * \
            np.array([subimg_h * img_w, subimg_w, img_w, 1])
        )
        return array

    def reshapeSplit(self, array):
        '''
        Transfer a 4D array to a 2D array  
        '''
        imgCU_h, imgCU_w, subimg_h, subimg_w = array.shape
        # print(imgCU_h, imgCU_w, subimg_h, subimg_w)
        reshapedArray = np.reshape(
            array, (imgCU_h * imgCU_w, subimg_h * subimg_w))
        return reshapedArray

    def Sobel_Filter(self, image):
        '''
        Using Sobel filter to an image, 
        also using GaussianBlur to remove noise before Sobel
        '''
        ddepth = cv.CV_16S
        # GaussianBlur to remove noise
        image = cv.GaussianBlur(image, (3, 3), 0)
        Sx = cv.Sobel(image, ddepth, 1, 0)
        Sy = cv.Sobel(image, ddepth, 0, 1)
        abs_gradX = cv.convertScaleAbs(Sx)
        abs_grady = cv.convertScaleAbs(Sy)
        grad = cv.addWeighted(abs_gradX, 0.5, abs_grady, 0.5, 0)
        return grad

    def Laplacian_Filter(self, image):
        '''
        Using laplacian filter to an image    
        '''
        ddepth = cv.CV_16S
        kernal_size = 3
        image = cv.GaussianBlur(image, (3, 3), 0)
        dst = cv.Laplacian(image, ddepth, ksize=kernal_size)
        abs_dst = cv.convertScaleAbs(dst)
        return abs_dst

    # need to change::
    def videoProcess(self):
        '''
        Extract video frames to array 
        And put this as attribute in this class 
        '''
        yuv_frames = self.yuvFrames.copy()
        length = yuv_frames.__len__() - 1

        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        array = self.imageSplit(y)
        ra0 = self.reshapeSplit(array)

        if ():
            yuv_frames = self.yuvFrames.copy()
            # img_h = self.videoH
            # img_w = self.videoW
            # subimg_h, subimg_w = self.subimg_h, self.subimg_w

            # 初始化 initialization
            yuv_frame = yuv_frames.pop()
            y = yuv_frame.y
            # grad = sobelFilter.SobelFilter(y)
            ##################### Divide frame and produce 2d matrix ############################
            array = self.imageSplit(y)
            ra0 = self.reshapeSplit(array)
            length = yuv_frames.__len__() - 1
            for i in range(length):
                yuv_frame = yuv_frames.pop()
                y = yuv_frame.y
                # grad = sobelFilter.SobelFilter(y)
                ##################### Divide frame and produce 2d matrix ############################
                array = self.imageSplit(y)
                ra = self.reshapeSplit(array)
                ra0 = np.row_stack((ra0, ra))

            self.video_NormalArray = ra0
            return ra0


if __name__ == '__main__':
    videoPath = r"video/BasketballDrive_1920x1080_50.yuv"
    videoW = 1920
    videoH = 1080
    yuvForm = "yuv420p"
    csvPath = r"csvFile/basketballCU.csv"

    vtc = VideoToCsv(videoPath, videoW, videoH, yuvForm, csvPath)

    # print(vtc.mssk.shape)
    # print(vtc.video_csv)
