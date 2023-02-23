import numpy as np
import cv2 as cv
import pandas as pd
import yuvio
# from scipy.stats import skew, kurtosis
# import time


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

        # self.FrameNum = self.yuvFrames.__len__() - 1
        self.FrameNum = 100
        self.resultMSSK = np.empty((0, 12))
        self.video_csv = "No value now"

        print("init sucess!")

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

    def split_re_MSSK(self, Frame):
        '''
        Calculate each frames MSSK in normal, Sobel, and Laplacian
        '''
        NormalImage = Frame
        SobelImage = self.Sobel_Filter(Frame)
        LapImage = self.Laplacian_Filter(Frame)

        # Define the size of the smaller arrays
        sub_array_size = self.subimg_h

        # Calculate the number of smaller arrays that can fit in the larger array
        num_sub_arrays_x = self.videoH // sub_array_size
        num_sub_arrays_y = self.videoW // sub_array_size

        for i in range(num_sub_arrays_x):
            for j in range(num_sub_arrays_y):
                # calculate MSSK for each CTU in nomal n_mssk, Soebl s_mssk and Laplacian l_mssk array
                CTU_N = NormalImage[i*sub_array_size:(i+1)*sub_array_size,
                                    j*sub_array_size:(j+1)*sub_array_size].flatten()
                n_mean, n_std, n_skew, n_kurt = self.calculateMSSK(CTU_N)

                CTU_S = SobelImage[i*sub_array_size:(i+1)*sub_array_size,
                                   j*sub_array_size:(j+1)*sub_array_size].flatten()
                s_mean, s_std, s_skew, s_kurt = self.calculateMSSK(CTU_S)

                CTU_L = LapImage[i*sub_array_size:(i+1)*sub_array_size,
                                 j*sub_array_size:(j+1)*sub_array_size].flatten()
                l_mean, l_std, l_skew, l_kurt = self.calculateMSSK(CTU_L)

                ctuInfo = [n_mean, n_std, n_skew, n_kurt, s_mean, s_std,
                           s_skew, s_kurt, l_mean, l_std, l_skew, l_kurt]

                self.resultMSSK = np.vstack([self.resultMSSK, ctuInfo])

    def calculateMSSK(self, CTU):
        series = pd.Series(CTU)
        ctu_mean = CTU.mean()
        ctu_std = CTU.std()
        ctu_skew = series.skew()
        ctu_kurt = series.kurtosis()
        return ctu_mean, ctu_std, ctu_skew, ctu_kurt

    def videoProcess(self):
        '''
        Extract video frames to array 
        And put this as attribute in this class 
        '''
        FrameNum = self.FrameNum
        for frame in range(FrameNum):
            # start = time.time()

            print(f"Processing frame::{frame}")
            yuv_frame = self.yuvFrames.pop().y
            self.split_re_MSSK(yuv_frame)

        self.resultMSSK = pd.DataFrame(self.resultMSSK)


if __name__ == '__main__':
    videoPath = r"video/BasketballDrive_1920x1080_50.yuv"
    videoW = 1920
    videoH = 1080

    # videoPath = r"video/BasketballPass_416x240_50.yuv"
    # videoW = 416
    # videoH = 240

    yuvForm = "yuv420p"
    csvPath = r"csvFile/test.csv"

    vtc = VideoToCsv(videoPath, videoW, videoH, yuvForm, csvPath)
    vtc.videoProcess()
    print(vtc.resultMSSK)
    # print(data)
