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
        self.MSSK_MEAN = 0
        self.MSSK_STD = 1
        self.MSSK_SKEW = 2
        self.MSSK_KURT = 3
        self.MSSK_SOBEL_MEAN = 4
        self.MSSK_SOBEL_STD = 5
        self.MSSK_SOBEL_SKEW = 6
        self.MSSK_SOBEL_KURT = 7
        self.MSSK_LAP_MEAN = 8
        self.MSSK_LAP_STD = 9
        self.MSSK_LAP_SKEW = 10
        self.MSSK_LAP_KURT = 11

        self.videoPath = videoPath
        self.videoW = videoW
        self.videoH = videoH
        self.yuvForm = yuvForm
        self.subimg_h = 128
        self.subimg_w = 128
        self.csvPath = csvPath
        self.yuvFrames = yuvio.mimread(
            self.videoPath, self.videoW, self.videoH, self.yuvForm)

        self.FrameNum = self.yuvFrames.__len__() - 1
        # self.FrameNum = 100
        self.CTUNuminFrame = (int(videoH / 128))*(int(videoW / 128))
        CTUnum = (int(videoH / 128))*(int(videoW / 128)) * \
            (self.FrameNum)
        self.mssk = np.zeros((CTUnum, 12))

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

    def calculate_MSSK(self, FrameCTUs):
        '''
        Calculate MSSK in loop
        '''
        FrameMssk = np.zeros((FrameCTUs.shape[0], 4))
        for CTUid in range(FrameCTUs.shape[0]):
            row = FrameCTUs[CTUid]
            series = pd.Series(row)
            # n = len(row)

            # mean = sum(row) / n
            # variance = sum((x - mean) ** 2 for x in row) / n
            # std = variance ** 0.5
            # # Add a small number to the denominator to avoid division by zero
            # skewness = np.sum((row - mean) ** 3) / ((n * std ** 3) + 1e-9)
            # kurtosis = np.sum((row - mean) ** 4) / ((n * std ** 4) + 1e-9)

            mean = np.mean(row)
            std = np.std(row)
            skewness = series.skew()
            kurtosis = series.kurt()
            # skewness = np.sum((row - mean) ** 3) / ((n * std ** 3) + 1e-9)
            # kurtosis = np.sum((row - mean) ** 4) / ((n * std ** 4) + 1e-9)

            FrameMssk[CTUid, 0] = mean
            FrameMssk[CTUid, 1] = std
            FrameMssk[CTUid, 2] = skewness
            FrameMssk[CTUid, 3] = kurtosis
        return FrameMssk

    # need to change::
    def videoProcess(self):
        '''
        Extract video frames to array 
        And put this as attribute in this class 
        '''
        FrameNum = self.FrameNum

        for frame in range(FrameNum):
            print(f"Processing frame::{frame}")
            yuv_frame = self.yuvFrames.pop()
            y = yuv_frame.y
            sobely = self.Sobel_Filter(y)
            lapy = self.Laplacian_Filter(y)
            ##################### Divide frame and produce 2d matrix ############################

            NomalFrameCTUs = self.reshapeSplit(self.imageSplit(y))
            SobelFrameCTUs = self.reshapeSplit(self.imageSplit(sobely))
            LapFrameCTUs = self.reshapeSplit(self.imageSplit(lapy))

            NomalFrameMssk = self.calculate_MSSK(NomalFrameCTUs)
            SobelFrameMssk = self.calculate_MSSK(SobelFrameCTUs)
            LapFrameMssk = self.calculate_MSSK(LapFrameCTUs)

            ########## ######################

            for ctu in range(self.CTUNuminFrame):
                mssk_ID = frame*self.CTUNuminFrame + ctu
                ##################### ###################
                # print(frame, ctu, mssk_ID)
                self.mssk[mssk_ID, self.MSSK_MEAN] = NomalFrameMssk[ctu, 0]
                self.mssk[mssk_ID, self.MSSK_STD] = NomalFrameMssk[ctu, 1]
                self.mssk[mssk_ID, self.MSSK_SKEW] = NomalFrameMssk[ctu, 2]
                self.mssk[mssk_ID, self.MSSK_KURT] = NomalFrameMssk[ctu, 3]

                self.mssk[mssk_ID,
                          self.MSSK_SOBEL_MEAN] = SobelFrameMssk[ctu, 0]
                self.mssk[mssk_ID,
                          self.MSSK_SOBEL_STD] = SobelFrameMssk[ctu, 1]
                self.mssk[mssk_ID,
                          self.MSSK_SOBEL_SKEW] = SobelFrameMssk[ctu, 2]
                self.mssk[mssk_ID,
                          self.MSSK_SOBEL_KURT] = SobelFrameMssk[ctu, 3]

                self.mssk[mssk_ID,
                          self.MSSK_LAP_MEAN] = LapFrameMssk[ctu, 0]
                self.mssk[mssk_ID,
                          self.MSSK_LAP_STD] = LapFrameMssk[ctu, 1]
                self.mssk[mssk_ID,
                          self.MSSK_LAP_SKEW] = LapFrameMssk[ctu, 2]
                self.mssk[mssk_ID,
                          self.MSSK_LAP_KURT] = LapFrameMssk[ctu, 3]

        df = pd.DataFrame(self.mssk)
        return df

    def writeToCsvFile(self, mssk):
        # mssk = pd.DataFrame()
        mssk.to_csv(self.csvPath)


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
    data = vtc.videoProcess()
    print(data)
    vtc.writeToCsvFile(data)
    # print(vtc.mssk.shape)
    # print(vtc.video_csv)
