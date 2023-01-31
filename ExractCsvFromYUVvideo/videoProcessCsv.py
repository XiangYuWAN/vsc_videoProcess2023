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

        self.video_NormalArray = "No value now"
        self.video_SobelArray = "No value now"
        self.video_LapArray = "No value now"

        self.video_csv = "No value now"

    def ReadFirstFrameY_of_Video(self):
        '''
        Read first Frame Y of video  
        '''
        image = yuvio.imread(self.videoPath, self.videoW,
                             self.videoH, self.yuvForm)
        y = image.y
        return y

    def imageSplit(self, image, img_h, img_w, subimg_h, subimg_w):
        '''
        Split a frame(image) to a 4D array
        '''
        array = np.lib.stride_tricks.as_strided(
            image,
            shape=(img_h // subimg_h, img_w // subimg_w,
                   subimg_h, subimg_w),  # rows, cols
            strides=image.itemsize * \
            np.array([subimg_h * img_w, subimg_w, img_w, 1])
        )
        return array

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

    def Array2DTo_MSSK(self, data):
        '''
        Transfer a 2D array to a csv with Mean Std Skew Kurt
        '''
        df1 = pd.DataFrame(data)
        mean = df1.mean(axis=1)
        std = df1.std(axis=1)
        skew = df1.skew(axis=1)
        kurt = df1.kurt(axis=1)

        # print("mean:\n", mean)
        # print("std:\n", std)
        # print("skew:\n", skew)
        # print("kurt:\n", kurt)

        result = pd.concat([mean, std, skew, kurt], axis=1)
        result.columns = ['mean', 'std', 'skew', 'kurt']
        # print(result)
        # result.to_csv(path, 'w')
        return result

    def getVideoArray(self):
        '''
        Extract video frames to array 
        And put this as attribute in this class 
        '''
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

    def getSobel_VideoArray(self):
        '''
        Extract video frames to array, But these frames are applied SobelF  
        '''
        yuv_frames = self.yuvFrames.copy()
        # img_h = self.videoH
        # img_w = self.videoW
        # subimg_h, subimg_w = self.subimg_h, self.subimg_w

        # 初始化 initialization
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        grad = self.Sobel_Filter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = self.imageSplit(grad)
        ra0 = self.reshapeSplit(array)

        length = yuv_frames.__len__() - 1
        for i in range(length):
            yuv_frame = yuv_frames.pop()
            y = yuv_frame.y
            grad = self.Sobel_Filter(y)
            ##################### Divide frame and produce 2d matrix ############################
            array = self.imageSplit(grad)
            ra = self.reshapeSplit(array)
            ra0 = np.row_stack((ra0, ra))

        self.video_SobelArray = ra0
        return ra0

    def getLaplacian_VideoArray(self):
        '''
        Extract video Frame to array, but these frames are applied LaplacianFilter  
        '''
        yuv_frames = self.yuvFrames.copy()
        # img_h = self.videoH
        # img_w = self.videoW
        # subimg_h, subimg_w = self.subimg_h, self.subimg_w

        # 初始化 initialization
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        grad = self.Laplacian_Filter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = self.imageSplit(grad)
        ra0 = self.reshapeSplit(array)

        length = yuv_frames.__len__() - 1
        for i in range(length):
            yuv_frame = yuv_frames.pop()
            y = yuv_frame.y
            grad = self.Laplacian_Filter(y)
            ##################### Divide frame and produce 2d matrix ############################
            array = self.imageSplit(grad)
            ra = self.reshapeSplit(array)
            ra0 = np.row_stack((ra0, ra))

        self.video_LapArray = ra0
        return ra0

    def merge_MSSK_Array_together(self):
        '''
        Combine normal array, sobel array, lap array together  
        '''
        # calculate each CU's mean, std, skew and kurt
        n_result = self.Array2DTo_MSSK(self.video_NormalArray)
        s_result = self.Array2DTo_MSSK(self.video_SobelArray)
        l_result = self.Array2DTo_MSSK(self.video_LapArray)

        # change column name
        s_result.columns = ['Sobel_Mean',
                            'Sobel_std', 'Sobel_Skew', 'Sobel_Kurt']
        l_result.columns = ['Lap_Mean', 'Lap_std', 'Lap_Skew', 'Lap_Kurt']

        #  merge all these data frame together and produce a csv file
        mssk_result = pd.concat([n_result, s_result, l_result], axis=1)
        return mssk_result

    def videoProcess(self):
        '''
        Start the whole video process to get dataframe result
        '''
        # get 2D arrays of video with normal, after sobel, after lap, each row of arrays stand for a CU
        self.getVideoArray()
        self.getSobel_VideoArray()
        self.getLaplacian_VideoArray()

        # merge together and return
        mssk_result = self.merge_MSSK_Array_together()
        self.video_csv = mssk_result
        print(f"The result Csv table of this video is:  \n")
        print(self.video_csv)
        return mssk_result

    def writeToCsvFile(self, mssk):
        # mssk = pd.DataFrame()
        mssk.to_csv(self.csvPath)

    def writeToCsvFile(self):
        # mssk = pd.DataFrame()
        mssk = self.video_csv
        mssk.to_csv(self.csvPath)


if __name__ == '__main__':
    videoPath = r"ExractCsvFromYUVvideo\video\BasketballPass_416x240_50.yuv"
    videoW = 416
    videoH = 240
    yuvForm = "yuv420p"
    csvPath = r"ExractCsvFromYUVvideo\csvFile\basketballCU.csv"

    vtc = VideoToCsv(videoPath, videoW, videoH, yuvForm, csvPath)
    # print(vtc.yuvFrames)
    vtc.videoProcess()
    # print(vtc.video_csv)
