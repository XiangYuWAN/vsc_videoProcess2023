import numpy as np
import cv2 as cv
import pandas as pd
import yuvio


def ReadFirstFrameY_of_Video(videoPath, videoW, videoH, yuvForm):
    '''
    Read first Frame Y of video  
    '''
    image = yuvio.imread(videoPath, videoW, videoH, yuvForm)
    y = image.y
    return y


def imageSplit(image, img_h, img_w, subimg_h, subimg_w):
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


def reshapeSplit(array):
    '''
    Transfer a 4D array to a 2D array  
    '''
    imgCU_h, imgCU_w, subimg_h, subimg_w = array.shape
    # print(imgCU_h, imgCU_w, subimg_h, subimg_w)
    reshapedArray = np.reshape(array, (imgCU_h * imgCU_w, subimg_h * subimg_w))
    return reshapedArray


def Sobel_Filter(image):
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


def Laplacian_Filter(image):
    '''
    Using laplacian filter to an image    
    '''
    ddepth = cv.CV_16S
    kernal_size = 3
    image = cv.GaussianBlur(image, (3, 3), 0)
    dst = cv.Laplacian(image, ddepth, ksize=kernal_size)
    abs_dst = cv.convertScaleAbs(dst)
    return abs_dst


def Array2DTo_MSSK(data):
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


def getVideoArray(videoPath, videoW, videoH, yuvForm):
    '''
    Extract video frames to array  
    '''
    yuv_frames = yuvio.mimread(videoPath, videoW, videoH, yuvForm)

    img_h = videoH
    img_w = videoW
    subimg_h, subimg_w = 128, 128

    # 初始化 initialization
    yuv_frame = yuv_frames.pop()
    y = yuv_frame.y
    # grad = sobelFilter.SobelFilter(y)
    ##################### Divide frame and produce 2d matrix ############################
    array = imageSplit(y, img_h, img_w, subimg_h, subimg_w)
    ra0 = reshapeSplit(array)

    length = yuv_frames.__len__()
    for i in range(length):
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        # grad = sobelFilter.SobelFilter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = imageSplit(y, img_h, img_w, subimg_h, subimg_w)
        ra = reshapeSplit(array)
        ra0 = np.row_stack((ra0, ra))

    return ra0


def getSobel_VideoArray(videoPath, videoW, videoH, yuvForm):
    '''
    Extract video frames to array, But these frames are applied SobelF  
    '''
    yuv_frames = yuvio.mimread(videoPath, videoW, videoH, yuvForm)

    img_h = videoH
    img_w = videoW
    subimg_h, subimg_w = 128, 128

    # 初始化 initialization
    yuv_frame = yuv_frames.pop()
    y = yuv_frame.y
    grad = Sobel_Filter(y)
    ##################### Divide frame and produce 2d matrix ############################
    array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
    ra0 = reshapeSplit(array)

    length = yuv_frames.__len__()
    for i in range(length):
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        grad = Sobel_Filter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
        ra = reshapeSplit(array)
        ra0 = np.row_stack((ra0, ra))

    return ra0


def getLaplacian_VideoArray(videoPath, videoW, videoH, yuvForm):
    '''
    Extract video Frame to array, but these frames are applied LaplacianFilter  
    '''
    yuv_frames = yuvio.mimread(videoPath, videoW, videoH, yuvForm)

    img_h = videoH
    img_w = videoW
    subimg_h, subimg_w = 128, 128

    # 初始化 initialization
    yuv_frame = yuv_frames.pop()
    y = yuv_frame.y
    grad = Laplacian_Filter(y)
    ##################### Divide frame and produce 2d matrix ############################
    array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
    ra0 = reshapeSplit(array)

    length = yuv_frames.__len__()
    for i in range(length):
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        grad = Laplacian_Filter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
        ra = reshapeSplit(array)
        ra0 = np.row_stack((ra0, ra))

    return ra0


def merge_MSSK_Array_together(normal_array,  sobel_array, lap_array):
    '''
    Combine normal array, sobel array, lap array together  
    '''
    # calculate each CU's mean, std, skew and kurt
    n_result = Array2DTo_MSSK(normal_array)
    s_result = Array2DTo_MSSK(sobel_array)
    l_result = Array2DTo_MSSK(lap_array)

    # change column name
    s_result.columns = ['Sobel_Mean', 'Sobel_std', 'Sobel_Skew', 'Sobel_Kurt']
    l_result.columns = ['Lap_Mean', 'Lap_std', 'Lap_Skew', 'Lap_Kurt']

    #  merge all these data frame together and produce a csv file
    mssk_result = pd.concat([n_result, s_result, l_result], axis=1)
    return mssk_result


def videoProcess(videoPath, videoW, videoH, yuvForm):
    '''
    Start the whole video process to get dataframe result
    '''
    # get 2D arrays of video with normal, after sobel, after lap, each row of arrays stand for a CU
    normal_array = getVideoArray(videoPath, videoW, videoH, yuvForm)
    sobel_array = getSobel_VideoArray(videoPath, videoW, videoH, yuvForm)
    lap_array = getLaplacian_VideoArray(videoPath, videoW, videoH, yuvForm)

    # merge together and return
    mssk_result = merge_MSSK_Array_together(
        normal_array, sobel_array, lap_array)
    return mssk_result


if __name__ == '__main__':
    videoPath = "video/BasketballPass_416x240_50.yuv"
    videoW = 416
    videoH = 240
    yuvForm = "yuv420p"
    Nomalarray = getVideoArray(videoPath, videoW, videoH, yuvForm)
    # print(Nomalarray.shape)
    sobelarray = getSobel_VideoArray(videoPath, videoW, videoH, yuvForm)
    # print(sobelarray)
    Nresult = Array2DTo_MSSK(Nomalarray)
    Sresult = Array2DTo_MSSK(sobelarray)
    Sresult.columns = ['SobelMean', 'Sobel_std', 'SobelSkew', 'SobelKurt']
    result = pd.concat([Nresult, Sresult], axis=1)
    print(result)
