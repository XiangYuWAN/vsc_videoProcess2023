import pandas as pd
from videoProcessCsv import VideoToCsv


if __name__ == '__main__':
    videoPath = r"ExractCsvFromYUVvideo\video\BasketballPass_416x240_50.yuv"
    videoW = 416
    videoH = 240
    yuvForm = "yuv420p"
    csvPath = r"ExractCsvFromYUVvideo\csvFile\basketballCU.csv"

    vtc = VideoToCsv(videoPath, videoW, videoH, yuvForm, csvPath)
    vtc.videoProcess()
    vtc.writeToCsvFile()
