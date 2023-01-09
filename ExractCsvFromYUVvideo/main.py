import pandas as pd
import videoProcessFunctions as vpf

if __name__ == '__main__':
    videoPath = "video/BasketballPass_416x240_50.yuv"
    videoW = 416
    videoH = 240
    yuvForm = "yuv420p"
    csvPath = "csvFile/basketballCU.csv"

    mssk_result = pd.DataFrame
    mssk_result = vpf.videoProcess(videoPath, videoW, videoH, yuvForm)

    # mssk_result.to_csv(csvPath)
    print(mssk_result)
