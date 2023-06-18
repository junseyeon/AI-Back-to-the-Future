import numpy as np
import cv2
import time
from os.path import splitext, basename, join


class Colorizer:
    def __init__(self, height=480, width=600):
        (self.height, self.width) = height, width

        self.colorModel = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt",
                                                  "model/colorization_release_v2.caffemodel")

        clusterCenters = np.load("model/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2,313,1,1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1,313], 2.606, np.float32)]


    # def processImage(self, imgName):
    #     self.img = cv2.imread(imgName)
    #     self.img = cv2.resize(self.img, (self.width,self.height))

    #     self.processFrame()
    #     cv2.imwrite(join("output", basename(imgName)), self.imgFinal)
    #     cv2.imshow("Output", self.imgFinal)
    #     key = cv2.waitKey()


    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)

        if(cap.isOpened() == False):
            print('error opening video')
            return
        (sucess, self.img) = cap.read()

        prevFrameTime = 0
        nextFrameTime = 0

        out = cv2.VideoWriter('video.mp4',
                              cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (self.width * 2, self.height))


        while sucess:
            self.img = cv2.resize(self.img, (self.width,self.height))

            self.processFrame()
            out.write(self.imgFinal)

            nextFrameTime = time.time()
            fps = 1/(nextFrameTime - prevFrameTime)
            prevFrameTime = nextFrameTime

            fps = "FPSS:" + str(int(fps))
            cv2.putText(self.imgFinal, fps, (5,25), cv2.FONT_HERSHEY_SIMPLEX,1,
                        (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("color_video", self.imgFinal)

            key = cv2.waitKey(1)& 0xFF
            if key == ord('q'):
                break
            (sucess, self.img) = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()



    def processFrame(self):
        imgNormalized = (self.img[:,:,[2,1,0]]* 1.0/255).astype(np.float32)

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        channelL = imgLab[:,:,0]

        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (244,244)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:,:,0]
        channelLResized -= 50

        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))

        resultResized = cv2.resize(result, (self.width, self.height))

        self.imgOut = np.concatenate((channelL[:,:,np.newaxis], resultResized), axis=2)
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut,cv2.COLOR_Lab2BGR), 0, 1)
        self.imgOut = np.array((self.imgOut)*255,dtype = np.uint8)

        self.imgFinal = np.hstack((self.img,self.imgOut))
        