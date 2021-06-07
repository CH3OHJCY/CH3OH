# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:20:40 2021

@author: ch3oh
"""
import cv2


#function part
def video_edit(path,start,end,name):
    """
    path：视频路径
    start：开始时间 s/秒
    end：结束时间 s/秒
    """
    #读取视频
    video_capture = cv2.VideoCapture(path)
    #相关信息
    video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(video_capture.get(5))
    #输出相关信息
    print("视频像素:{}".format(video_size))
    print("视频总帧数:{}".format(total_frames))
    print("视频帧率:{}".format(video_fps))
    print("视频时长:{} s".format(total_frames/video_fps))
    #选择截取时间
    start_fps=start*video_fps
    end_fps=end*video_fps
    #判断是否符合条件
    if(end_fps>total_frames):
        end_fps=total_frames
    #输出.mp4格式
    videoWriter =cv2.VideoWriter(name,cv2.VideoWriter_fourcc('M','P','4','V'),video_fps,video_size)
    i = 0
    while True:
        success,frame = video_capture.read()
        if success:
            i += 1
            if(i>=start_fps and i <= end_fps):
                #图像录入视频
                videoWriter.write(frame)
        else:
            #剪辑完成
            print('好耶，剪辑完成！')   
            break
        
#main_part
path = "C:/Users/ch3oh/Desktop/细/192.168.10.3_02_20210510093122433.mp4"
name = "192.168.10.3_02_20210510093122433_1.mp4"
start = 19
end= 62
video_edit(path,start,end,name)


