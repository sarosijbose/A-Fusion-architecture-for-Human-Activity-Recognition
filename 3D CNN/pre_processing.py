import os
import cv2
import numpy as np
import logging

logging.basicConfig(level = logging.INFO)

DATA_DIR = '/data/sample videos'
SAVE_DIR = 'output'

_EXT = ['.avi', '.mp4']
_IMAGE_SIZE = 224

def frame_count(video_path):
  
  _, ext = os.path.splitext(video_path)
  if not ext in _EXT:
    raise ValueError('Extension "%s" not supported' % ext)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened(): 
    raise ValueError("Could not open the file.\n{}".format(video_path))
  if cv2.__version__ >= '3.0.0':
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
  else:
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
  length = int(cap.get(CAP_PROP_FRAME_COUNT))
  cap.release()
  return length

def rgb_preprocessing(video_path):
    
    rgb = []
    frame_limit = 79
    count_frame = 0
    videocapture = cv2.VideoCapture(video_path)
    success,frame = videocapture.read()
    while success:
        if count_frame <= frame_limit:
            frame = cv2.resize(frame, (342,256)) 
            frame = (frame/255.)*2 - 1
            frame = frame[16:240, 59:283]    
            rgb.append(frame)        
            success,frame = videocapture.read()
            count_frame+=1
        else:
            break
    # To be used when frame-length is not enough.
    '''
    total_frames = frame_count(video_path)
    if count_frame < total_frames:
        for _ in range(total_frames - count_frame):
            rgb.append(frame)'''

    videocapture.release()
    rgb = rgb[:-1]
    rgb = np.asarray([np.array(rgb)])
    print('Shape of RGB Data: ',rgb.shape)
    file_name = '/' + (video_path.split('/')[-1:][0]).split('.')[0] + '.npy'
    np.save(SAVE_DIR + file_name, rgb)
    return rgb
        
def main():
  
  rgb_preprocessing(DATA_DIR + '/insert file name.avi')
  logging.info('Video pre-processing finished.')
  
if __name__ == '__main__':
  main()