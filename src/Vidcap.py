
from __future__ import division, print_function, absolute_import

import os
from timeit import time

import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import json 
import face_recognition
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import json
warnings.filterwarnings('ignore')


frame_file=[]

def generate_json(bbox,frame_id,track_id,frame_file):
    image_dict = {"frame_id":'', "Persons_faces_coordinates":[]}
    label_dict = {"Person_id":'', "coordinates":{}}
    coord_dict = {"x_mini":int, "y_mini":int, "x_maxi":int, "y_maxi":int}
    coord_dict['x_mini'] = int(bbox[0])
    coord_dict['y_mini'] = int(bbox[1])
    coord_dict['x_maxi'] = int(bbox[2])
    coord_dict['y_maxi'] = int(bbox[3])
    label_dict['Person_id'] = track_id
    label_dict['coordinates'] = coord_dict
    image_dict['frame_id'] = frame_id
    image_dict['Persons_faces_coordinates'].append(label_dict)
    frame_file.append(image_dict)
   
def facebox(bbox):
    face_list = []
    i1 = bbox[0] + 0.35*(bbox[2]-bbox[0])
    i2 = bbox[1] + 0.035*(bbox[3]-bbox[1])
    i3 = bbox[2] - 0.35*(bbox[2]-bbox[0])
    i4 = bbox[1] + 0.23*(bbox[3]-bbox[1])
    face_list.append(i1)
    face_list.append(i2)
    face_list.append(i3)
    face_list.append(i4)
    return face_list

def main(yolo):

  
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture('test.mp4')

    if writeVideo_flag:
    # writing objects
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    c=0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        c+=1
       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
        count = len(boxs)
        print("count: ",len(boxs))
        features = encoder(frame,boxs)
        
        
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        #non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
       
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            face_list =  facebox(bbox)
            cv2.rectangle(frame, (int(face_list[0]), int(face_list[1])), (int(face_list[2]), int(face_list[3])),(255,255,255), 2)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            facebox(bbox)
            generate_json(face_list,c,track.track_id,frame_file)
            #facedet(i1,i2,i3,i4)
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('clip', frame)
        
        if writeVideo_flag:
            # saving frame
            out.write(frame)
            frame_index += 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
          
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))

        
  
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    json_file = json.dumps(my_file)
    with open('output.json', 'w') as f:    
         f.write(json_file)    
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main(YOLO())
    
    
