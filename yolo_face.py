import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
import time
    
os.chdir("C:/Users/user/Desktop/tubitak_face_detection")

from yolo_model import YOLO
yolo=YOLO(0.6,0.5)



#train için labelleri alıyorum

file="data/coco_classes.txt"

with open(file) as f:
    class_name=f.readlines()

all_classes=[c.strip() for c in class_name]





# 3 tane futbolcum var bunların içine girip hepsini okuyup dosya isimlerini bir yere atıcam
def detect_person(folder_name,save=1):
    
    label_names=[]
    imgs=[]
    preIm=[]
    normalImg=[]
    
    def preImage(img):
        pimage=cv2.resize(img,(416,416))
        pimage=np.array(pimage,dtype=("float32"))
        pimage=pimage/255.0
        pimage=np.expand_dims(pimage,axis=0)
        preIm.append(pimage)
        
    
    
    os.chdir("C:/Users/user/Desktop/tubitak_face_detection/trainSet/"+folder_name)
    imgs=os.listdir() #burda ilgili kişinin fotograf dosya adları geldi
    
    
    for im in imgs:
        
        os.getcwd()
        p=cv2.imread(im)
        normalImg.append(p)
        preImage(p)
       
        
        #buraya kadar ilk forda dosya isimlerini okuyrarak label eşitlemesi yaptım 
        #ikinci forda imread işleminin yaparak preprocess işlemi uyguladım
    
    
    os.chdir("C:/Users/user/Desktop/tubitak_face_detection/faces")
    try:
        os.mkdir(folder_name)
    except:
        pass
    
    for i in range(len(preIm)):
        os.chdir("C:/Users/user/Desktop/tubitak_face_detection/faces/"+folder_name)
        boxes,classes,scores=yolo.predict(preIm[i],normalImg[i].shape)
        
        try:    
            for box,score,cl in zip(boxes,scores,classes):
                x,y,w,h=box
                
                top=max(0,np.floor(x+0.5).astype(int))
                
                left=max(0,np.floor(y+0.5).astype(int))
                
                right=max(0,np.floor(x+w+0.5).astype(int))
                
                bottom=max(0,np.floor(y+h+0.5).astype(int))
                
                #cv2.rectangle(normalImg[i], (top,left), (right, bottom),(255,0,0),2)
                #cv2.putText(normalImg[i], "{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
            
                
           
        except:
            continue
            
        if len(boxes)==1 and x>0 and y>0 and save==1:
            cv2.imwrite(str(i)+".jpg",normalImg[i][int(y):int(y+h),int(x):int(x+w)])
        
        elif save==0:
            return [int(x),int(y),int(w),int(h)]
        
    

#buraya kadar insan tespiti yaptım şimdi kafaları kesmek için har cascade kullanıcam ama kafaları bir dosyaya yazmam lazım

detect_person("luka_modric",1)



