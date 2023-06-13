import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
import time
    
os.chdir("C:/Users/user/Desktop/tubitak_face_detection")

from yolo_model import YOLO




#train için labelleri alıyorum

file="data/coco_classes.txt"

with open(file) as f:
    class_name=f.readlines()

all_classes=[c.strip() for c in class_name]



yolo=YOLO(0.6,0.5)
# 3 tane futbolcum var bunların içine girip hepsini okuyup dosya isimlerini bir yere atıcam



#dosya okuma işlemi

os.getcwd()
os.chdir("C:/Users/user/Desktop/tubitak_face_detection/trainSet")
Person_class=os.listdir()

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
    


for i in Person_class: 
    img_names=os.listdir(i)
    
    
    for j in img_names:
        
        imgs.append(j)
        label_names.append(i)

        
        
        
        
for (im,label) in zip(imgs,label_names):
    
    
    p=cv2.imread(label+"/"+im)
    normalImg.append(p)
    preImage(p)
   
    
    #buraya kadar ilk forda dosya isimlerini okuyrarak label eşitlemesi yaptım 
    #ikinci forda imread işleminin yaparak preprocess işlemi uyguladım




for i in range(len(preIm)):
    os.chdir("C:/Users/user/Desktop/tubitak_face_detection/faces/cristiano_ronaldo")
    boxes,classes,scores=yolo.predict(preIm[i],normalImg[i].shape)
    
    try:    
        for box,score,cl in zip(boxes,scores,classes):
            x,y,w,h=box
            
            top=max(0,np.floor(x+0.5).astype(int))
            
            left=max(0,np.floor(y+0.5).astype(int))
            
            right=max(0,np.floor(x+w+0.5).astype(int))
            
            bottom=max(0,np.floor(y+h+0.5).astype(int))
            
            cv2.rectangle(normalImg[i], (top,left), (right, bottom),(255,0,0),2)
            cv2.putText(normalImg[i], "{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
        
            print("****",x,y,w,h)
       
    except:
        continue
        
    if len(boxes)==1 and x>0 and y>0:
        cv2.imwrite(str(i)+".jpg",normalImg[i][int(y):int(y+h),int(x):int(x+w)])

    cv2.imshow("s",normalImg[i])
    

#buraya kadar insan tespiti yaptım şimdi kafaları kesmek için har cascade kullanıcam ama kafaları bir dosyaya yazmam lazım





