import cv2
import numpy as np
import os 
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir("C:/Users/user/Desktop/tubitak_face_detection")
face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profil_detector=cv2.CascadeClassifier("haarcascade_profileface.xml")



os.chdir("C:/Users/user/Desktop/tubitak_face_detection/test")
images=[]
modelImg=[]
model=load_model("yeni.h5")
model.summary()
#model.summary()
os.getcwd()


img_names=os.listdir()
img_names
for i in img_names:
    if i.endswith(".jpg") or i.endswith(".JPG"):
        img=cv2.imread(i)
        images.append(img)
        
        

def preProcess(img):
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img
    


for i in images:

    
    img=i
    gercekResim=img.copy()
    
    box=face_detector.detectMultiScale(img)
    if len(box)<1:
        box=profil_detector.detectMultiScale(img)
    
    
        
    elif len(box)>=1 :
        for x,y,w,h in box:
            
            img=gercekResim[y:y+h,x:x+w]
            img=np.asarray(img)
            img=cv2.resize(img,(32,32))
            img=preProcess(img)
            img=img.reshape(1,32,32,1)
            predictions = model.predict(img)
            
            
            sinif=np.argmax(predictions)
            
            print("yüzde",np.max(predictions))
            
            if np.max(predictions)>0.85:
                cv2.rectangle(gercekResim,(x,y),(x+w,y+h),1,2)
                if sinif==0:
                    cv2.putText(gercekResim,"drogba-- {}".format(np.max(predictions)),(x,y-10),1,1,(0,255,50),1)
                    print("drogba")
                elif sinif==1:
                    cv2.putText(gercekResim,"messi-- {}".format(np.max(predictions)),(x,y-10),1,1,(0,255,50),1)
                    print("messi")
                elif sinif==2:
                    cv2.putText(gercekResim,"ronaldo--{}".format(np.max(predictions)),(x,y-10),1,1,(0,255,50),1)
                    print("ronaldo")
                elif sinif==3:
                    cv2.putText(gercekResim,"modric--{}".format(np.max(predictions)),(x,y-10),1,1,(0,255,50),1)
                    print("luka modric")
            else:
                print("birden çok yüz tanımlandı ancak kimlik tespit edilemedi")
                
            
    cv2.imshow("sadfads",gercekResim)
    if cv2.waitKey(0)==ord("q"):
        break
    
    
    
   

  





