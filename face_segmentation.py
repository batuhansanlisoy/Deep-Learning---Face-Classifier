import cv2
import os
import time

os.chdir("C:/Users/user/Desktop/tubitak_face_detection")
face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profil_detector=cv2.CascadeClassifier("haarcascade_profileface.xml")


def face_detect(folder_name,save=1):
    
    os.chdir("C:/Users/user/Desktop/tubitak_face_detection/real_face")
    try:
        os.mkdir(folder_name)
    except:
        pass
        
             
    
    
    os.chdir("C:/Users/user/Desktop/tubitak_face_detection/faces/"+folder_name)

    img_names=os.listdir()
    imgs=[]
    
    for i,img in enumerate(img_names):
        os.chdir("C:/Users/user/Desktop/tubitak_face_detection/faces/"+folder_name)
        p=cv2.imread(img,0)
        box=face_detector.detectMultiScale(p)
        if len(box)<1:
            box=profil_detector.detectMultiScale(p)
        
        
            
        elif len(box)==1 :
            for x,y,w,h in box:
                #cv2.rectangle(p,(x,y),(x+w,y+h),1,2)
                
                os.chdir("C:/Users/user/Desktop/tubitak_face_detection/real_face/"+folder_name)
                if save==1:
                    
                    cv2.imwrite(str(i)+".jpg",p[y:y+h,x:x+h])
                elif save==0:
                    return [x,y,w,h]
        
       
        
    

face_detect("luka_modric",1)