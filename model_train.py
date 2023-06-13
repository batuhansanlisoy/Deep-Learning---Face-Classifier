import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
#import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import label_binarize
siniflar=[]
images=[]
os.chdir("C:/Users/user/Desktop/tubitak_face_detection/real_face/")

#ronaldo ve messi klasörünü açıp içerisindeki fotografları images altında toplayadcam


f_list=os.listdir()
for i,fol in enumerate(f_list):
    os.rename(fol,str(i))
    
    

def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=cv2.resize(img,(32,32))
    img=img/255
    return img




isimler=os.listdir()
class_count=len(isimler)

for i in isimler:
    x=os.listdir(i)
    for j in x:
        img=cv2.imread(str(i)+"/"+j)
        if img.shape[0]>=32 and img.shape[1]>=32:
            images.append(img)
            siniflar.append(i)
           
siniflar=np.array(siniflar)
images=np.array(images)


x_train,x_test,y_train,y_test=train_test_split(images,siniflar, test_size = 0.5, random_state = 42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train, test_size = 0.2, random_state = 42)



x_train=np.array(list(map(preProcess,x_train)))
x_test=np.array(list(map(preProcess,x_test)))
x_validation=np.array(list(map(preProcess,x_validation)))


x_train=x_train.reshape(-1,32,32,1) #-1 kaç tane resim varsa demek
x_test=x_test.reshape(-1,32,32,1)
x_validation=x_validation.reshape(-1,32,32,1)  #bunu yapmaktadi amaç verim (150,32,32) şekline geliyorsa bunu (150,32,32,1) şeklinde yazmak için

x_train.shape



# x_train=np.expand_dims(x_train, axis=3)  yukarıdaki işlemin farklı bir yolu


dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1,
                           rotation_range=0.1)

dataGen.fit(x_train)


y_train=to_categorical(y_train,class_count) # 10 burda kaç tane sinifim oldugunu belirtiyor
y_test=to_categorical(y_test,class_count)
y_validation=to_categorical(y_validation,class_count)


model=Sequential()

model.add(Conv2D(input_shape=(32,32,1),filters=8,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3 ),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3 ),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=512, activation="relu"))


model.add(Dense(units=class_count, activation="softmax"))
#burdaki 10 tane sınıfm oldugu için 10 yazdım

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy","AUC","MeanSquaredError"])

batch_size=50 


hist=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batch_size),
                         validation_data=(x_validation,y_validation),
                         epochs=175,steps_per_epoch=x_train.shape[0]//batch_size)


"""pickle_out=open("yeni.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
"""

os.chdir("C:/Users/user/Desktop/tubitak_face_detection")
#kaydetme
model.save("yeni.h5")



k=hist.history

dir(hist)

#%%

hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"],label="eğitim kaybı")
plt.plot(hist.history["val_loss"],label="val loss")
plt.legend()
plt.show()



plt.figure()
plt.plot(hist.history["accuracy"],label="accuracy")
plt.plot(hist.history["val_accuracy"],label="val loss")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history["mean_squared_error"],label="mse")
plt.plot(hist.history["val_mean_squared_error"],label="val mse")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history["auc"],label="auc")
plt.plot(hist.history["val_auc"],label="val auc")
plt.legend()
plt.show()


score=model.evaluate(x_test,y_test,verbose=1)
print("test loss:", score[0])
print("test accuracy:",score[1])
print("test auc:",score[2])
print("test mse:",score[3])


#%%
#model test degerleri
y_pred=model.predict(x_validation)
y_pred_class=np.argmax(y_pred,axis=1)
Y_true=np.argmax(y_validation,axis=1)


cm=confusion_matrix(Y_true,y_pred_class)


f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("predicted")
plt.ylabel("True")
plt.title("confusion matrix")
plt.show()




#%%














