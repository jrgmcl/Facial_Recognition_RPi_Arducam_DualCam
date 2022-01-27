import cv2
import os
import numpy as np
from PIL import Image


dataset_path = '/home/pi/Desktop/FACIALRECOGNITION/DATASETS'
users = os.listdir(dataset_path)

os.chdir("/home/pi/opencv/data/haarcascades")
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier('/home/pi/Desktop/FACIALRECOGNITION/haarcascade_frontalface_default.xml')

for newdir in users:
        #Get Every user dataset directories
        user_dataset_path = os.path.join(dataset_path, newdir).replace("\\","/")
        print (user_dataset_path)
        #images = os.listdir(user_dataset_path)
        list_image = os.listdir(user_dataset_path)
        print(list_image)
        print(len(list_image))
        
        count = 0
        for imagePath in list_image:
            imageFPath = os.path.join(user_dataset_path, imagePath).replace("\\","/")
            #print("Converted to B&W: "+imageFPath)
            img = cv2.imread(imageFPath, 0)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(img, 1.3, 5)
            
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                split_filename = newdir.split('.')
                idno = split_filename[0].lstrip('0')
                # Save the captured image into the datasets folder
                print(user_dataset_path)
                cv2.imwrite(user_dataset_path + '/' + newdir + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])
                if os.path.exists(imageFPath):
                    os.remove(imageFPath)
                print(list_image)
                #cv2.imshow('image', img)
                
print ("\n [INFO] Conversion completed. Generating 'Trainer.yml'...")

def getImagesAndLabels(dataset_path):
    
    #list_image = os.listdir(user_dataset_path)
    for newdir in users:
        user_dataset_path = os.path.join(dataset_path, newdir).replace("\\","/")
        list_image = os.listdir(user_dataset_path)
        for imagePath in list_image:
            imageFPath = os.path.join(user_dataset_path, imagePath).replace("\\","/")
            print(imageFPath)
            faceSamples=[]
            ids = []
            img = cv2.imread(imageFPath, 0)
            img_numpy = np.array(img, 'uint8')
            split_filename = newdir.split('.')
            id = int(split_filename[0].lstrip('0'))
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
               
            
        return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(dataset_path)
recognizer.train(faces, np.array(ids))
recognizer.save('/home/pi/Desktop/FACIALRECOGNITION/TRAINER/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


