import cv2
import numpy as np
import os

os.chdir("/home/pi/opencv/data/haarcascades")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/TRAINER/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

#Generate the names on the dataset
dataset_path = '/home/pi/Desktop/FACIALRECOGNITION/DATASETS'
users = os.listdir(dataset_path)

# names related to ids: example ==> KUNAL: id=1,  etc
#Put the user's name in array
first_name = []
last_name = []
for newdir in users:
    split_filename = newdir.split('.')
    first_name.append(split_filename[1])
    last_name.append(split_filename[2])

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 1280) #Camera Size
cam.set(4, 720)

width = 2560
height = 720
new_size = (width, height) #Camera size for 2 cameras

# Define min window size to be recognized as a face
minW = 0.01*width
minH = 0.01*height

while True:
    ret, raw =cam.read()
    stretched = cv2.resize(raw, new_size, interpolation = cv2.INTER_AREA) #Set the new size
    crop = stretched[0:720, :1280] #Crop the camera 1
    img = crop[180:540, 320:960]
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            for newdir in users:
                user_dataset_path = os.path.join(dataset_path, newdir).replace("\\","/")
                #list_image = os.listdir(user_dataset_path)
                #print(user_dataset_path)
                split_filename = newdir.split('.')
                id_dataset = int(split_filename[0].lstrip('0'))
                if (id_dataset==id):
                    print(split_filename[1]+' '+split_filename[2])
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(first_name[id]), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
