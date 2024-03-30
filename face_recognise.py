import cv2
import os
import numpy as np

print(cv2.__version__)

haar_file = "haarcascade_frontalface_default.xml"
datasets_dir = "datasets"

print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

for subdir, dirs, files in os.walk(datasets_dir):
    for sub_dir in dirs:
        names[id] = sub_dir
        path = os.path.join(datasets_dir, sub_dir)
        for file in os.listdir(path):
            if file.endswith(".jpg") or file.endswith(".png"):  # Filter image files
                img_path = os.path.join(path, file)
                label = id
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (130, 100))  # Resize image to specified width and height
                images.append(image)
                labels.append(int(label))
        id += 1

images = np.array(images)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer.create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)

cam = cv2.VideoCapture(0)
cnt = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (130, 100))  # Resize face image
        prediction = model.predict(face_resized)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        if prediction[1] < 100:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", img)
                cnt = 0
                
    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
