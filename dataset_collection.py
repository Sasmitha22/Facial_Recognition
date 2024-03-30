import cv2, os
haar = "haarcascade_frontalface_default.xml"
path1 = 'datasets'
path2 = 'Lakshith'
path = os.path.join(path1,path2)
if not os.path.isdir(path1):
    os.mkdir(path1)
if not os.path.isdir(path):
    os.mkdir(path)

(w,h) = (130,100)
face_cascade = cv2.CascadeClassifier(haar)
count = 400
cam = cv2.VideoCapture(0)
while count < 439:
    print(f"{count} over")
    _, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        face_g = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face_g,(w,h))
        cv2.imwrite('%s/%s.png' %(path,count),face_resize)
    count += 1
    cv2.imshow('Opnecv',img)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()