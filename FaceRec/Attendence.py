import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imgf'
images = []
classname = []
mylist = os.listdir(path)
print(mylist)

for a in mylist:
    curimg=cv2.imread(f'{path}/{a}')
    images.append(curimg)
    classname.append(os.path.splitext(a)[0])
print(classname)

def findencode(images):
    encodelist=[]
    for a in images:
        a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(a)[0]
        encodelist.append(encode)
    return encodelist

def MarkAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        namelist=[]
        for line in myDataList:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



       



encodenumbers=findencode(images)
print('Encoding completes')


cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    findloc=face_recognition.face_locations(imgs)
    encodecur=face_recognition.face_encodings(imgs,findloc)
    
    for a,b in zip(encodecur,findloc):
       matches=face_recognition.compare_faces(encodenumbers,a)
       facedis=face_recognition.face_distance(encodenumbers,a)
       print(facedis)
       matchindex=np.argmin(facedis)
       if matches[matchindex]:
           name=classname[matchindex].upper()
           print(name)

           y1,x2,y2,x1=b
           y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
           cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
           MarkAttendance(name)




    cv2.imshow('webcam',img)
    cv2.waitKey(1)

 