import cv2
import numpy as np
import face_recognition

imgElon=face_recognition.load_image_file('images/Elon.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgelontest=face_recognition.load_image_file('images/elontest.jpg')
imgelontest=cv2.cvtColor(imgelontest,cv2.COLOR_BGR2RGB)

facloc=face_recognition.face_locations(imgElon)[0]
encode=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)

factest=face_recognition.face_locations(imgelontest)[0]
encodetest=face_recognition.face_encodings(imgelontest)[0]
cv2.rectangle(imgelontest,(factest[3],factest[0]),(factest[1],factest[2]),(255,0,255),2)

result=face_recognition.compare_faces([encode],encodetest)
ds=face_recognition.face_distance([encode],encodetest)

print(result,ds)
cv2.putText(imgelontest,(f'{result} {round(ds[0],2)}'),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
cv2.imshow('Elon Musk Test ',imgelontest)
cv2.imshow('Elon Musk',imgElon)
cv2.waitKey(0)



