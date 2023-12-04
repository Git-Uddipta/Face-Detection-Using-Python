import cv2
#classifying (loading the cascade file)
a=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

b=cv2.VideoCapture(0)

#infinite loop:
while True:
    c_rec,d_image=b.read()
    e=cv2.cvtColor(d_image,cv2.COLOR_BGR2GRAY)
    f=a.detectMultiScale(e,1.3,6)

    for(x1,y1,w1,h1) in f:
        cv2.rectangle(d_image,(x1,y1),(x1+w1,y1+h1),(0,255,0),10)

        cv2.imshow('img',d_image)
        h=cv2.waitKey(40) & 0xff
        if h==40:
            break

b.release()
cv2.destroyAllWindows()

#In this Project, we will see how to develop real-time human face detection in Python using OpenCV. The objective of the program given is to detect objects of interest(faces) in real-time and to keep track of the same object. Face Recognition is a part of computer vision where we locate and visualize faces using ML models in any digital image.
#OpenCV contains many pre-trained classifiers for face, eyes, smiles, etc. Face detection using Haar cascades files is a machine-learning approach where the model is trained with a set of input data.