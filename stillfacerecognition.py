import os
import cv2
import numpy as np

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('C:/Users/ADMIN/Downloads/face recognition/haarcascade/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)
    return faces,gray_img

def train_classifier(faces,face_ID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer
def draw_rect(test_img,face):
    (x,y,w,h)=face
cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0),6)
   
      def labels_for_training_data(directory):
    faces=[]
    faceID=[]
   
    for path,subdirnames,filenames in os.walk(directory):
       
        for filename in filenames:
def labels_for_training_data(directory):
    faces=[]
    faceID=[]
   
    for path,subdirnames,filenames in os.walk(directory):
       
        for filename in filenames:
           if filename.startswith("."):

                    print("Skipping system files")
                    continue
                id=os.path.basename(path)
                img_path=os.path.join(path,filename)
                print("img_path",img_path)
                print("id:",id)
                test_img=cv2.imread(img_path)
                if test_img is None:
                      print("Image not loaded properly")
                     
                      continue
                faces_rect,gray_img=faceDetection(test_img)
                if len(faces_rect)!=1:
 continue
                faces_rect,gray_img=faceDetection(test_img)
                if len(faces_rect)!=1:
                      continue

                (x,y,w,h)=faces_rect[0]
                rol_gray=gray_img[y:y+w,x:x+h]
                faces.append(rol_gray)
                faceID.append(int(id))
    return faces,faceID

                 


test_img=cv2.imread('C:/Users/ADMIN/Downloads/face recognition/test_image.jpg')
faces_detected,gray_img=faceDetection(test_img)
print("face detected",faces_detected)
'''for (x,y,w,h) in faces_detected:
 cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
    pass
resized_img=cv2.cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ",resized_img)
cv2.waitKey(0)
cv2.destoyAllWindows()'''

#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
faces,faceID=labels_for_training_data('C:/Users/ADMIN/Downloads/face recognition/training_images')
face_recognizer=train_classifier(faces,faceID)
#face_recognizer.read('C:/Users/ADMIN/Downloads/face recognition/trainingdata.yml')
face_recognizer.save('C:/Users/ADMIN/Downloads/face recognition/training_images.yml')
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/ADMIN/Downloads/face recognition/trainingdata.yml')
name={0:"Bill Gates"}
for face in faces_detected:
        (x,y,w,h)=face
        rol_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(rol_gray)
        print("confidence",confidence)
        print("label",label)
        draw_rect(test_img,face)
predict_name=name[label]
        if confidence>68:
            continue
        put_text(test_img,predict_name,x,y)      
for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)
    pass
resized_img=cv2.cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

               
           

           
               
                      
                 
                 
