from __future__ import print_function
import cv2 as cv
import argparse
import time,json

ip_address = '192.168.1.98'  
port = 4747
url = f'http://{ip_address}:{port}/video'

try:
    with open("file.json") as f:
        datas=json.load(f)
except:
    datas={}
nome = ''
id=0
def detectAndDisplay(frame,count):
    #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces

    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
       # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        #frame = cv.rectangle(frame,(x+w//2-10,y+h//2-10),(x+w//2+10,y+h//2+10),frame_gray)
        #frame = cv.rectangle(frame,(center[0]-(center[0]//3),center[1]-(center[0]//3)),(center[0]+(center[0]//3),center[1]+(center[0]//3)),(255,0,255))
        #frame = cv.rectangle(frame,(center[0]-100,center[1]-100),(center[0]+100,center[1]+100),(255,0,255))
        
        #frame = cv.ellipse()
        faceROI = frame[y:y+h,x:x+w]
        
    cv.imshow('Capture - Face detection', frame)
    if count<31:
        try:
            cv.imwrite(f"Faces/{nome}.{id}.{count}.jpg",faceROI)
            count+=1
        except UnboundLocalError:
            pass
        return count
while True:
    nome=input("inserisci il nome della faccia da riconoscere\n")
    id = int(input( "inserisci l'id\n"))
    if nome not in datas and id not in datas:
        datas[nome] = id
        break



parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='MachineLearning\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')

parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

face_cascade = cv.CascadeClassifier()


#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera

#-- 2. Read the video stream
#cap = cv.VideoCapture(url)
cap = cv.VideoCapture(1)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

count = 0
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    count =detectAndDisplay(frame,count)
    if cv.waitKey(1) == ord('q'):
        break
    
    if count>30:
        with open('file.json',"w") as f:
            print(datas)
            json.dump(datas,f)

        break
    