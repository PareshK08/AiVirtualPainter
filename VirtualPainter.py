import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

###################
breshthickness = 12
eraserthickness = 100
###################


FOLDER_PATH = "Header"
myList = os.listdir(FOLDER_PATH)
overlayList = []
count = 0

for imPath in myList:
    image = cv2.imread(f'{FOLDER_PATH}/{imPath}')
    overlayList.append(image)
    
header = overlayList[0]   
#print(len(overlayList))

drawColor = (0,255,255)  

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imageCanvas = np.zeros((720,1280,3), np.uint8)

while True: 
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    
    img = detector.findHands(img)
    
    # Find the hand landmarks
    lmlist ,bbox= detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:] # Index finger tip
        x2,y2 = lmlist[12][1:] # Middle finger tip
    
        # Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        
        # If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                
                if x1 < 100:
                    drawColor = (0,255,255)
                    header = overlayList[0]
                
                if 150 < x1 <300:
                    drawColor = (0,0,255)
                    header = overlayList[1]

                if 400<x1<530:
                    drawColor = (0,255,0)
                    header = overlayList[2]
                    
                if 660<x1<800:
                    drawColor = (200,50,50)
                    header = overlayList[3]
                    
                if 930<x1<1010:
                    drawColor = (0,0,0)
                    header = overlayList[4]    
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)   
            #print("Selection Mode")
            
        
        # If Drawing mode - Index finger is up
        
        if fingers[1] and fingers[2] == False:
                     
            
            cv2.circle(img, (x1,y1), 10, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserthickness)
                cv2.line(imageCanvas,(xp,yp),(x1,y1),drawColor,eraserthickness)
            
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,breshthickness)
                cv2.line(imageCanvas,(xp,yp),(x1,y1),drawColor,breshthickness)
            
            xp,yp = x1,y1  
            
        if fingers[0]==0 and fingers[1]==0 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
            count+=1
            if count>=30:

                break
    imgGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imageCanvas)
    # Setting the header
    img[0:120, 0:1280] = header
    
    #img = cv2.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imageCanvas)
    cv2.waitKey(1) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    