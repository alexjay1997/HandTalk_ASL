import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("myhaar_A.xml")
BCascade = cv2.CascadeClassifier("myhaarB.xml")
cap = cv2.VideoCapture(0)
cap.set(3,300)
cap.set(4,300)
while True:
    ret, frame = cap.read()
    ret2, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )
    handsB = BCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 60, 200), 2,)
        cv2.putText(frame, "A", (200, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    for (x, y, w, h) in handsB:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 60, 150), 2,)
        cv2.putText(frame, "B", (200, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))


        
    # Display the resulting frame
   
    cv2.imshow('Gray',gray)
    cv2.imshow('Original',frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# Convert to grayscale

 # Draw a rectangle around recognized faces

 # Detect features specified in Haar Cascade
