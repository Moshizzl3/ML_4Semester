import cv2
vid = cv2.VideoCapture(0)

myVegieDic = {'Bean': 0, 'Bitter_Gourd': 1, 'Bottle_Gourd': 2, 'Brinjal': 3, 'Broccoli': 4, 
              'Cabbage': 5, 'Capsicum': 6, 'Carrot': 7, 
              'Cauliflower': 8, 'Cucumber': 9, 'Papaya': 10, 'Potato': 11, 
              'Pumpkin': 12, 'Radish': 13, 'Tomato': 14}

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    

    cv2.putText(frame, 'It is a potato', (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,(255,255,255), 4, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()