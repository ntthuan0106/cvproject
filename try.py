import cv2

# Set your camera's IP address and port number
camera_url = "http://192.168.137.185:8080/video"

# Create a VideoCapture object to connect to the camera
cap = cv2.VideoCapture(camera_url)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('IP Camera Feed', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
