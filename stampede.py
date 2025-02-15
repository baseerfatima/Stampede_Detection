import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces within a specific area and count them
def detect_faces_in_area(frame, area):
    # Convert area coordinates to integers
    x, y, w, h = area
    # Crop the frame to the specified area
    area_frame = frame[y:y+h, x:x+w]
    # Convert the cropped frame to grayscale
    gray = cv2.cvtColor(area_frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the area
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Draw bounding boxes around the detected faces
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(area_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Blue color for face bounding box
    # Return the number of faces detected
    return len(faces)

# Define the area where you want to detect faces (x, y, width, height)
area_of_interest = (300, 100, 600, 600)  # Adjusted bounding box size and position

def process_frame(frame, window_width, window_height):
    # Resize the frame to match the aspect ratio of the display window
    frame = cv2.resize(frame, (window_width, window_height))
    
    # Draw bounding box for the area of interest
    x, y, w, h = area_of_interest
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for main bounding box

    # Detect faces in the specified area
    num_faces = detect_faces_in_area(frame, area_of_interest)
    # Generate an alert if any face is detected
    if num_faces > 0:
        cv2.putText(frame, "ALERT: Chances of Stampede!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

# Specify the path to the video file
video_path = "rush1.mp4"  # Update with the actual path to your video file

# Flag to indicate whether to use webcam or video file
use_webcam = True

if use_webcam:
    # Open webcam
    cap = cv2.VideoCapture(0)
else:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

# Get the dimensions of the display window
window_width = 800  # Update with the width of your display window
window_height = 600  # Update with the height of your display window

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        # Process the frame
        frame = process_frame(frame, window_width, window_height)
        
        # Display the frame
        cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed or end of video is reached
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

# Release the capture
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
