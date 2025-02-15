import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
        cv2.rectangle(area_frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)  # Draw in red color
    # Return the number of faces detected
    return len(faces)

# Define the area where you want to detect faces (x, y, width, height)
area_of_interest = (80, 80, 260, 260)  # Adjusted bounding box size

# Email configuration
email_sender = "baseerfatima01@gmail.com"  # Sender email address
email_receiver = "2020a1r045@mietjammu.in"  # Receiver email address
email_password = "upqgvfxwbxwghggr"  # App password generated for Gmail
email_subject = "Alert: Face Detected!"  # Email subject

# Function to send email
def send_email():
    message = MIMEMultipart()
    message['From'] = email_sender
    message['To'] = email_receiver
    message['Subject'] = email_subject
    body = "Alert: A face has been detected in the specified area."
    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_sender, email_password)
        text = message.as_string()
        server.sendmail(email_sender, email_receiver, text)
        server.quit()
        print("Email notification sent successfully!")
    except Exception as e:
        print("Failed to send email notification:", e)

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        # Draw bounding box for the area of interest
        x, y, w, h = area_of_interest
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color
        
        # Detect faces in the specified area
        num_faces = detect_faces_in_area(frame, area_of_interest)
        # Generate an alert if any face is detected
        if num_faces > 0:
            cv2.putText(frame, "ALERT: Face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red color
            # Send email notification
            send_email()
        
        # Display the frame
        cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()