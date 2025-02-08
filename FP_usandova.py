import cv2
import math
import time


# Selection variable: 'face' or 'nose'
selection = "face"  # Change to "face" if whole face detection is desired

# Load pre-trained face detector model
face_cascade = cv2.CascadeClassifier('/Users/uveimarsandoval/Documents/2024_Fall_Semester/CS485_Robotics/CS485_Final_Project/haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('/Users/uveimarsandoval/Documents/2024_Fall_Semester/CS485_Robotics/CS485_Final_Project/haarcascade_mcs_nose.xml')  # Path to the nose Haar cascade


# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

last_update_time = time.time()
angle_to_display = None
distance_to_display = None

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break
    
    # Flip the frame for a natural mirror effect
    frame = cv2.flip(frame, 1)

    # Get the frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Draw a dot at the center of the frame
    center_x, center_y = frame_width // 2, frame_height // 2
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

    # Convert the frame to grayscale (required for Haar cascades)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selection == "face":
        # Detect faces in the frame
        objects = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    elif selection == "nose":
        # Detect noses in the frame
        objects = nose_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    else:
        print("Invalid selection. Use 'face' or 'nose'.")
        break

    # Process detected objects
    for (x, y, w, h) in objects:
        # Calculate the center of the detected object
        object_center_x = x + w // 2
        object_center_y = y + h // 2

        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a dot at the center of the object
        cv2.circle(frame, (object_center_x, object_center_y), 5, (255, 0, 0), -1)  # Blue dot

        # Draw a line between the frame center and the object center
        cv2.line(frame, (center_x, center_y), (object_center_x, object_center_y), (0, 255, 0), 2)

        # Calculate the angle and distance between the frame center and the object center
        dx = object_center_x - center_x
        dy = object_center_y - center_y
        angle = math.degrees(math.atan2(dy, dx))
        distance = math.sqrt(dx**2 + dy**2)

        # Capture a snapshot if the dots meet
        if abs(dx) < 5 and abs(dy) < 5:  # Threshold for "meeting"
            print("Dots meet! Capturing snapshot...")
            snapshot = frame.copy()
            cv2.imshow("Snapshot", snapshot)

        # Update the angle and distance every 3 seconds
        if time.time() - last_update_time >= 3:
            angle_to_display = angle
            distance_to_display = distance
            last_update_time = time.time()

            # Print the ROS-like command to the terminal
            print(f"ROS Command: linear.x (distance)={distance_to_display:.2f}, angular.z (angle)={angle_to_display:.2f}")

        # Display the stored angle and distance on the frame
        if angle_to_display is not None and distance_to_display is not None:
            cv2.putText(frame, f"Adjust Angle: {angle_to_display:.2f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance_to_display:.2f} px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    
    # Display the frame with detected noses
    cv2.imshow("Original with Detections", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()