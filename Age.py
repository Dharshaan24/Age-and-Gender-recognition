import cv2
import numpy as np

# Load face, age, and gender detection models
face_proto = "opencv_face_detector.pbtxt"  # Face model configuration
face_model = "opencv_face_detector_uint8.pb"  # Face model weights
age_proto = "age_deploy.prototxt"  # Age model configuration
age_model = "age_net.caffemodel"  # Age model weights
gender_proto = "gender_deploy.prototxt"  # Gender model configuration
gender_model = "gender_net.caffemodel"  # Gender model weights

# Mean values for model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Labels for age and gender
age_buckets = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load the models
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Start video capture
cap = cv2.VideoCapture(0)

# Function to detect faces
def highlight_face(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)))
    return frame_opencv_dnn, face_boxes

# Main loop to read frames from the webcam and perform detection
while True:
    ret, frame = cap.read()  # Corrected from 'trame' to 'frame'
    if not ret:
        break

    result_img, face_boxes = highlight_face(face_net, frame)
    if not face_boxes:
        print("No face detected")
    
    for face_box in face_boxes:
        face = frame[max(0, face_box[1]):min(face_box[3], frame.shape[0] - 1),
                     max(0, face_box[0]):min(face_box[2], frame.shape[1] - 1)]
        
        # Prepare the face for gender and age prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_buckets[age_preds[0].argmax()]

        # Display the results on the frame
        label = f'{gender}, {age}'
        cv2.putText(result_img, label, (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Age and Gender Prediction", result_img)

    key = cv2.waitKey(1)
    if key == 27:  # Exit if 'ESC' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
