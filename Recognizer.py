from __future__ import print_function
import sys
import cv2 as cv
import argparse
import json
import os
import numpy

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default=r'MachineLearning\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default=r'MachineLearning\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')

parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade

face_cascade = cv.CascadeClassifier(face_cascade_name)
eyes_cascade = cv.CascadeClassifier(eyes_cascade_name)


with open('file.json') as f:
	database=json.load(f)

def train_model(label):
	# Create lists to store the face samples and their corresponding labels
	faces = []
	labels = []
	
	# Load the images from the 'Faces' folder
	for file_name in os.listdir('Faces'):
		if file_name.endswith('.jpg'):
			# Extract the label (person's name) from the file name
			name = file_name.split('.')[0]
			
			# Read the image and convert it to grayscale
			image = cv.imread(os.path.join('Faces', file_name))
			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

			# Detect faces in the grayscale image
			detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

			# Check if a face is detected
			if len(detected_faces) > 0:
				# Crop the detected face region
				face_crop = gray[detected_faces[0][1]:detected_faces[0][1] + detected_faces[0][3],
				    detected_faces[0][0]:detected_faces[0][0] + detected_faces[0][2]]

				# Append the face sample and label to the lists
				faces.append(face_crop)
				labels.append(label[name])

	# Train the face recognition model using the faces and labels
	recognizer = cv.face.LBPHFaceRecognizer_create()
	recognizer.train(faces, numpy.array(labels))

	# Save the trained model to a file
	recognizer.save('trained_model.xml')
	return recognizer

# Train the model
Recognizer =train_model(database)

# Function to recognize faces
def recognize_faces(recognizer, label):
	# Open the camera
	cap = cv.VideoCapture(1)
	
	# Reverse keys and values in the dictionary
	label_name = {value: key for key, value in label.items()}
	while True:
		# Read a frame from the camera
		ret, frame = cap.read()

		# Convert the frame to grayscale
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Detect faces in the grayscale frame
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
		
		# Recognize and label the faces
		for (x, y, w, h) in faces:
			# Recognize the face using the trained model
			label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
			#print(confidence)
			if confidence > 40:
				# Display the recognized label and confidence level
				cv.putText(frame, label_name[label], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	
				# Draw a rectangle around the face
				cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				print(f'Recognized {label_name[label]} with confidence {confidence}')
				#commentare sys.exit per while true loop
				sys.exit(0)
			else:
				#print('Unrecognized')
				pass
		# Display the frame with face recognition
		cv.imshow('Recognize Faces', frame)

		# Break the loop if the 'q' key is pressed
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	# Release the camera and close windows
	cap.release()
	cv.destroyAllWindows()

recognize_faces(Recognizer, database)