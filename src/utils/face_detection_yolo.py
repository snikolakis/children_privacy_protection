import cv2
import gdown
import shutil
import os
import sys
from utils import *


PATH_TO_DATASET = './yolo_test'
OUTPUT_PATH = './faces_extracted_yolo_test'
PATH_TO_YOLOFACE = './yoloface'
PATH_TO_CFG = PATH_TO_YOLOFACE + '/cfg/yolov3-face.cfg'
PATH_TO_WEIGHTS = PATH_TO_YOLOFACE + '/model-weights/yolov3-wider_16000.weights'


def create_directory(path):
	# create output directory if it does not already exist
	if not os.path.exists(path):
		os.makedirs(path)


def get_pretrained_network():
	# create network from configuration and weights files
	net = cv2.dnn.readNetFromDarknet(PATH_TO_CFG, PATH_TO_WEIGHTS)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	return net


def extract_faces_from_image(image):
	create_directory(OUTPUT_PATH)
	net = get_pretrained_network()
	cap = cv2.VideoCapture(image)
	output_file_base = image.rsplit('/')[-1].rsplit('.')[0]
	output_dir = OUTPUT_PATH + '/' + output_file_base
	_, frame = cap.read()
	# create a 4D blob from a frame
	blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
									[0, 0, 0], 1, crop=False)

	# set the network's input
	net.setInput(blob)

	# forward pass
	outs = net.forward(get_outputs_names(net))

	# remove the bounding boxes with low confidence
	faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

	# extract and save faces
	if len(faces) > 0:
		# create directory with the image name
		create_directory(output_dir)
		for i, face in enumerate(faces):
			cropped = frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
			cv2.imwrite(output_dir + f'/face_{i + 1:03}.jpeg', cropped)

	cap.release()


if __name__ == '__main__':
	if not os.path.isfile(PATH_TO_WEIGHTS):
		create_directory(PATH_TO_YOLOFACE + '/model-weights')
		weights_url = "https://drive.google.com/uc?export=download&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX"
		gdown.download(weights_url, PATH_TO_YOLOFACE + '/model-weights/yolov3-wider_16000.weights.zip', quiet=False)
		shutil.unpack_archive(PATH_TO_YOLOFACE + '/model-weights/yolov3-wider_16000.weights.zip', PATH_TO_YOLOFACE + '/model-weights/')
	
	images = [f for f in os.listdir(PATH_TO_DATASET) if os.path.isfile(os.path.join(PATH_TO_DATASET, f))]
	for image in images:
		print(f'Extracting faces from image "{image}"')
		try:
			extract_faces_from_image(os.path.join(PATH_TO_DATASET, image))
		except:
			print(f'Couldn\'t extract faces from image "{image}"')
