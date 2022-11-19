# TODO:
# Create a function to take parameter inputs and return a trained cnn model, possibly check if the model already exists, possibly return some diagnostic data
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt

def cnn(dataset):
	# TODO: For ease of generic use, possibly have all models be in the same form, but unused data is blank / neutral, if one sample of spectrum apply it to all spots

	# split datasets
  training_set = preprocessing.image_dataset_from_directory(dataset,
                                                            validation_split=0.2,
                                                            subset="training",
                                                            label_mode="categorical",
                                                            seed=0,
                                                            image_size=(100,100))#TODO: Does this work with our modified images, need to change
  test_set = preprocessing.image_dataset_from_directory(dataset,
                                                            validation_split=0.2,
                                                            subset="validation",
                                                            label_mode="categorical",
                                                            seed=0,
                                                            image_size=(100,100))#TODO: Does this work with our modified images, need to change
  #TODO: Need to look into pretrained CNN to work off of
	# build the model #TODO: All of this will need to be reworked
	m = Sequential()
	m.add(Rescaling(1/255))
	m.add(Conv2D(32, kernel_size=(3, 3),
	             activation='relu',
	             input_shape=(100,100,3)))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # I believe encoder will be similar just without these convolutional layers
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dropout(0.5))
	m.add(Dense(5, activation='softmax'))

	# setting and training
	m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
	history  = m.fit(training_set, batch_size=32, epochs=25,verbose=0)
	print(history.history["accuracy"])
	print(training_set.class_names)

	# testing
	print("Testing.")
	score = m.evaluate(test_set, verbose=0)
	print('Test accuracy:', score[1])

	return m

def cnn_test(m, image_file) : #TODO: All of this needs to be updated
	""" Classifies the given image using the given model. """
	# load the image
	img = preprocessing.image.load_img(image_file,target_size=(100,100))
	img_arr = preprocessing.image.img_to_array(img)

	# show the image
	plt.imshow(img_arr.astype("uint8"))
	plt.show()

	# classify the image
	img_cl = img_arr.reshape(1,100,100,3)
	score = m.predict(img_cl)
	print(score)