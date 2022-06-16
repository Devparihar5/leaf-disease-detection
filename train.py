#importing libraries

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


#basic cnn layers
model = Sequential()
model.add(Conv2D(32, kernel_size= (3,3), activation = 'relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size =(2,2,)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size= (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2,)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size= (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2,)))
model.add(BatchNormalization())
model.add(Conv2D(96,kernel_size= (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2,)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size= (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2,)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(38, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#start training
train_datagen = ImageDataGenerator(rescale=None,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./225)

training_set=train_datagen.flow_from_directory("E:/AI and ML-pr/day14/dataset/train",
                                               target_size=(128,128),
                                               batch_size=32,
                                               class_mode='categorical')
labels=(training_set.class_indices)
test_set=test_datagen.flow_from_directory('E:/AI and ML-pr/day14/dataset/test',
                                               target_size=(128,128),
                                               batch_size=32,
                                               class_mode='categorical')

labels2=(test_set.class_indices)
model.fit(training_set,
                    steps_per_epoch=375,
                    epochs=10,
                    validation_data = test_set,
                    validation_steps = 125)


model_json=model.to_json()
with open("modell.json","w") as json_file:
    json_file.write(model_json)
    model.save_weights("modell.h5")
    print("Saved model to disk")
    












