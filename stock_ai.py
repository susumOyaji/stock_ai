from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import math
 
data_dir = 'data/chart' #データフォルダの名前
base_dir = "data/" #ログ出力用
def_batch_size = 1 #バッチサイズ

#データの偏りを重みで表現
def weight(classes_name, dir_name):
    data_element_num = {}
    max_buf = 0
    for class_name in classes_name:
        class_dir = dir_name + os.sep + class_name
        files = os.listdir(class_dir)
        data_element_num[class_name] = len(files)
        if max_buf < len(files):
            max_buf = len(files)
    weights = {}
    count = 0
    for class_name in classes_name:
        weights[count] = round(float(math.pow(data_element_num[class_name]/max_buf, -1)), 2)
        count = count + 1
    return weights
 
#モデルの定義
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])
 
#検証用データ(validation_split)を0.3で設定
datagen = ImageDataGenerator(validation_split=0.3, rescale=1./255)
 
#学習用データ
train_generator = datagen.flow_from_directory(
       data_dir,
       batch_size=def_batch_size,
       class_mode='binary',
       target_size=(256, 256),
       color_mode='grayscale',
       subset='training')
 
#検証用データ
validation_generator = datagen.flow_from_directory(
       data_dir,
       batch_size=def_batch_size,
       class_mode='binary',
       target_size=(256, 256),
       color_mode='grayscale',
       subset='validation')
 
for data_batch, labels_batch in train_generator:
   print('data batch shape:', data_batch.shape)
   print('labels batch shape:', labels_batch.shape)
   break
 
fpath = base_dir + 'chart.{epoch:02d}.h5'
modelCheckpoint = ModelCheckpoint(filepath = fpath,
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  save_freq='epoch')
 
class_weights = weight(classes_name = ['down', 'up'], dir_name = data_dir)
print('class weight:', class_weights)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // def_batch_size,
    validation_data = validation_generator,
    epochs = 100,
    validation_steps=validation_generator.samples // def_batch_size,
    class_weight=class_weights,
    callbacks=[modelCheckpoint])
 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
fig = plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(base_dir + 'accuracy.png')
plt.close()
 
fig = plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(base_dir + 'loss.png')
plt.close()