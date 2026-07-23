from __future__ import print_function, division
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential
from utils import *

channels = 2
# c denotes channels to be trained, 0=geochemistry, 2=+DEM+slope, 3= +granites

fulldata, samples, labels = read_data(augmentation=10, rate=0.15, windowsize=32, c=channels)
# augmentation denotes the times to be augmented, 1 for no-augmentation

base = 32

model = Sequential()

model.add(Conv2D(base, (3, 3), strides=(1, 1), padding='same', input_shape=(32, 32, 25+channels)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Conv2D(base, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Conv2D(base * 2, (2, 2), strides=(1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Flatten())
model.add(Dense(base * 8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

Adam = Adam(lr=1e-5)
model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
model.fit(samples, labels, batch_size=128, epochs=500, verbose=2, validation_split=0.2)


pres = []
for i in range(92416):
    a = read_pre(fulldata, i)
    pres.append(a)
x = model.predict(np.asarray(pres, np.float32), verbose=2)
result = np.reshape(x, [304, 304])
np.savetxt('./pre.csv', x, delimiter=',')

deposits = pd.read_csv('./data/deposit.csv', header=None)
deposits = deposits.values


plt.imshow(result, cmap='jet', origin='lower')
plt.scatter(deposits[:, 0]-16, deposits[:, 1]-16, c='g', marker='^')
plt.show()
