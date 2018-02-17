from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def model1(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, kernel_size=(2, 2), padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model2(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(2, 2), padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model3(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, kernel_size=(2, 2), padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model4(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(2, 2), padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model