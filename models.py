from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def model1(width, height):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model2(width, height):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

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
    model.add(Conv2D(16, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model4(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model5(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model6(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model7(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model8(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model9(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Asegurarse de que la salida sea de 1 x 1 x (num canales). Normalmente se termina en una convolucion (no un pooling).

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model10(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model11(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model12(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model13(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model14(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model15(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model16(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model17(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model18(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model19(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model20(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model21(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model22(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model23(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model24(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model25(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model26(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model27(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model28(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model29(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model30(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model31(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model32(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model33(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model34(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(9, 9), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(7, 7), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model35(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model36(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model37(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model38(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model39(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model40(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model41(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model42(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model43(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model44(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model45(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model46(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model47(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model48(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model49(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model50(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model51(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model52(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model53(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model54(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model55(width, height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model56(width, height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model57(width, height):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def model58(width, height):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model