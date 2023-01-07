import tensorflow as tf
from settings import EPOCHS

def trainNumReader():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  #10 because dataset is numbers from 0 - 9


    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # training
    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=EPOCHS)

    # evaluation
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)
    model.save('numReader.model')