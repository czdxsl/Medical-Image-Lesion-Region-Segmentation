from keras import layers
from keras import models
from load_dataset import load_dataset
from keras.utils.vis_utils import plot_model


def train_and_test():
    print("构建模型")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(63, 63, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    print("开始导入数据集")
    x_train, y_train, x_test, y_test = load_dataset()
    print("数据集导入完成")
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=400)
    print("训练完成")
    print(model.summary())
    model.save('CAImodel.h5')
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print("loss:")
    # print(test_loss)
    # print("acc:")
    # print(test_acc)

