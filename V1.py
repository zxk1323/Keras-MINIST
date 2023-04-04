import numpy as np
import keras.backend as K
from tensorflow import keras
from sklearn.manifold import TSNE

input_shape = (28, 28, 1)
#对比损失函数
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

#自定义DataGenerater生成minibatch的样本图片和标签进行训练
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
#加载数据集
path = r'D:\project\python\Keras-MINIST\mnist.npz'
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path)

#归一化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#转为28×28×1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#定义用于预训练的模型体系结构
inputs = keras.Input(input_shape)
x = keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
encoded = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(64)(encoded)
pretraining_model = keras.Model(inputs=inputs, outputs=outputs)

#使用对比损失函数编译模型
pretraining_model.compile(optimizer='adam', loss=contrastive_loss)

#使用小批量数据生成器训练模型
batch_size = 128
data_generator = DataGenerator(x_train, y_train, batch_size)
pretraining_model.fit(data_generator, epochs=20)

#使用预训练模型提取编码特征
encoder = keras.Model(inputs=inputs, outputs=encoded)
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)

#使用 TSNE 可视化
tsne = TSNE(n_components=2, random_state=42)
x_encoded = np.concatenate((x_train_encoded, x_test_encoded), axis=0)
y_encoded = np.concatenate((y_train, y_test), axis=0)
x_encoded_tsne = tsne.fit_transform(x_encoded)
import matplotlib.pyplot as plt

plt.scatter(x_encoded_tsne[:, 0], x_encoded_tsne[:, 1], c=y_encoded)
plt.show()

#定义分类的模型结构
inputs = keras.Input(shape=(64,))
outputs = keras.layers.Dense(10, activation='softmax')(inputs)
classification_model = keras.Model(inputs=inputs, outputs=outputs)

#使用分类交叉熵损失函数编译模型
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classification_model.fit(x_train_encoded, y_train, batch_size=batch_size, epochs=10)

#在测试集上评估模型
test_loss, test_acc = classification_model.evaluate(x_test_encoded, y_test)
print('Test accuracy:', test_acc)