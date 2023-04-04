import numpy as np


from keras.preprocessing.image import ImageDataGenerator
import keras.utils
import keras.datasets.mnist
from sklearn.manifold import TSNE
from keras import layers
import keras.backend as K

# V2版本使用数据增强器生成正采样，V1版本使用随机采样
input_shape = (28, 28, 1)
epochs = 10
#对比损失函数
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# 定义数据增强器用于生成正采样
data_augmentation = ImageDataGenerator(
    rotation_range=10,  # 随机旋转角度范围
    width_shift_range=0.2,  # 随机水平平移范围
    height_shift_range=0.2,  # 随机竖直平移范围
    horizontal_flip=True,  # 随机水平翻转
    vertical_flip=True,  # 随机竖直翻转
    fill_mode='nearest'  # 填充模式
)
# 自定义DataGenerater生成minibatch的样本图片和标签进行训练
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: anchor 锚点 positive 正样本 negative 负样本
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchor = batch_x
        pos = data_augmentation.flow(anchor, shuffle=False, batch_size=self.batch_size).next()

        return [anchor, pos], batch_y


# 加载数据集
path = r'D:\project\python\Keras-MINIST\mnist.npz'
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path)

# 归一化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# 转为28×28×1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 定义用于预训练的模型体系结构
anchor_input = layers.Input(shape=input_shape, name="anchor_input")
positive_input = layers.Input(shape=input_shape, name="positive_input")

encoder = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
    ],
    name="encoder",
)
# 将锚点，正样本，负样本传入编码器
encoded_anchor = encoder(anchor_input)
encoded_positive = encoder(positive_input)

merged_output = layers.concatenate([encoded_anchor, encoded_positive], axis=-1, name="merged_layer")
pretraining_model = keras.Model(inputs=[anchor_input, positive_input], outputs=merged_output,
                                name="triplet_model")

# 使用对比损失函数编译模型
pretraining_model.compile(optimizer='adam', loss=contrastive_loss)

# 使用小批量数据生成器训练模型
batch_size = 128
data_generator = DataGenerator(x_train, y_train, batch_size)
pretraining_model.fit(data_generator, epochs=epochs)

# 获取编码器
encoder = pretraining_model.get_layer("encoder")
# 使用预训练模型提取编码特征
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)

# 使用 TSNE 可视化
tsne = TSNE(n_components=2, random_state=42)
x_encoded = np.concatenate((x_train_encoded, x_test_encoded), axis=0)
y_encoded = np.concatenate((y_train, y_test), axis=0)
x_encoded_tsne = tsne.fit_transform(x_encoded)
import matplotlib.pyplot as plt

plt.scatter(x_encoded_tsne[:, 0], x_encoded_tsne[:, 1], c=y_encoded)
plt.show()

# 定义分类的模型结构
inputs = keras.Input(shape=(64,))
outputs = keras.layers.Dense(10, activation='softmax')(inputs)
classification_model = keras.Model(inputs=inputs, outputs=outputs)

# 使用分类交叉熵损失函数编译模型
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classification_model.fit(x_train_encoded, y_train, batch_size=batch_size, epochs=epochs)

# 在测试集上评估模型
test_loss, test_acc = classification_model.evaluate(x_test_encoded, y_test)
print('Test accuracy:', test_acc)