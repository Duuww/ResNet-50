# 设置GPU
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import os
import warnings
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,BatchNormalization,Activation,Dense
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import sys
##############################################################################################################
gpus = tf.config.list_physical_devices("GPU")

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

warnings.filterwarnings("ignore")  # 忽略警告信息

plt.rcParams['axes.unicode_minus'] = False

###############################################################################################################
train_dir = './flower/train'
valid_dir = './flower/valid'

# Path将文件或者文件夹路径（str）转换为Path对象，解决实不同OS路径连接符的问题
train_dir = pathlib.Path(train_dir)
# 通过glob函数查看train_dir下的所有文件
image_count = len(list(train_dir.glob('*/*')))
print("训练集图片总数为：",image_count)

valid_dir = pathlib.Path(valid_dir)
# 通过glob函数查看valid_dir下的所有文件
image_count = len(list(valid_dir.glob('*/*')))
print("验证集图片总数为：",image_count)

# 批处理量
batch_size = 30
# 设置预处理图片大小
img_height = 224
img_width = 224

# 训练集的数据增强
train_dsgen = ImageDataGenerator(
    # 通过将图像像素值缩放到 [0, 1] 范围内，以确保输入数据在神经网络中的值范围合理。
    rescale=1.0/255,
    # 随机旋转图像最多30度
    rotation_range=30,
    # 以50%的概率随机水平翻转图像。
    horizontal_flip=True,
    # 最近邻插值来填充由于平移等操作而导致的图像边界的空白像素。
    fill_mode='nearest'
)
# 验证集的数据增强
valid_dsgen = ImageDataGenerator(
    rescale=1.0/255
)

# 训练集的Tensorflow数据集对象
# 会返回训练集多少图片，多少标签
train_ds = train_dsgen.flow_from_directory(
    train_dir,
    # 训练集图片大小
    target_size=(img_width,img_height),
    batch_size=batch_size,
    # 标签采用独热编码的形式提供，适用于多分类任务。这确保了模型能够正确理解标签。
    class_mode='categorical',
    # 数据打乱
    shuffle=True,
    seed= 17
)

valid_ds = valid_dsgen.flow_from_directory(
    valid_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

def get_subdirectories(path):
    # 列出给定路径下的所有项，并只保留子文件夹
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
path = './flower/train'
subdirs = get_subdirectories(path)
print(subdirs,'\n共有',len(subdirs), '种')

##################################################################################################################
# 加载预训练模型，进行迁移学习

# 加载一个在 ImageNet 数据集上预训练过的 ResNet50 模型，不包括顶层分类器
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
# resnet50模型所有层可训练
for layer in base_model.layers:
    layer.trainable = True

# 添加了一些新的全连接层，来适应特定任务
# 自定义层添加的起点
X = base_model.output
# 将特征图的维度降低，计算每个特征图的平均值，帮助减少模型参数，减轻过拟合
X = GlobalAveragePooling2D()(X)

# 添加一个具有 512 个神经元的全连接层，使用 He 均匀初始化方法初始化权重
# 提供模型以进一步学习和组合特征的能力
X = Dense(512, kernel_initializer='he_uniform')(X)
# 批量归一化，标准化前一层的输出，加速训练过程，提高模型的稳定性
X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Dense(102, kernel_initializer='he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

# 添加一个全连接层，神经元数量等于subdirs的长度，使用softmax激活函数
# 输出层，用于多分类任务。Softmax 激活函数用于计算每个类别的概率分布
output = Dense(len(subdirs), activation='softmax')(X)

model = Model(inputs=base_model.input, outputs=output)

# 优化器设置，学习率设置
optimizer = tf.keras.optimizers.Adam(lr=1e-6)

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# sys.exit(0)

model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(train_ds,
                  validation_data=valid_ds,
                  epochs=60
                    )

model.save("resnet50_model.h5")

# 绘制loss曲线图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
# 绘制accuracy曲线图
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()
