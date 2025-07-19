# -*- coding: utf-8 -*-
# FileName  : image_check.py
import tensorflow as tf
from .conf import model_path
import numpy as np

model = tf.keras.models.load_model(model_path)

class_name = ['一品红', '万寿菊', '三色堇', '乌头', '仙客来', '光叶蝴蝶草', '六出花', '凌霄', '凤梨', '刺芹', '勋章菊', '卡特兰', '卷丹', '叶子花', '向日葵', '唐菖蒲', '嘉兰', '大丽花', '大星芹', '天人菊', '天竺葵', '天蓝绣球', '姜荷花', '射干', '山姜', '山桃草', '山茶', '帝王花', '平贝母', '康乃馨', '德国鸢尾', '报春花', '换锦花', '旱金莲', '月见草', '木槿', '朱槿', '朱顶红', '杜鹃', '松果菊', '桂竹香', '桔梗', '款冬', '毛地黄', '毛曼陀罗', '毛莨', '水塔花', '沙漠玫瑰', '油点草', '洋桔梗', '海菜花', '牵牛花', '玉兰', '玫瑰', '瓜叶菊', '番红花', '百日菊', '睡莲', '石竹', '硬叶兜兰', '碧冬茄', '秋英', '红掌', '罂粟花', '美人蕉', '美国薄荷', '耧斗菜', '肿柄菊', '花烛', '花菱草', '花葵', '荷花', '菜蓟', '葡萄风信子', '蓝刺头', '蓝玉簪龙胆', '蓝盆花', '蓟', '蝴蝶兰', '西番莲', '野棉花', '金盏花', '金莲花', '金鱼草', '铁筷子', '铁线莲', '银灰旋花', '银莲花', '闭鞘姜', '雏菊', '非洲菊', '风铃草', '香豌豆', '马蹄莲', '鸡冠花', '鸡蛋花', '鹤望兰', '黄水仙', '黄菊花', '黄菖蒲', '黑心菊', '黑种草']

def load_and_preprocess_image(path):
    # 图像预处理
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def check_handle(img_path):
    # 图像预测
    test_img = img_path
    test_tensor = load_and_preprocess_image(test_img)
    # 模型预测数据集对象中的第一个，即上传图片的tensor
    test_tensor = tf.expand_dims(test_tensor, axis=0)
    pred = model.predict(test_tensor)
    # 找到权重最大的索引值
    pred_num = np.argmax(pred)
    # 通过索引值找到模型所预测的类名
    result = class_name[int(pred_num)]
    return result
