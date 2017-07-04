
# coding: utf-8

# 51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html<br>
# 优酷频道：http://i.youku.com/sdxxqbf<br>
# 微信公众号：深度学习与神经网络<br>
# Github：https://github.com/Qinbf<br>

# In[1]:

import tensorflow as tf
import numpy as np
import os
import time
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from six.moves import xrange


# In[2]:

# Parameters
# ==================================================

# Data loading params
# validation数据集占比
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# 数据集
tf.flags.DEFINE_string("data_file", "./ieee_zhihu_cup/data_topic_block_0.txt", "Data source for the positive data.")

# Model Hyperparameters
# 词向量长度
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
# 卷积核大小
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# 每一种卷积核个数
tf.flags.DEFINE_integer("num_filters", 1024, "Number of filters per filter size (default: 1024)")
# dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# l2正则化参数
tf.flags.DEFINE_float("l2_reg_lambda", 0.0005, "L2 regularization lambda (default: 0.0005)")

# Training parameters
# 批次大小
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 迭代周期
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
# 多少step测试一次
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
# 多少step保存一次模型
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 200)")
# 保存多少个模型
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# flags解析
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# 打印所有参数
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# In[3]:

y = []
x_text = []

# 读取训练数据和标签
reader = pd.read_table(FLAGS.data_file,sep='\t',header=None)
for i in tqdm(xrange(reader.shape[0])):
    # 按','切分标签
    temp = reader.iloc[i][1].split(',')
    # 如果分类数大于5，只取前5个分类
    if (len(temp)>5):
        temp = temp[0:5]
    # 设置标签的对应位置为1，其余位置为0
    label = np.zeros(1999)
    for temp_label in temp:
        label[int(temp_label)] = 1
    y.append(label)
    x_text.append(reader.iloc[i][0])


# In[4]:

# 打印x_text和y的前5行
print(x_text[0:5])
y = np.array(y, dtype = np.float32)
print(y[0:5])


# In[5]:

# Build vocabulary
# 计算一段文本中最多的词汇数
max_document_length = max([len(x.split(",")) for x in x_text])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)

x = np.array(list(vocab_processor.fit_transform(x_text)))
print("x_shape:",x.shape)
print("y_shape:",y.shape)

# 保存字典
vocab_processor.save("vocab_dict")

# Split train/test set
# 数据集切分为两部分，训练集和验证集
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("x:",x_train[0:5])
print("y:",y_train[0:5])


# In[6]:

# 定义三个placeholder
input_x = tf.placeholder(tf.int32, [None, x_train.shape[1]], name="input_x")
input_y = tf.placeholder(tf.float32, [None, y_train.shape[1]], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# sequence_length-最长词汇数
sequence_length=x_train.shape[1]
# num_classes-分类数
num_classes=y_train.shape[1]
# vocab_size-总词汇数
vocab_size=len(vocab_processor.vocabulary_)
# embedding_size-词向量长度
embedding_size=FLAGS.embedding_dim
# filter_sizes-卷积核尺寸3，4，5
filter_sizes=list(map(int, FLAGS.filter_sizes.split(",")))
# num_filters-卷积核数量
num_filters=FLAGS.num_filters
        
Weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Weights")
# shape:[None, sequence_length, embedding_size]
embedded_chars = tf.nn.embedding_lookup(Weights, input_x)
# 添加一个维度，shape:[None, sequence_length, embedding_size, 1]
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(
            tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
print("num_filters_total:", num_filters_total)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
with tf.name_scope("dropout"):h_drop = tf.nn.dropout(h_pool_flat,dropout_keep_prob)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    
# 定义loss
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=input_y))

# 定义优化器
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)


# In[7]:

# 生成批次数据
def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:",num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[ ]:

# 知乎提供的评测方案
def eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    
    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0   #总问题数量
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return 2*(precision * recall) / (precision + recall )


# In[ ]:

# 定义saver，只保存最新的5个模型
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

with tf.Session() as sess:
    predict_top_5 = tf.nn.top_k(scores, k=5)
    label_top_5 = tf.nn.top_k(input_y, k=5) 
    sess.run(tf.global_variables_initializer())
    i = 0
    # 生成数据
    batches = batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    for batch in batches:
        i = i + 1
        # 得到一个batch的数据
        x_batch, y_batch = zip(*batch)
        # 优化模型
        sess.run([optimizer],feed_dict={input_x:x_batch, input_y:y_batch, dropout_keep_prob:FLAGS.dropout_keep_prob})

        # 每训练50次测试1次
        if (i % FLAGS.evaluate_every == 0):
            print ("Evaluation:step",i)
            predict_5, label_5, _loss = sess.run([predict_top_5,label_top_5,loss],feed_dict={input_x:x_batch,
                                                                                      input_y:y_batch,
                                                                                      dropout_keep_prob:1.0})
            print ("label:",label_5[1][:5])
            print ("predict:",predict_5[1][:5])
            print ("predict:",predict_5[0][:5])
            print ("loss:",_loss)
            predict_label_and_marked_label_list = []
            for predict,label in zip(predict_5[1],label_5[1]):
                predict_label_and_marked_label_list.append((list(predict),list(label)))
            score = eval(predict_label_and_marked_label_list)
            print("score:",score)

        # 每训练200次保存1次模型
        if (i % FLAGS.checkpoint_every == 0):
            path = saver.save(sess, "models/model", global_step=i)
            print("Saved model checkpoint to {}".format(path))


# In[ ]:
