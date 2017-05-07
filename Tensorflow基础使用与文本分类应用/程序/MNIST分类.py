
# coding: utf-8

# 51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html<br>
# 优酷频道：http://i.youku.com/sdxxqbf<br>
# 微信公众号：深度学习与神经网络<br>
# Github：https://github.com/Qinbf<br>

# In[2]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次100张照片
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
#square是求平方
#reduce_mean是求平均值
loss = tf.reduce_mean(tf.square(y-prediction))

#使用梯度下降法来最小化loss，学习率是0.2
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast是进行数据格式转换，把布尔型转为float32类型

with tf.Session() as sess:
    #执行初始化
    sess.run(init)
    #迭代21个周期
    for epoch in range(21):
        #每个周期迭代n_batch个batch，每个batch为100
        for batch in range(n_batch):
            #获得一个batch的数据和标签
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            #通过feed喂到模型中进行训练
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        #计算准确率
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:



