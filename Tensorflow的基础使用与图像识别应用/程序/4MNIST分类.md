

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```


```python
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
```

    Extracting MNIST_data\train-images-idx3-ubyte.gz
    Extracting MNIST_data\train-labels-idx1-ubyte.gz
    Extracting MNIST_data\t10k-images-idx3-ubyte.gz
    Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
    Iter 0,Testing Accuracy 0.8304
    Iter 1,Testing Accuracy 0.8705
    Iter 2,Testing Accuracy 0.8808
    Iter 3,Testing Accuracy 0.8886
    Iter 4,Testing Accuracy 0.8938
    Iter 5,Testing Accuracy 0.8966
    Iter 6,Testing Accuracy 0.9004
    Iter 7,Testing Accuracy 0.9019
    Iter 8,Testing Accuracy 0.9034
    Iter 9,Testing Accuracy 0.9054
    Iter 10,Testing Accuracy 0.9066
    Iter 11,Testing Accuracy 0.9071
    Iter 12,Testing Accuracy 0.9084
    Iter 13,Testing Accuracy 0.9094
    Iter 14,Testing Accuracy 0.91
    Iter 15,Testing Accuracy 0.9103
    Iter 16,Testing Accuracy 0.9118
    Iter 17,Testing Accuracy 0.913
    Iter 18,Testing Accuracy 0.9129
    Iter 19,Testing Accuracy 0.9131
    Iter 20,Testing Accuracy 0.9135
    


```python

```
