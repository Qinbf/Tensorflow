
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

#创建一个常量op
m1 = tf.constant([[3,3]])
#创建一个常量op
m2 = tf.constant([[2],[3]])
#创建一个矩阵乘法op，把m1和m2传入
product = tf.matmul(m1,m2)
#这个时候打印product，只能看到product的属性，不能计算它的值
print(product)


# In[3]:

#第一种定义会话的方式：
#定义一个会话，启动默认图
sess = tf.Session()
#调用sess的run方法来执行矩阵乘法op
#run(product)触发了图中3个op
result = sess.run(product)
print(result)
sess.close()


# In[4]:

#第二种定义会话的方式：
with tf.Session() as sess:
    #调用sess的run方法来执行矩阵乘法op
    #run(product)触发了图中3个op
    result = sess.run(product)
    print(result)


# In[ ]:



