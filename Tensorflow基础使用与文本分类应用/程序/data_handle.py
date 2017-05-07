
# coding: utf-8

# 51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html<br>
# 优酷频道：http://i.youku.com/sdxxqbf<br>
# 微信公众号：深度学习与神经网络<br>
# Github：https://github.com/Qinbf<br>
# 
# question_train_set.txt：  
#     第一列为 问题id；  
#     第二列为 title 的字符编号序列；  
#     第三列为 title 的词语编号序列；  
#     第四列为描述的字符编号序列；  
#     第五列为描述的词语标号序列。  
#     
# question_topic_train_set.txt：  
#     第一列 问题 id；  
#     第二列 话题 id。  
# 
# topic_info.txt：  
#     第一列为话题 id  
#     第二列为话题的父话题 id。话题之间是有向无环图结构，一个话题可能有 0 到多个父话题；  
#     第三列为话题名字的字符编号序列；  
#     第四列为话题名字的词语编号序列；  
#     第五列为话题描述的字符编号序列；  
#     第六列为话题描述的词语编号序列。  
# 
# 1.title通常来说包含的信息最重要。对于question_train_set.txt文件，为了简单起见，我们只取第三列，title的词语编号序列。    
# 2.对于topic_info.txt，为了简单起见，我们不考虑2,3,4,5,6列。只是简单的提取话题id，然后转为0-1998的数字（一共有1999个话题）  
# 3.然后合并以上一些数据，得到最后处理后的数据。  

# In[1]:

import pandas as pd
from tqdm import tqdm # pip install tqdm
from six.moves import xrange


# In[2]:

# 导入question_train_set
reader = pd.read_table('./ieee_zhihu_cup/question_train_set.txt',sep='\t',header=None)
print(reader.iloc[0:5])


# In[3]:

# 导入question_topic_eval_set
topic_reader = pd.read_table('./ieee_zhihu_cup/question_topic_train_set.txt',sep='\t',header=None)
print(topic_reader.iloc[0:5])


# In[4]:

# 合并title 的词语编号序列和话题 id
data_topic = pd.concat([reader.ix[:,2], topic_reader.ix[:,1]], axis=1, ignore_index=True)
print(data_topic.iloc[0:5])


# In[5]:

# 导入topic_info
label_reader = pd.read_table('./ieee_zhihu_cup/topic_info.txt',sep='\t',header=None)
print(label_reader.iloc[0:5])


# In[6]:

# 把标签转为0-1998的编号
labels = list(label_reader.iloc[:,0])
my_labels = []
for label in labels:
    my_labels.append(label)
    
# 建立topic字典
topic_dict = {}
for i,label in enumerate(my_labels):
    topic_dict[label] = i

print(topic_dict[7739004195693774975])


# In[7]:

for i in tqdm(xrange(data_topic.shape[0])):
    new_label = ''
    # 根据“,”切分话题id
    temp_topic = data_topic.iloc[i][1].split(',')
    for topic in temp_topic:
        # 判断该label是否在label文件中，并得到该行
        label_num = topic_dict[int(topic)]
        new_label = new_label + str(label_num) + ','
    data_topic.iloc[i][1] = new_label[:-1]
print(data_topic.iloc[:5])


# In[8]:

# 保存处理过后的文件
data_topic.to_csv("./ieee_zhihu_cup/data_topic.txt", header=None, index=None, sep='\t')

# 切分成10块保存
for i in xrange(10):
    data_topic_filename = './ieee_zhihu_cup/data_topic_block_' + str(i) + '.txt'
    if (i+1)*300000 < data_topic.shape[0]:
        data_topic.iloc[i*300000:(i+1)*300000].to_csv(
            data_topic_filename, header=None, index=None, sep='\t')
    else:
        data_topic.iloc[i*300000:data_topic.shape[0]].to_csv(
            data_topic_filename, header=None, index=None, sep='\t')


# In[ ]:



