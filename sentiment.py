#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from nltk import word_tokenize
from keras import backend as K
from keras.preprocessing import sequence 
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense,Dropout 
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt

#1)preprocessing
#read data / split string
#mood=0:negative,mood=1:positive
test = pd.DataFrame(columns = ['mood','sentence'])
testfile = open('testing_label.txt','r')
line = testfile.readline()
while line:
    if line != '\n':
        temp = line.split("#####", 1)
        new = pd.DataFrame({'mood':temp[0],'sentence':temp[1]},index=[1])
        test = test.append(new,ignore_index=True)
    line = testfile.readline()
testfile.close()

train = pd.DataFrame(columns = ['mood','sentence'])
trainfile = open('training_label.txt','r',encoding='utf8')
line = trainfile.readline()
i = 0
while line:
    if line != '\n':
        i+=1
        temp = line.split("+++$+++", 1)
        new = pd.DataFrame({'mood':temp[0],'sentence':temp[1]},index=[1])
        train = train.append(new,ignore_index=True)
    if i == 10000 : break 
    line = trainfile.readline()
trainfile.close()

#remove stop words
stop_words = [',','.','..','...',"'",'`',':','1','2','3','4','5','6','7','8','9','0',
              '00','000','0000','000pv','day','so','all','up','got','today','from','one',
              'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
              't','u','v','w','x','y','z','im','you','your','u','ur','they','an','him',
              'we','he','she','me','my','his','her','it','is','am','are','was','were',
              'btw','thing','be','have','has','had','do','does','did','ll','re','ve',
              'this','that','there','say','says','said','to','for','in','at',
              'when','the','on','its','and','in','of','some','someone','before','after',
              'with','been','being','which','them','their','left','10','these',
              '30','site','online','12','da','20','sometimes','11','sat','15','google'
              '24','info','sit','web','website','09','17','18','33','50','16','21',
              '25','b4','13','333','45','70','2009','month','yr','yrs','23',
              'txt','06','22','26','29','31','37','80','07','19','28',
              '32','34','35','36','38','48','52','69','79','89','95','97']
for i in range(len(test)):
    test_sent = test['sentence'][i] 
    word_tokens = word_tokenize(test_sent)  
    sentence='' 
    for w in word_tokens: 
        if w not in stop_words: 
            sentence += w + ' '
    test['sentence'][i] = sentence
for i in range(len(train)):
    train_sent = train['sentence'][i] 
    word_tokens = word_tokenize(train_sent)   
    sentence=''
    for w in word_tokens: 
        if w not in stop_words: 
            sentence += w + ' ' 
    train['sentence'][i] = sentence

train_y = train.mood
train_x = train.sentence
test_y = test.mood
test_x = test.sentence 

token = Tokenizer(num_words=4000)
token.fit_on_texts(train_x)
token.word_index
x_train_seq = token.texts_to_sequences(train_x)
x_test_seq = token.texts_to_sequences(test_x)
train_fit = sequence.pad_sequences(x_train_seq, maxlen=400) 
test_fit = sequence.pad_sequences(x_test_seq, maxlen=400)

#2)build model
#RNN
modelRNN = Sequential()
modelRNN.add(Embedding(output_dim=32,input_dim=4000,
                       input_length=400))
modelRNN.add(Dropout(0.2))
modelRNN.add(SimpleRNN(units=16)) #16個神經元的RNN層
modelRNN.add(Dense(units=128,activation='relu')) #128個神經元的隱藏層
modelRNN.add(Dropout(0.3))
modelRNN.add(layers.Dense(1, activation='sigmoid')) #1個神經元的輸出層
modelRNN.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
RNN_history = modelRNN.fit(train_fit,train_y, epochs=20,
                             batch_size=1000, verbose=2, validation_split=0.2)

#LSTM
modelLSTM = Sequential()
modelLSTM.add(Embedding(output_dim=32,input_dim=4000,
                       input_length=400))
modelLSTM.add(Dropout(0.2))
modelLSTM.add(LSTM(16)) #16個神經元的LSTM層
modelLSTM.add(Dense(units=128,activation='relu')) #128個神經元的隱藏層
modelLSTM.add(Dropout(0.3))
modelLSTM.add(layers.Dense(1, activation='sigmoid')) #1個神經元的輸出層
modelLSTM.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
LSTM_history = modelLSTM.fit(train_fit,train_y, epochs=20,
                             batch_size=1000, verbose=2, validation_split=0.2)
#plot
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1,len(acc)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(x,acc,label='training')
    plt.plot(x,val_acc,label='validation')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(x,loss,label='training')
    plt.plot(x,val_loss,label='validation')
    plt.title('Loss')
    plt.legend()
plot_history(RNN_history)
plt.show()
plot_history(LSTM_history)
plt.show() 

#3)evaluate model
scores = modelRNN.evaluate(test_fit,test_y,verbose=1) 
scores[1]
scores = modelLSTM.evaluate(test_fit,test_y,verbose=1) 
scores[1]