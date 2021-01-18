import numpy as np
from sklearn import pipeline, metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM,Dense,Flatten,Dropout,Embedding,Bidirectional,concatenate,Dense,Input,LSTM,SpatialDropout1D,Bidirectional,Activation,Conv1D,GRU,GlobalAveragePooling1D,GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
df=pd.read_csv('https://github.com/srivatsan88/YouTubeLI/blob/master/dataset/consumer_compliants.zip?raw=true', compression='zip', sep=',', quotechar='"')

target = df['Product'].unique()
print(target)

target = pd.get_dummies(df["Product"])
print(target)

Y = target

X = df["Consumer complaint narrative"]

import nltk
import re

nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import RegexpStemmer, PorterStemmer
## We can use stemmer or lemmitization
ps = PorterStemmer()
corpus = []
for i in tqdm(range(0, len(X))):
    rev = re.sub('[^a-zA-z]', ' ', X[i])
    rev = rev.lower()
    rev = rev.split()

    rev = [ps.stem(words) for words in rev if not words in stopwords.words('english')]
    rev = ' '.join(rev)
    corpus.append(rev)


tokenizer = Tokenizer(lower=True)
max_len = 600
embed_size = 300
tokenizer.fit_on_texts(corpus)
X = tokenizer.texts_to_sequences(texts=corpus)
X = pad_sequences(X,maxlen=max_len)
vocab_size = len(tokenizer.word_index)+1

model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_len))
model.add(Bidirectional(LSTM(300,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)))
model.add(Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform"))
model.add( GlobalMaxPooling1D())
model.add(Dense(1028,activation = 'relu'))
model.add(Dense(6,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])

model.summary()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.21,random_state = 21,stratify = Y.values)

filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
callback = [checkpoint,early]

## Start Training

model.fit(X_train, Y_train, batch_size=256,
          epochs=5, validation_data=(X_test, Y_test),callbacks = callback,verbose=1)

