import nltk 
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.python.ops.gen_array_ops import expand_dims_eager_fallback

stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import json
import tensorflow as tf
import pickle
import discord
import os
from dotenv import load_dotenv

load_dotenv()

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]


    for x,doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]

        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)


tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn") 
    
  
def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:
        for i,w in enumerate(words):
            if w == s_word:
                bag[i] = 1

    return numpy.array(bag)

def chat(message):
    result = model.predict([bag_of_words(message,words)])[0]
    result_index = numpy.argmax(result)
    if result[result_index] > 0.80:
        tag = labels[result_index]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                responses = intent["responses"]
            
        return random.choice(responses)
    else:
        return "Didn't get that :("


client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith('$roe'):
        text = message.content.lstrip('$roe')
        await message.channel.send(chat(text))
    
client.run(os.getenv('TOKEN'))


