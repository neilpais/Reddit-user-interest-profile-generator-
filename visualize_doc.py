import matplotlib
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.neighbors import KNeighborsClassifier
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import *
#from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics
import os
from scipy.spatial import distance
import math
import json
import re
from tqdm import tqdm

sources=[r"C:\ML project CIS\submission\RS_2012-01"]
plt.style.use('dark_background')

stemmer = PorterStemmer()

PAD = "<pad>"
UNK = "<unk>"
nltk_tokeniser = nltk.tokenize.TweetTokenizer()

EmbeddingSize = 50


def WordEmbeddingLoader(fp, embedding_size):
    embedding = []
    vocab = []
    linenumber = 0
    with open(fp, 'r', encoding='UTF-8') as f:
        for each_line in f:

            linenumber += 1
            row = each_line.split(' ')

            if len(row) == 2:
                continue
            vocab.append(row[0])
            #if len(row[1:]) != embedding_size:
                #print(row[0])
                #print(len(row[1:]))
            embedding.append(np.asarray(row[1:], dtype='float32'))

    word2id = dict(zip(vocab, range(2, len(vocab) + 2)))
    word2id[PAD] = 0
    word2id[UNK] = 1

    extra_embedding = [np.zeros(embedding_size), np.random.uniform(-0.1, 0.1, embedding_size)]
    embedding = np.append(extra_embedding, embedding, 0)


    return word2id, embedding, vocab
fp = 'glove.6B\glove.6B.'+str(EmbeddingSize)+'d.txt'
word2id, embeddingmatrix, allwords = WordEmbeddingLoader(fp,EmbeddingSize)




def review_to_words(raw_review):
    letters_only = raw_review
    #letters_only = re.sub("[^a-zA-Z]", " ", raw_review)

    words = letters_only.lower()

    words = nltk.word_tokenize(words)

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return( " ".join( meaningful_words ))


text2=""
processtxt = []
processvector = []
processvector2 = []
uservector = []
user_names = []

docvec=open("docs_dem300.txt",'a+')
#for i in range(0,1):
input_file = open("C:\ML project CIS\submission\RS_2012-01",'r',encoding="utf8")
with input_file as f:
  for line in tqdm(f): 
   data = json.loads(line)
   #print(data['id'])
   try:
    try:
     try:  
      if os.path.isfile(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id']):
       input= open(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id']  ,"r", encoding="utf8")
       file_lines=input.readlines()
       words_process=re.findall(r'"(.*?)"', str(file_lines[6].strip()))
       #print(words_process)
       for word in words_process:
          text2=text2+word+" "
       #print (text2) 
       text2 = text2.lower()
       #print (text2) 
       words = nltk.word_tokenize(text2)
       for w in words:
        #print (w)
        if w in word2id:
            processtxt.append(w)
            idx = word2id[w]
            vector = embeddingmatrix[idx,:]
            processvector.append(vector )
            processvector2.append(vector )
       #print('embeddingmatrix', embeddingmatrix)
       mean = embeddingmatrix.mean()
       uservector.append(list(processvector2))
       user_names.append(data['id'])
       docvec.write(str(data['id'])+" "+str(np.round(np.mean(processvector2,axis=0),5))[1:-1].replace("\n","").replace("  "," ").replace("  "," ")+"\n")
       #print ("hello___________________________________",uservector)
       #print ("hello___________________________________",processvector)
       #print ("hello___________________________________",processvector2)
       processvector2.clear()
       #print ("hello___________________________________",uservector)
       text2=""
     except ValueError as error:
          continue		  
    except IndexError as error:
         continue
   except KeyError as error:
         continue



#print('processtxt=')
#print(processtxt)
#print('processvector=', processvector)
#len_vec = len(processvector)
#print(len_vec)
#print(processvector)
#ave_vector = np.average(processvector, axis=0)
#print('.........................', ave_vector)


#print(embeddingmatrix[word2id['analyzing'],:])





#processed_wmn = [ review_to_words(text) for text in reddit]
#processed_wmn1 = [w for w in processed_wmn if w in word2id]


#print(processed_wmn)
#print('after word2id...')
#print(processed_wmn1)
def tsne_plot3(words, vwords):
    "Creates a TSNE model and plots it"
    newlist =[]
    for v in vwords:
       if v.shape == (50,):
          newlist.append(v)
    newvwords = np.array(newlist)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(np.array(newvwords))
    #print('tsne_model.fit_transform')
    #print('value of the pca', new_values)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i],c="lightblue",alpha=0.5)
    plt.axis('off')
    plt.show()
	

def tsne_plot1(words, vwords):
    "Creates a TSNE model and plots it"
    newlist =[]
    for v in vwords:
       if v.shape == (50,):
          newlist.append(v)
    newvwords = np.array(newlist)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(newvwords)
    #print('tsne_model.fit_transform')
    #print('value of the pca', new_values)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i],alpha=0.5)
        plt.annotate(words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.axis('off')
    plt.show()
	
def tsne_plot2(vwords, uservec):
    "Creates a TSNE model and plots it"
    newlist =[]
    for v in vwords:
       if v.shape == (50,):
          newlist.append(v)
    newvwords = np.array(newlist)
    newuservec =[]
    for v in uservec:
       if v.shape == (50,):
          newuservec.append(v)
    newuse = np.array(newuservec)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(newvwords)
    user_value = tsne_model.fit_transform(newuse)
    #print('tsne_model.fit_transform')
    #print('value of the pca', new_values)

    x = []
    y = []
    x1 = []
    y1 = []
    All= []
    a=0
    fig=plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111,aspect='equal') 
    for value1 in user_value:
        x1.append(value1[0])
        y1.append(value1[1])

	
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    
    for i in range(len(x)):
        distance = math.sqrt( ((x[i]-x1[0])**2)+((y[i]-y1[0])**2) )
        All.append(distance)
        if distance < 9:
           a=0.0
        elif distance < 18:
           a=0.1
        elif distance < 27:
           a=0.2
        elif distance < 36:
           a=0.3
        elif distance < 45:
           a=0.4
        elif distance < 54:
           a=0.5
        elif distance < 63:
           a=0.6
        elif distance < 72:
           a=0.7
        elif distance < 81:
           a=0.8
        elif distance < 90:
           a=0.9
        elif distance < 99:
           a=1.0
        plt.scatter(x[i], y[i],c=(0.75, a, a),alpha=0.5)
    print("value-----------------",min(All))
    print("value-----------------",max(All))
    circle1 = plt.Circle((x1[0], y1[0]),9, color='white',fill=False)
    circle2 = plt.Circle((x1[0], y1[0]), 18, color='white',fill=False)
    circle3 = plt.Circle((x1[0], y1[0]), 27, color='white',fill=False)
    circle4 = plt.Circle((x1[0], y1[0]), 36, color='white',fill=False)
    circle5 = plt.Circle((x1[0], y1[0]), 45, color='white',fill=False)
    circle6 = plt.Circle((x1[0], y1[0]), 54, color='white', fill=False)
    circle7 = plt.Circle((x1[0], y1[0]), 63, color='white',fill=False)
    circle8 = plt.Circle((x1[0], y1[0]), 72, color='white',fill=False)
    circle9 = plt.Circle((x1[0], y1[0]), 81, color='white',fill=False)
    

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    ax.add_artist(circle5)
    ax.add_artist(circle6)
    ax.add_artist(circle7)
    ax.add_artist(circle8)
    ax.add_artist(circle9)

    plt.scatter(x1[0], y1[0],c="white")
    plt.axis('off')
    plt.show()

#tsne_plot1(processtxt,processvector)


#X = processvector
temp_av=[]

#print(processvector)
#plt.figure()
#plt.subplot(111)
firstuser=list(uservector[15])
for vector in uservector:
  temp_av.append(np.mean(vector,axis=0))

  #plt.scatter(temp_av)
  #plt.plot(,temp_av)
tsne_plot1(user_names, temp_av )
tsne_plot3(user_names, temp_av )

tsne_plot2(temp_av, firstuser)
#plt.show()

X=temp_av
NUM_CLUSTERS=40
y=[]
for s in X:
   if s.shape == (50,):
      y.append(s)
C = np.array(y)
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
#print(X)
Cluster2dimension= tsne_model.fit_transform(C)


kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(Cluster2dimension)
y_kmeans = kmeans.predict(Cluster2dimension)
assigned_clusters = kmeans.labels_



plt.scatter(Cluster2dimension[:, 0], Cluster2dimension[:, 1], s=50, c=assigned_clusters,alpha=0.5)
plt.axis('off')
plt.show()

X_embedded = TSNE(n_components=1).fit_transform(C)
neighs_tsne = KNeighborsClassifier(n_neighbors=100)
neighs_tsne.fit(X_embedded, firstuser)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=50,alpha=0.5)
plt.axis('off')
plt.show()
# plt.scatter(X[:, 0], X[:, 1], c="g", s=50, cmap='viridis')
#plt.show()
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50);