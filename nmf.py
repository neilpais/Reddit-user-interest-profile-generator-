import json
import re
import datetime
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import os
import nltk
nltk.download('wordnet')

doc_text=""
sources=[r'C:\ML project CIS\submission\RS_2012-01']

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

for i in range(0,1):
  input_file = open(sources[i],'r',encoding="utf8")
  with input_file as f:
   for line in f: 
      data = json.loads(line)
      try:
       try:
        try:
         print(data['id'])
         if os.path.isfile(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id']): 
          input2= open(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id'],"r", encoding="utf8")
          file_lines=input2.readlines()
          text=file_lines[5].strip()
          input2.close()
          doc_complete = text.split('[comment_break]')
          doc_clean = [clean(doc).split() for doc in doc_complete] 
          dictionary = corpora.Dictionary(doc_clean)
          doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean] 
          nmf = gensim.models.Nmf # TODO: EnsembleLDA
          nmfmodel = nmf(doc_term_matrix, num_topics=1, id2word = dictionary, passes=100)
          print(nmfmodel.print_topics(num_topics=1, num_words=50))
          output= open(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id'],"w+", encoding="utf8")
          print(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id'])
          output.write(file_lines[0].strip()+"\n"+file_lines[1].strip()+"\n"+file_lines[2].strip()+"\n"+file_lines[3].strip()+"\n"+file_lines[4].strip()+"\n"+ file_lines[5].strip() +"\n" +str(nmfmodel.print_topics(num_topics=1, num_words=100) )+"\n")
          output.close() 
        except ValueError as error:
          continue		  
       except IndexError as error:
         continue
      except KeyError as error:
         continue





