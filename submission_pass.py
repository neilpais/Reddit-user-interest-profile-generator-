import json
import re
import datetime

doc_text=""
sources=[r'C:\ML project CIS\submission\RS_2012-01']



for i in range(0,1):
  input_file = open(sources[i],'r',encoding="utf8")
  with input_file as f:
   for line in f: 
     data = json.loads(line)
     try:
      try:
       try:
         doc_text=doc_text + data["title"] +" "+ data["selftext"]+" "
         doc_text = doc_text.replace("\n"," ")
         datetime_obj = datetime.datetime.fromtimestamp(data["created_utc"]) 
         output= open(r"C:\ML project CIS\Finaloutput\%s.txt" %data['id'],"w+", encoding="utf8")
         output.write(data["id"]+"\n0\n"+str(datetime_obj.date())+"\n"+str(datetime_obj.date())+"\n"+data["author"]+"\n"+ doc_text +"\n" )
         doc_text=""
       except ValueError as error:
          continue		  
      except IndexError as error:
         continue
     except KeyError as error:
         continue

