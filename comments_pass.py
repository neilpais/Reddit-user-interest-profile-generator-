import json
import re
import datetime
import os


j=0
sources=[r'C:\ML project CIS\submission\RS_2012-01']


for i in range(0,1):
  input_file = open(sources[i],'r',encoding="utf8")
  with input_file as f:
    for line in f:
     data = json.loads(line)

     try:
      try:
       try:
         id=data["link_id"].replace("t3_","")
         datetime_obj = datetime.datetime.fromtimestamp(int(data["created_utc"]))
         if os.path.isfile(r"C:\ML project CIS\Finaloutput" .format(id)):
              print ("yes ", id) 
               			  
              sub_file = open(r"C:\ML project CIS\Finaloutput" %id,'r',encoding="utf8") 
              file_lines=sub_file.readlines()
              comment_num=int(file_lines[1].strip())+1
              start_date=file_lines[2].strip()
              end_datetime_obj= datetime.datetime.strptime(file_lines[3].strip(), '%Y-%m-%d')
              if datetime_obj>end_datetime_obj:
                  end_datetime_obj=datetime_obj
              text=file_lines[5].strip()+" [comment_break] "+ data["body"]+" "
              text = text.replace("\n"," ")
              authors=file_lines[4].strip()+","+data["author"]
              sub_file.close()
              output= open(r"C:\ML project CIS\Finaloutput\   %s.txt" %id,"w+", encoding="utf8")
              output.write(id+"\n"+str(comment_num)+"\n"+start_date+"\n"+str(end_datetime_obj.date())+"\n"+authors+"\n"+ text +"\n" )
              output.close()
              text=""
              authors=""
              comment_num=0
              j=j+1
              print (j)			  
       except ValueError as error:
          continue		  
      except IndexError as error:
         continue
     except KeyError as error:
         continue





