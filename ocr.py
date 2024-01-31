import easyocr
reader = easyocr.Reader(['en'] ,gpu=True) # this needs to run only once to load the model into memory
result = reader.readtext('Car number plate_108.jpeg')
path="save.txt"

with open(path,'w') as f:
	for (bbox,text,prob) in result:
		if(prob >= 0.3):
			print(result,file=f)