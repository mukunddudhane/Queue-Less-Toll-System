import mysql.connector as mysql
import json
db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "toll@2024",
    database = "toll"
)
np="MH 12 QY 7778"
#print("np type",type(np))
cursor = db.cursor()
cursor.execute("SELECT License_Plate FROM toll_front")

myresult = cursor.fetchall() ##fetches first row of the record
data={'data':myresult}
json_data = json.dumps(data)
loaded = json.loads(json_data)
l=len(loaded)
print("lemgth ",l)
for i in loaded:
	print(type(loaded[i]))


