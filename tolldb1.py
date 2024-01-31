import mysql.connector
mydb = mysql.connector.connect(host="localhost",user="root",password="toll@2023")
mycursor=mydb.cursor()
#mycursor.execute("Create database toll ")
mycursor.execute("Show databases")

for db in mycursor:
	print(db)

	