import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="toll@2023"
)

print(mydb)
if(mydb):
	print ("Connection succesfull")

else:
	print("Unsuccesfull")