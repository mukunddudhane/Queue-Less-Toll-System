# python script for sending message update

import time
from time import sleep
from sinchsms import SinchSMS

# function for sending SMS
def sendSMS():

	# enter all the details
	# get app_key and app_secret by registering
	# a app on sinchSMS
	number = '+919588661022'
	app_key = '0d0aec2b49a24341afb99e5e8d83b9dc'
	app_secret = '626f967989774925a23881a2d55f2324'

	# enter the message to be sent
	message = 'Hello Message!!!'

	client = SinchSMS(app_key, app_secret)
	print("Sending '%s' to %s" % (message, number))

	response = client.send_message(number, message)
	message_id = response['messageId']
	response = client.check_status(message_id)

	# keep trying unless the status returned is Successful
	while response['status'] != 'Successful':
		print(response['status'])
		time.sleep(1)
		response = client.check_status(message_id)

	print(response['status'])

if __name__ == "__main__":
	sendSMS()
