import plivo

# Your Plivo authentication credentials
auth_id = 'MANTA2NMFJYZYWNZJHN2'
auth_token = 'MjFmMDc4NjczNjlhOGI3ZDUxMzcwMGQ4ODU4MWM3'

# Create a Plivo client
client = plivo.RestClient(auth_id=auth_id, auth_token=auth_token)

# Send an SMS message
response = client.messages.create(
    src='+91 9588661022',   # Your Plivo phone number
    dst='+91 9511844678',   # Recipient's phone number
    text='Hello, World!'
)

print(response)