import requests
import json
import time

t1=time.time()
json_file = 'api_data.json'

print(time.time()-t1)
url = 'http://localhost:5000/detect'
with open(json_file) as f:
    data = json.load(f)
print(time.time()-t1)

server_return = requests.post(url, json=data)
print(time.time()-t1)

print(server_return)

