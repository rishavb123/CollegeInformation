import requests
from api_key import api_key

root = 'https://api.data.gov/ed/collegescorecard/v1/schools'
college_id = 170976 # for University of Michigan
url = root + '?id=' + str(college_id) + '&api_key=' + api_key

r = requests.get(url)

with open('response.json', 'w') as file:
    file.write(r.text)
    file.close()