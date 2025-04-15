import requests
import os

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json()
    for model in models['data']:
        print(model['id'])
else:
    print(f"Error: {response.status_code} - {response.text}")
