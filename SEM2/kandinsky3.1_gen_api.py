import json
import time
import requests
import base64
import os
from random import randint as r

class Text2ImageAPI:

    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        if response.status_code == 200 and data:
            return data[0]['id']
        else:
            print(f"Error fetching model: {data}")
            return None

    def generate(self, prompt, model, images=1, width=1024, height=1024):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        
        return data['uuid']

    def check_generation(self, request_id, attempts=20, delay=15):
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if response.status_code == 200:
                if data['status'] == 'DONE':
                    return data['images']
                elif data['status'] == 'ERROR':
                    print(f"Error in generation process: {data}")
                    return None
                else:
                    print(f"Current status: {data['status']}, attempts left: {attempts}")
            else:
                print(f"Error checking status: {data}")

            attempts -= 1
            time.sleep(delay)
        return None


if __name__ == '__main__':
    dirr = "./kandinsky3.1/"
    
    
    # Создание списка запросов
    
    prompts = []
    
    for prompt in prompts:
        api = Text2ImageAPI('https://api-key.fusionbrain.ai/', '7292D8211666BA5834F53C40CAF4E038', '360ED7997871D869E1E17FDE1436C564')

        model_id = api.get_model()
        if model_id is None:
            print("Failed to get model.")
        else:
            uuid = api.generate(prompt, model_id)
            if uuid is None:
                print("Failed to generate image.")
            else:
                images = api.check_generation(uuid)
                if images is None:
                    print("Failed to retrieve images.")
                else:
                    image_base64 = images[0]
                    image_data = base64.b64decode(image_base64)
                    file_name = f"{dirr}/{prompt.replace(' ', '_')}_{r(0, 100000)}.jpg"
                    with open(file_name, "wb") as file:
                        file.write(image_data)
                    print(f"Image saved as {file_name}")
