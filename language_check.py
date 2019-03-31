
import requests


def check(input_text, lang='en', api_url='https://languagetool.org/api/v2/'):
    data = {
        'text': input_text,
        'language': lang,
        }
    response = requests.post(api_url + 'check', data=data)
    if response.status_code != 200:
        raise ValueError(response.text)
    return response.json()['matches']
