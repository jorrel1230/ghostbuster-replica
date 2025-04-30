import os
import json
import requests

# Replace with your actual API key
API_KEY = 'your-api-key-here'
API_HOST = 'api.zerogpt.com'
API_URL = f'https://{API_HOST}/api/detect/detectText'

HEADERS = {
    'Content-Type': 'application/json',
    'ApiKey': API_KEY,
}

def analyze_text(text):
    payload = {'input_text': text}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def process_txt_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"\nProcessing file: {filename}")
                result = analyze_text(content)
                if result and result.get('success'):
                    data = result.get('data', {})
                    # unclear what data returns
                    # print(f"  - Words Count: {data.get('words_count')}")
                    # print(f"  - GPT Generated: {data.get('is_gpt_generated')}%")
                    # print(f"  - Human Written: {data.get('is_human_written')}%")
                    # print(f"  - Feedback: {data.get('feedback_message')}")
                else:
                    print("  - Failed to retrieve analysis.")

# Specify the directory containing your .txt files
TEXT_FILES_DIRECTORY = 'data/selected/gpt_hs'

process_txt_files(TEXT_FILES_DIRECTORY)
