import os
import google.generativeai as genai
import json
import re # to clean json string
from typing import Dict, Any

def call_gemini_pro_vision_api(prompt: str, video_path: str) -> Dict[str, Any]:
    """
    Calls the Gemini Pro Vision API with the given prompt and video file.
    
    Args:
        prompt (str): The text prompt to send to the API.
        video_path (str): The path to the video file to be analyzed.
        
    Returns:
        Dict[str, Any]: The JSON response from the API as a dictionary.
    """
    # Ensure the API key is set in your environment
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        print("Error: GENAI_API_KEY environment variable not set.")
        return {}
    genai.configure(api_key=api_key)

    #initialize gemini pro vision model
    model = genai.GenerativeModel('gemini-pro-vision')

    try:
        video_file = genai.upload_file(path=video_path)
        print(f"Video file uploaded successfully: {video_file}")

        response = model.generate_content([prompt, video_file], stream=False)

        if not response.candidates:
            print("No candidates found in the response.")
            return {}
        
        response_text = response.text
        print(f"Raw response text: {response_text}")

        json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'{\s*".*?":.*?}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print("No JSON object found in the response.")
                return {}
            
        parsed_response = json.loads(json_str)
        return parsed_response
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    finally:
        if video_file:
            try:
                genai.delete_file(video_file.name)
                print(f"Temporary video file deleted: {video_file.name}")
            except Exception as e:
                print(f"Could not delete temporary video file: {e}")
               

        