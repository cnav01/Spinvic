import json
from typing import Dict, Any

def identify_llm_phase(video_path: str, prompt: str, bowler_hand: str) -> Dict[str, Any]:
    """
    This function takes a video path and a prompt, calls the Gemini Pro Vision API,
    and returns the parsed JSON response.

    Args:   a
        video_path (str): The path to the video file.
        prompt (str): The text prompt to send to the API.

    Returns:
        Dict[str, Any]: The JSON response from the API as a dictionary.
    """
    phase_detection_prompt = f"""
    You are an expert cricket biomechanics analyst specializing in high-speed video analysis.
    You will be provided with a video of a single cricket bowling action by a {bowler_hand} bowler.
    Your task is to meticulously analyze the video and identify the precise frame index for the
    following four critical biomechanical events.

    The video frames start from index 0.

    ### Critical Events to Identify: 

    1.  **Back Foot Contact (BFC):** The first frame where the bowler's back foot makes solid contact
        with the ground after their jump or gather. This marks the beginning of the delivery stride.
    2.  **Front Foot Contact (FFC):** The frame where the bowler's front (bracing) foot lands firmly
        on the ground, providing a stable base for rotation.
    3.  **Ball Release (BR):** The single, exact frame where the ball is visibly leaving the bowler's
        fingertips. This typically occurs at or near the highest point of the bowling arm's arc.
    4.  **Follow-Through Completion:** The frame where the primary bowling arm motion clearly concludes,
        usually as the bowling arm passes the opposite side of the body and the bowler begins to regain their balance.

    ### Output Instructions:

    Return your findings ONLY as a single, valid JSON object. Do not include any other conversational
    text, explanations, or markdown outside of the JSON structure.

    If a specific event is not clearly visible or cannot be determined from the provided frames,
    use the value `null` for that frame number.

    **JSON Output Format:**
    ```json
    {{
      "bfc_frame": <integer or null>,
      "ffc_frame": <integer or null>,
      "ball_release_frame": <integer or null>,
      "follow_through_completion_frame": <integer or null>,
      "analysis_notes": "<A brief, one-sentence summary of your confidence in the detection, or any notable observations.>"
    }}
    ```
    """

    print(f"[{bowler_hand} Bowler] Initiating LLM phase detection for video: {video_path}")

    llm_output = call_gemini_pro_vision_api(phase_detection_prompt, video_path)

    if llm_output:
        print(f"LLM successfully identified phases. Raw output snippet: {json.dumps(llm_output, indent=2)[:200]}...") # Print a snippet for brevity
        
        required_keys = ["bfc_frame", "ffc_frame", "ball_release_frame", "follow_through_completion_frame"]
        if all(key in llm_output for key in required_keys):
            return llm_output
        else:
            print("Error: LLM output is missing one or more required keys.")
            return {}
    else:
        print("Error: No output received from LLM.")
        return {}
