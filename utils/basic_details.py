import random
from utils.llm_call import get_response_from_llm, parse_json_response
from utils.prompts import basic_details


def extract_resume_info_using_llm(resume_content):
    # Use LLM to extract resume info
    try:
        final_prompt = basic_details.format(resume_content=resume_content)
        response = get_response_from_llm(final_prompt)
        response = parse_json_response(response)
        
        if response is None:
            print("Warning: LLM returned None, using fallback resume info")
            return "Candidate", "Experienced professional with strong skills and background."
        
        name = response.get("name", "Candidate")
        resume_highlights = response.get("resume_highlights", "Experienced professional with strong skills and background.")
        return name, resume_highlights
        
    except Exception as e:
        print(f"Error extracting resume info: {str(e)}")
        print("Using fallback resume information")
        return "Candidate", "Experienced professional with strong skills and background."


ai_greeting_messages = [
    lambda name, interviewer_name: f"Hi {name}, welcome to this AI interview! My name is {interviewer_name} and I'll be your interviewer today. Let's get started!\n\nCan you tell me a bit about yourself and what you're looking for in a job?",
    lambda name, interviewer_name: f"Hi {name}, welcome to this AI interview! My name is {interviewer_name} and I'll be your interviewer today. Let's get started!\n\nCan you give me a quick overview of your background and experience?",
    lambda name, interviewer_name: f"Hi {name}, welcome to this AI interview! My name is {interviewer_name} and I'll be your interviewer today. Let's get started!\n\nCan you tell me a little bit about your goals and aspirations?",
    lambda name, interviewer_name: f"Hi {name}, welcome to this AI interview! My name is {interviewer_name} and I'll be your interviewer today. Let's get started!\n\nCan you briefly introduce yourself and tell me about your achievements and skills?",
]


final_thanks_for_taking_interview_msgs = [
    lambda name: f"Thanks for taking the time to chat today, {name}. I really enjoyed our conversation. Wishing you all the best in your career!",
    lambda name: f"It was great speaking with you, {name}. I hope the interview was a valuable experience for you. Good luck moving forward!",
    lambda name: f"Appreciate your time today, {name}. Best of luck with the rest of your job applications and interviews!",
    lambda name: f"Thank you for the engaging conversation, {name}. I wish you success in your job hunt and future endeavors!",
    lambda name: f"It was a pleasure talking to you, {name}. I hope the interview helped clarify your goals. All the best!",
    lambda name: f"Thanks again for your time, {name}. I hope you found the interview insightful. Good luck on your journey ahead!",
]


def get_ai_greeting_message(name, interviewer_name="Alex"):
    return random.choice(ai_greeting_messages)(name, interviewer_name)


def get_final_thanks_message(name):
    return random.choice(final_thanks_for_taking_interview_msgs)(name)
