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
    lambda name, interviewer_name: f"Hi {name}, welcome! My name is {interviewer_name} and I'll be conducting your interview today. Thanks for taking the time to chat with me. I've had a chance to review your resume, and I'm excited to learn more about you.\n\nTo get us started, can you tell me a bit about yourself and what you're looking for in your next role?",
    lambda name, interviewer_name: f"Hello {name}! I'm {interviewer_name}, and I'm really glad we can have this conversation today. I've gone through your background, and there are several things that caught my attention.\n\nWhy don't we start with you giving me a quick overview of your background and experience?",
    lambda name, interviewer_name: f"Hi {name}, it's great to meet you! I'm {interviewer_name}, and I'll be talking with you today. I took a look at your resume, and I'm looking forward to our discussion.\n\nLet's start off with you telling me a little bit about yourself - what drives you in your career and what you're hoping to achieve?",
    lambda name, interviewer_name: f"Hey {name}! Thanks for being here today. I'm {interviewer_name}, and I'm excited to get to know you better. Your resume shows some interesting experience.\n\nTo kick things off, can you introduce yourself and share what you're most proud of in your career so far?",
    lambda name, interviewer_name: f"Hello {name}! I'm {interviewer_name}, and it's a pleasure to meet you. I've reviewed your application, and I'm curious to learn more about your journey.\n\nLet's dive in - can you tell me about yourself and what brings you here today?",
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
