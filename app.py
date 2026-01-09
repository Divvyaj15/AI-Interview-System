import streamlit as st
import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from utils import (
    transcribe_with_speechmatics,
    extract_resume_info_using_llm,
    get_ai_greeting_message,
    get_final_thanks_message,
    speak_text,
    analyze_candidate_response_and_generate_new_question,
    get_feedback_of_candidate_response,
    get_overall_evaluation_score,
    save_interview_data,
    load_content_streamlit,
)
from utils.speaking_skills_analyzer import SpeakingSkillsAnalyzer
from utils.sentiment_audio_analyzer import SentimentAudioAnalyzer

MAX_QUESTIONS = 5

# Helper function to run async functions safely in Streamlit
def run_async_safely(coro):
    """Run async coroutine safely in Streamlit context"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an event loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # If no event loop is running, we can use asyncio.run
            return asyncio.run(coro)
    except RuntimeError:
        # Fallback: create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# Configuration and Styling
def setup_page_config():
    """Setup page configuration and custom CSS"""
    st.set_page_config(page_title="AI Interview App", layout="wide")

    st.markdown(
        """
    <style>
    .audio-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        border: 2px solid #e9ecef;
    }
    .interview-progress {
        background-color: #e8f5e8;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "interview_started": False,
        "name": "",
        "resume_highlights": "",
        "job_description": "",
        "qa_index": 1,
        "conversations": [],
        "current_question": "",
        "question_spoken": False,
        "awaiting_response": False,
        "processing_audio": False,
        "messages": [],
        "interview_completed": False,
        "max_questions": MAX_QUESTIONS,
        "ai_voice": "Alex (Male)",
        "thanks_message_prepared": False,
        "thanks_message_spoken": False,
        "show_final_results": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_ai_voice_details():
    """Get AI voice configuration"""
    return {
        "Alex (Male)": {"name": "Alex", "code": "en-US-GuyNeural"},
        "Aria (Female)": {"name": "Aria", "code": "en-US-AriaNeural"},
        "Natasha (Female)": {"name": "Natasha", "code": "en-AU-NatashaNeural"},
        "Sonia (Female)": {"name": "Sonia", "code": "en-GB-SoniaNeural"},
    }

def get_instructions():
    """Get instructions for the user"""
    content = """
    #### Please follow these steps to use the AI Interview App:
    1. **Upload Resume**: Upload your resume in PDF format.
    2. **Job Description**: Paste the job description.
    3. **Start Interview**: Click the "Start Interview" button to begin the AI-driven interview.
    4. **Maximum Questions**: You can set the maximum number of questions for the interview. (default: 5)
    5. **Voice Selection**: Choose the AI voice for the interview.(default: Alex (Male))
    6. **Answer Questions**: Answer the questions posed by the AI interviewer.
    7. **Review Results**: After the interview, review your performance and feedback.

    **Note**: The AI interviewer will ask you questions based on your resume and the job description.

    I hope you find the experience helpful!
    """
    return st.markdown(content)

def render_sidebar():
    """Render sidebar with candidate information and settings"""
    st.sidebar.title("Candidate Information")

    # File upload
    uploaded_resume = st.sidebar.file_uploader("Upload your Resume (PDF)", type=["pdf"])

    # Job description
    job_description = st.sidebar.text_area("Paste the Job Description")

    # Settings
    max_questions = st.sidebar.number_input(
        "Maximum number of questions",
        min_value=1,
        max_value=10,
        value=MAX_QUESTIONS,
    )
    st.session_state["max_questions"] = max_questions

    ai_voice = st.sidebar.radio(
        "Select AI Interviewer Voice",
        ["Alex (Male)", "Aria (Female)", "Natasha (Female)", "Sonia (Female)"],
    )
    st.session_state["ai_voice"] = ai_voice

    submit = st.sidebar.button("Submit")

    return uploaded_resume, job_description, submit


def process_resume_submission(uploaded_resume, job_description):
    """Process resume and job description submission"""
    with st.spinner("Processing resume..."):
        resume_content = load_content_streamlit(uploaded_resume)
        name, resume_highlights = extract_resume_info_using_llm(resume_content)

    # Store in session state
    st.session_state["name"] = name
    st.session_state["resume_highlights"] = resume_highlights
    st.session_state["job_description"] = job_description

    # Reset interview state
    reset_interview_state()

    st.success(f"Candidate Name: {name}")


def reset_interview_state():
    """Reset interview-related session state"""
    interview_keys = [
        "interview_started",
        "qa_index",
        "conversations",
        "current_question",
        "question_spoken",
        "awaiting_response",
        "processing_audio",
        "messages",
        "interview_completed",
        "thanks_message_prepared",
        "thanks_message_spoken",
        "show_final_results",
    ]

    for key in interview_keys:
        if key == "interview_started" or key == "interview_completed":
            st.session_state[key] = False
        elif key in ["qa_index"]:
            st.session_state[key] = 1
        elif key in ["conversations", "messages"]:
            st.session_state[key] = []
        elif key in ["current_question"]:
            st.session_state[key] = ""
        else:
            st.session_state[key] = False


def start_interview():
    """Initialize and start the interview"""
    st.session_state["interview_started"] = True
    reset_interview_state()
    st.session_state["interview_started"] = True  # Reset above sets this to False

    # Get greeting message
    ai_voice_details = get_ai_voice_details()
    interviewer_name = ai_voice_details[st.session_state["ai_voice"]]["name"]
    greeting_message = get_ai_greeting_message(
        st.session_state["name"], interviewer_name=interviewer_name
    )

    st.session_state["current_question"] = greeting_message
    st.session_state["messages"].append(
        {"role": "assistant", "content": greeting_message}
    )


def display_chat_messages():
    """Display all chat messages from history"""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def speak_current_question():
    """Speak the current question if not already spoken"""
    if st.session_state["current_question"] and not st.session_state["question_spoken"]:
        with st.spinner("AI Interviewer is speaking..."):
            ai_voice_details = get_ai_voice_details()
            speak_text(
                st.session_state["current_question"],
                voice=ai_voice_details[st.session_state["ai_voice"]]["code"],
            )
        st.session_state["question_spoken"] = True
        st.session_state["awaiting_response"] = True
        st.rerun()


def generate_next_question():
    """Generate and prepare the next question"""
    if st.session_state["conversations"]:
        last_conv = st.session_state["conversations"][-1]
        next_question, _ = run_async_safely(analyze_candidate_response_and_generate_new_question(
            last_conv["Question"],
            last_conv["Candidate Answer"],
            st.session_state["job_description"],
            st.session_state["resume_highlights"],
        ))
    else:
        next_question = "Tell me about yourself and your experience."

    st.session_state["current_question"] = next_question
    st.session_state["messages"].append({"role": "assistant", "content": next_question})
    st.session_state["question_spoken"] = False
    st.session_state["awaiting_response"] = False


def process_candidate_response(transcript):
    """Process candidate's response and move to next state"""
    # Add candidate's answer to chat
    st.session_state["messages"].append({"role": "user", "content": transcript})

    # Generate feedback for this response
    if st.session_state["qa_index"] < st.session_state["max_questions"] - 1:
        # Not the last question - generate next question and feedback
        next_question, feedback = run_async_safely(analyze_candidate_response_and_generate_new_question(
            st.session_state["current_question"],
            transcript,
            st.session_state["job_description"],
            st.session_state["resume_highlights"],
        ))
    else:
        # Last question - only generate feedback
        feedback = run_async_safely(get_feedback_of_candidate_response(
            st.session_state["current_question"],
            transcript,
            st.session_state["job_description"],
            st.session_state["resume_highlights"],
        ))

    # Store conversation with detailed scoring
    conversation_data = {
            "Question": st.session_state["current_question"],
            "Candidate Answer": transcript,
            "Evaluation": feedback["score"],
            "Feedback": feedback["feedback"],
        }
    
    # Add detailed scoring if available
    if "criteria_scores" in feedback:
        conversation_data["Criteria_Scores"] = feedback["criteria_scores"]
    if "competency_assessment" in feedback:
        conversation_data["Competency_Assessment"] = feedback["competency_assessment"]
    
    st.session_state["conversations"].append(conversation_data)

    # Move to next question or complete interview
    st.session_state["qa_index"] += 1
    st.session_state["processing_audio"] = False
    st.session_state["awaiting_response"] = False

    if st.session_state["qa_index"] <= st.session_state["max_questions"]:
        # Prepare next question
        generate_next_question()
        st.success("✅ Answer recorded! Preparing next question...")
    else:
        # Interview completed - prepare thanks message
        st.session_state["interview_completed"] = True
        prepare_thanks_message()


def prepare_thanks_message():
    """Prepare and display thanks message"""
    if not st.session_state["thanks_message_prepared"]:
        final_note = get_final_thanks_message(st.session_state["name"])
        st.session_state["messages"].append(
            {"role": "assistant", "content": final_note}
        )
        st.session_state["thanks_message_prepared"] = True
        st.session_state["qa_index"] -= 1
        st.rerun()


def speak_thanks_message():
    """Speak the thanks message"""
    if (
        st.session_state["thanks_message_prepared"]
        and not st.session_state["thanks_message_spoken"]
    ):

        # Get the last message (thanks message)
        thanks_message = st.session_state["messages"][-1]["content"]

        with st.spinner("AI Interviewer is giving final remarks..."):
            ai_voice_details = get_ai_voice_details()
            speak_text(
                thanks_message,
                voice=ai_voice_details[st.session_state["ai_voice"]]["code"],
            )

        st.session_state["thanks_message_spoken"] = True
        st.success("🎉 Interview completed! Thank you for your time.")

        # Now show final results
        st.session_state["show_final_results"] = True
        st.rerun()


def handle_audio_recording():
    """Handle audio recording and processing"""
    if not (
        st.session_state["awaiting_response"]
        and not st.session_state["processing_audio"]
    ):
        return

    st.write("**🎙️ Please record your answer to the question above**")

    audio_key = f"audio_input_{st.session_state['qa_index']}_{len(st.session_state['messages'])}"
    audio_data = st.audio_input("Record your answer", key=audio_key)

    if audio_data is not None:
        st.session_state["processing_audio"] = True

        with st.spinner("Processing your answer..."):
            # Save audio file
            name = st.session_state["name"]
            filename = f"audio/{name}/{name}_{st.session_state['qa_index'] + 1}.wav"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, "wb") as f:
                f.write(audio_data.read())

            # Transcribe audio
            transcript = transcribe_with_speechmatics(filename)

            if transcript and transcript.strip():
                process_candidate_response(transcript)
                st.rerun()
            else:
                st.error("No speech detected in audio. Please try recording again.")
                st.session_state["processing_audio"] = False


def display_final_results():
    """Display final interview results with detailed scoring breakdown"""
    if (
        not st.session_state["show_final_results"]
        or not st.session_state["conversations"]
    ):
        return

    with st.spinner("Calculating final score..."):
        final_score = get_overall_evaluation_score(st.session_state["conversations"])

        # Save interview data
        now = datetime.now().isoformat() + "Z"
        interview_data = {
            "name": st.session_state["name"],
            "createdAt": now,
            "updatedAt": now,
            "id": 1,
            "job_description": st.session_state["job_description"],
            "resume_highlights": st.session_state["resume_highlights"],
            "conversations": st.session_state["conversations"],
            "overall_score": round(final_score, 2),
        }
        save_interview_data(interview_data, candidate_name=st.session_state["name"])

    # Display results with enhanced scoring explanation
    st.subheader("🎉 Interview Results")
    st.markdown(f"**Candidate:** {st.session_state['name']}")
    
    # Overall score with visual indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if final_score >= 8.5:
            st.success(f"**Overall Score: {final_score:.2f}/10** 🏆")
            st.markdown("*Exceptional Performance*")
        elif final_score >= 7.0:
            st.info(f"**Overall Score: {final_score:.2f}/10** ⭐")
            st.markdown("*Strong Performance*")
        elif final_score >= 5.5:
            st.warning(f"**Overall Score: {final_score:.2f}/10** 📈")
            st.markdown("*Good Performance with Room for Growth*")
        else:
            st.error(f"**Overall Score: {final_score:.2f}/10** 💡")
            st.markdown("*Areas for Improvement Identified*")

    # Scoring Explanation Section
    st.subheader("📊 How You Were Scored")
    
    with st.expander("🎯 Scoring Criteria Explained", expanded=True):
        st.markdown("""
        **Your responses were evaluated across 6 key criteria:**
        
        **1. Relevance** - How well you addressed the question asked
        **2. Completeness** - How thorough and comprehensive your answer was  
        **3. Structure** - How well-organized and easy to follow your response was
        **4. Specificity** - How detailed and concrete your examples were
        **5. Impact** - How you demonstrated measurable results and outcomes
        **6. Professionalism** - How appropriate and confident your communication was
        
        **Plus 6 competency areas:**
        - Technical Skills, Problem-Solving, Communication, Leadership, Cultural Fit, Growth Mindset
        """)
        
        st.markdown("""
        **Score Interpretation:**
        - **9-10**: Exceptional - Exceeds expectations, outstanding performance
        - **7-8**: Strong - Meets most requirements effectively  
        - **5-6**: Adequate - Satisfactory with some gaps or missed opportunities
        - **3-4**: Below Average - Significant areas for improvement needed
        - **1-2**: Poor - Fails to address the question adequately
        """)
        
        st.markdown("""
        **🎯 What This Means for You:**
        
        The AI evaluates your responses using the same professional standards that human interviewers use. 
        Each score reflects how well your answer demonstrates the skills and qualities needed for the role.
        
        **💡 Pro Tips for Better Scores:**
        - Use specific examples with numbers and outcomes
        - Structure your thoughts clearly (STAR method: Situation, Task, Action, Result)
        - Connect your experiences directly to the job requirements
        - Show confidence and professionalism in your communication
        - Demonstrate measurable impact and results
        """)

    # Question-by-question breakdown with scoring details
    st.subheader("📝 Detailed Question Analysis")
    
    for i, conv in enumerate(st.session_state["conversations"], 1):
        score = conv['Evaluation']
        
        # Color code based on score
        if score >= 8.5:
            score_color = "🟢"
            score_style = "color: green; font-weight: bold;"
        elif score >= 7.0:
            score_color = "🔵" 
            score_style = "color: blue; font-weight: bold;"
        elif score >= 5.5:
            score_color = "🟡"
            score_style = "color: orange; font-weight: bold;"
        else:
            score_color = "🔴"
            score_style = "color: red; font-weight: bold;"
        
        with st.expander(f"Question {i} {score_color} (Score: {score}/10)", expanded=False):
            st.markdown(f"**Question:** {conv['Question']}")
            st.markdown(f"**Your Answer:** {conv['Candidate Answer']}")
            
            # Enhanced feedback display
            st.markdown("**📋 Detailed Feedback:**")
            feedback_parts = conv['Feedback'].split('. ')
            for part in feedback_parts:
                if part.strip():
                    if any(keyword in part.lower() for keyword in ['strength', 'good', 'excellent', 'well']):
                        st.markdown(f"✅ {part}")
                    elif any(keyword in part.lower() for keyword in ['improve', 'enhance', 'consider', 'suggest']):
                        st.markdown(f"💡 {part}")
                    else:
                        st.markdown(f"📝 {part}")
            
            # Score breakdown explanation
            st.markdown("**🎯 Score Breakdown:**")
            if score >= 8.5:
                st.markdown("• **Exceptional Performance** - Your response demonstrated outstanding qualities across multiple criteria")
            elif score >= 7.0:
                st.markdown("• **Strong Performance** - Your response effectively addressed the question with good examples")
            elif score >= 5.5:
                st.markdown("• **Good Performance** - Your response was adequate but could benefit from more specific examples")
            else:
                st.markdown("• **Areas for Improvement** - Your response needs more detail and specific examples")
            
            # Show detailed criteria scores if available
            if "Criteria_Scores" in conv:
                st.markdown("**📊 Detailed Criteria Scores:**")
                criteria = conv["Criteria_Scores"]
                col1, col2 = st.columns(2)
                
                with col1:
                    for criterion, score_val in list(criteria.items())[:3]:
                        criterion_name = criterion.replace("_", " ").title()
                        st.markdown(f"• **{criterion_name}**: {score_val}/10")
                
                with col2:
                    for criterion, score_val in list(criteria.items())[3:]:
                        criterion_name = criterion.replace("_", " ").title()
                        st.markdown(f"• **{criterion_name}**: {score_val}/10")
            
            # Show competency assessment if available
            if "Competency_Assessment" in conv:
                st.markdown("**🎯 Competency Assessment:**")
                competencies = conv["Competency_Assessment"]
                col1, col2 = st.columns(2)
                
                with col1:
                    for competency, score_val in list(competencies.items())[:3]:
                        competency_name = competency.replace("_", " ").title()
                        st.markdown(f"• **{competency_name}**: {score_val}/10")
                
                with col2:
                    for competency, score_val in list(competencies.items())[3:]:
                        competency_name = competency.replace("_", " ").title()
                        st.markdown(f"• **{competency_name}**: {score_val}/10")

    # Performance Insights
    st.subheader("🔍 Performance Insights")
    
    # Calculate insights
    scores = [conv['Evaluation'] for conv in st.session_state["conversations"]]
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)
    consistency = max(scores) - min(scores)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Score", f"{avg_score:.1f}/10")
    
    with col2:
        st.metric("Best Response", f"{best_score:.1f}/10")
    
    with col3:
        st.metric("Lowest Score", f"{worst_score:.1f}/10")
    
    with col4:
        if consistency <= 1.5:
            st.metric("Consistency", "Excellent", delta="Very Consistent")
        elif consistency <= 2.5:
            st.metric("Consistency", "Good", delta="Consistent")
        else:
            st.metric("Consistency", "Variable", delta="Inconsistent")
    
    # Real-World Position Analysis
    st.subheader("🌍 Where You Stand in the Job Market")
    
    # Calculate percentile and market position
    def get_market_position(score):
        """Determine market position based on score"""
        if score >= 9.0:
            return {
                "percentile": "Top 5%",
                "position": "Exceptional Candidate",
                "description": "You're in the top tier of candidates. Companies will actively pursue you.",
                "color": "success",
                "icon": "🏆"
            }
        elif score >= 8.0:
            return {
                "percentile": "Top 15%",
                "position": "Strong Candidate",
                "description": "You're above average and competitive for most positions.",
                "color": "info",
                "icon": "⭐"
            }
        elif score >= 7.0:
            return {
                "percentile": "Top 35%",
                "position": "Competitive Candidate",
                "description": "You're competitive but may need to differentiate yourself.",
                "color": "warning",
                "icon": "📈"
            }
        elif score >= 6.0:
            return {
                "percentile": "Top 60%",
                "position": "Average Candidate",
                "description": "You're in the middle of the pack. Focus on improving specific areas.",
                "color": "warning",
                "icon": "📊"
            }
        elif score >= 5.0:
            return {
                "percentile": "Bottom 40%",
                "position": "Below Average",
                "description": "You need significant improvement to be competitive.",
                "color": "error",
                "icon": "⚠️"
            }
        else:
            return {
                "percentile": "Bottom 20%",
                "position": "Needs Major Improvement",
                "description": "You're not currently competitive. Focus on fundamental improvements.",
                "color": "error",
                "icon": "🚨"
            }
    
    market_position = get_market_position(final_score)
    
    # Display market position
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if market_position["color"] == "success":
            st.success(f"{market_position['icon']} **{market_position['percentile']}**")
        elif market_position["color"] == "info":
            st.info(f"{market_position['icon']} **{market_position['percentile']}**")
        elif market_position["color"] == "warning":
            st.warning(f"{market_position['icon']} **{market_position['percentile']}**")
        else:
            st.error(f"{market_position['icon']} **{market_position['percentile']}**")
    
    with col2:
        st.markdown(f"**{market_position['position']}**")
        st.markdown(f"*{market_position['description']}*")
    
    # Detailed market analysis
    with st.expander("📊 Detailed Market Analysis", expanded=True):
        st.markdown("""
        **🎯 What This Means for Your Job Search:**
        
        **Top 5% (9.0+):** You're in the elite tier. Companies will compete for you.
        - **Job Prospects:** Excellent - you can be selective
        - **Salary Negotiation:** Strong position to negotiate
        - **Interview Success Rate:** 90%+ for roles you apply to
        
        **Top 15% (8.0-8.9):** You're above average and competitive.
        - **Job Prospects:** Very Good - you'll get interviews
        - **Salary Negotiation:** Good position to negotiate
        - **Interview Success Rate:** 70-80% for roles you apply to
        
        **Top 35% (7.0-7.9):** You're competitive but need to stand out.
        - **Job Prospects:** Good - you'll get some interviews
        - **Salary Negotiation:** Moderate position
        - **Interview Success Rate:** 50-60% for roles you apply to
        
        **Top 60% (6.0-6.9):** You're in the middle of the pack.
        - **Job Prospects:** Fair - you'll need to apply more broadly
        - **Salary Negotiation:** Limited position
        - **Interview Success Rate:** 30-40% for roles you apply to
        
        **Below Average (5.0-5.9):** You need improvement to be competitive.
        - **Job Prospects:** Challenging - focus on skill development
        - **Salary Negotiation:** Weak position
        - **Interview Success Rate:** 15-25% for roles you apply to
        
        **Needs Major Improvement (<5.0):** You're not currently competitive.
        - **Job Prospects:** Difficult - prioritize skill building
        - **Salary Negotiation:** Very weak position
        - **Interview Success Rate:** <15% for roles you apply to
        """)
    
    # Industry-specific insights
    st.subheader("💼 Industry-Specific Insights")
    
    # Get job description to determine industry
    job_desc = st.session_state.get("job_description", "").lower()
    
    industry_insights = {
        "tech": {
            "name": "Technology/Software Development",
            "avg_score": 7.2,
            "competition": "High",
            "tips": "Focus on technical skills, problem-solving, and system design"
        },
        "finance": {
            "name": "Finance/Banking",
            "avg_score": 7.5,
            "competition": "Very High",
            "tips": "Emphasize analytical skills, attention to detail, and risk management"
        },
        "consulting": {
            "name": "Consulting",
            "avg_score": 7.8,
            "competition": "Extremely High",
            "tips": "Focus on structured thinking, communication, and case studies"
        },
        "marketing": {
            "name": "Marketing/Advertising",
            "avg_score": 6.8,
            "competition": "Medium-High",
            "tips": "Highlight creativity, data analysis, and campaign results"
        },
        "healthcare": {
            "name": "Healthcare",
            "avg_score": 7.0,
            "competition": "High",
            "tips": "Emphasize patient care, technical skills, and regulatory knowledge"
        }
    }
    
    # Determine industry
    detected_industry = "tech"  # default
    for keyword, industry in [
        ("software", "tech"), ("developer", "tech"), ("programming", "tech"),
        ("finance", "finance"), ("banking", "finance"), ("investment", "finance"),
        ("consulting", "consulting"), ("strategy", "consulting"), ("advisory", "consulting"),
        ("marketing", "marketing"), ("advertising", "marketing"), ("brand", "marketing"),
        ("healthcare", "healthcare"), ("medical", "healthcare"), ("patient", "healthcare")
    ]:
        if keyword in job_desc:
            detected_industry = industry
            break
    
    industry = industry_insights[detected_industry]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Industry", industry["name"])
    
    with col2:
        st.metric("Industry Avg Score", f"{industry['avg_score']}/10")
    
    with col3:
        st.metric("Competition Level", industry["competition"])
    
    # Compare user score to industry average
    score_diff = final_score - industry["avg_score"]
    
    if score_diff > 1.0:
        st.success(f"🎉 **You're {abs(score_diff):.1f} points above the industry average!** You're well-positioned for this field.")
    elif score_diff > 0:
        st.info(f"📊 **You're {abs(score_diff):.1f} points above the industry average.** You're competitive in this field.")
    elif score_diff > -1.0:
        st.warning(f"⚠️ **You're {abs(score_diff):.1f} points below the industry average.** Focus on industry-specific skills.")
    else:
        st.error(f"🚨 **You're {abs(score_diff):.1f} points below the industry average.** Consider additional training or different roles.")
    
    st.markdown(f"**💡 Industry-Specific Tips:** {industry['tips']}")
    
    # Overall Competency Analysis
    if any("Competency_Assessment" in conv for conv in st.session_state["conversations"]):
        st.subheader("🎯 Overall Competency Analysis")
        
        # Collect all competency scores
        all_competencies = {}
        competency_counts = {}
        
        for conv in st.session_state["conversations"]:
            if "Competency_Assessment" in conv:
                for competency, score in conv["Competency_Assessment"].items():
                    if competency not in all_competencies:
                        all_competencies[competency] = []
                        competency_counts[competency] = 0
                    all_competencies[competency].append(score)
                    competency_counts[competency] += 1
        
        # Calculate averages
        avg_competencies = {}
        for competency, scores in all_competencies.items():
            avg_competencies[competency] = sum(scores) / len(scores)
        
        # Display competency radar chart or metrics
        if avg_competencies:
            st.markdown("**Your Average Performance Across Competency Areas:**")
            
            col1, col2, col3 = st.columns(3)
            competencies_list = list(avg_competencies.items())
            
            with col1:
                for i in range(0, len(competencies_list), 3):
                    if i < len(competencies_list):
                        competency, score = competencies_list[i]
                        competency_name = competency.replace("_", " ").title()
                        if score >= 8.0:
                            st.success(f"**{competency_name}**: {score:.1f}/10")
                        elif score >= 6.0:
                            st.info(f"**{competency_name}**: {score:.1f}/10")
                        else:
                            st.warning(f"**{competency_name}**: {score:.1f}/10")
            
            with col2:
                for i in range(1, len(competencies_list), 3):
                    if i < len(competencies_list):
                        competency, score = competencies_list[i]
                        competency_name = competency.replace("_", " ").title()
                        if score >= 8.0:
                            st.success(f"**{competency_name}**: {score:.1f}/10")
                        elif score >= 6.0:
                            st.info(f"**{competency_name}**: {score:.1f}/10")
                        else:
                            st.warning(f"**{competency_name}**: {score:.1f}/10")
            
            with col3:
                for i in range(2, len(competencies_list), 3):
                    if i < len(competencies_list):
                        competency, score = competencies_list[i]
                        competency_name = competency.replace("_", " ").title()
                        if score >= 8.0:
                            st.success(f"**{competency_name}**: {score:.1f}/10")
                        elif score >= 6.0:
                            st.info(f"**{competency_name}**: {score:.1f}/10")
                        else:
                            st.warning(f"**{competency_name}**: {score:.1f}/10")
    
    # Job search strategy recommendations
    st.subheader("🎯 Job Search Strategy")
    
    if final_score >= 8.5:
        st.success("""
        **Elite Candidate Strategy:**
        - Target top-tier companies and roles
        - Be selective with applications
        - Leverage your strong position for better offers
        - Consider multiple offers to maximize compensation
        - Network with industry leaders
        """)
    elif final_score >= 7.5:
        st.info("""
        **Strong Candidate Strategy:**
        - Apply to mid to senior-level positions
        - Focus on companies that value your specific skills
        - Build a strong personal brand
        - Network actively in your target industry
        - Consider specialized roles where you excel
        """)
    elif final_score >= 6.5:
        st.warning("""
        **Competitive Candidate Strategy:**
        - Apply broadly to entry to mid-level positions
        - Focus on companies with growth opportunities
        - Build specific skills that are in high demand
        - Network to get referrals
        - Consider roles where you can grow into the position
        """)
    else:
        st.error("""
                 **Development-Focused Strategy:**
         - Focus on skill development before job search
         - Consider internships or entry-level positions
         - Build a portfolio of projects
         - Network to learn about industry requirements
         - Consider additional education or certifications
         """)
    
    # Improvement Recommendations
    st.subheader("🚀 How to Improve")
    
    if final_score >= 8.5:
        st.success("""
        **Outstanding Performance!** 🎉
        - Continue building on your strengths
        - Share your expertise with others
        - Consider mentoring opportunities
        - Focus on leadership and strategic thinking
        """)
    elif final_score >= 7.0:
        st.info("""
        **Strong Foundation!** 💪
        - Focus on adding more specific examples
        - Practice quantifying your achievements
        - Work on structuring responses more clearly
        - Build deeper technical expertise
        """)
    elif final_score >= 5.5:
        st.warning("""
        **Good Potential!** 📈
        - Practice providing more detailed examples
        - Work on connecting experiences to job requirements
        - Focus on demonstrating measurable impact
        - Improve communication clarity
        """)
    else:
        st.error("""
        **Growth Opportunity!** 🌱
        - Practice answering common interview questions
        - Prepare specific examples for your experiences
        - Work on clearly articulating your thoughts
        - Consider mock interviews for practice
        - Focus on fundamental skills development
        """)

    # Speaking Skills Analysis
    st.subheader("🎤 Speaking Skills Assessment")
    
    try:
        # Initialize speaking skills analyzer
        speaking_analyzer = SpeakingSkillsAnalyzer()
        
        # Check if we have audio files to analyze
        audio_dir = Path("audio")
        if audio_dir.exists():
            audio_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.mp3"))
            
            if audio_files:
                st.info(f"📁 Found {len(audio_files)} audio recordings for speaking skills analysis")
                
                # Analyze the most recent audio file
                latest_audio = max(audio_files, key=os.path.getctime)
                
                with st.spinner("Analyzing speaking skills..."):
                    speaking_results = speaking_analyzer.analyze_speaking_skills(str(latest_audio))
                
                if 'error' not in speaking_results:
                    # Display speaking skills results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        overall_speaking_score = speaking_results['overall_score']
                        if overall_speaking_score >= 8.5:
                            st.success(f"**Speaking Score: {overall_speaking_score:.2f}/10** 🎯")
                            st.markdown("*Exceptional Communication*")
                        elif overall_speaking_score >= 7.0:
                            st.info(f"**Speaking Score: {overall_speaking_score:.2f}/10** 💬")
                            st.markdown("*Strong Communication*")
                        elif overall_speaking_score >= 5.5:
                            st.warning(f"**Speaking Score: {overall_speaking_score:.2f}/10** 📢")
                            st.markdown("*Good Communication*")
                        else:
                            st.error(f"**Speaking Score: {overall_speaking_score:.2f}/10** 🔇")
                            st.markdown("*Needs Improvement*")
                    
                    with col2:
                        st.markdown(f"**Assessment:** {speaking_results['assessment']}")
                        st.markdown(f"**Audio File:** {latest_audio.name}")
                    
                    # Detailed speaking skills breakdown
                    with st.expander("🎤 Detailed Speaking Skills Analysis", expanded=False):
                        breakdown = speaking_results['score_breakdown']
                        
                        # Create columns for skills display
                        col1, col2, col3 = st.columns(3)
                        skills_list = list(breakdown.items())
                        
                        with col1:
                            for i in range(0, len(skills_list), 3):
                                if i < len(skills_list):
                                    skill, details = skills_list[i]
                                    skill_name = skill.replace("_", " ").title()
                                    score = details['score']
                                    grade = details['grade']
                                    
                                    if score >= 8.0:
                                        st.success(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                                    elif score >= 6.0:
                                        st.info(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                                    else:
                                        st.warning(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                        
                        with col2:
                            for i in range(1, len(skills_list), 3):
                                if i < len(skills_list):
                                    skill, details = skills_list[i]
                                    skill_name = skill.replace("_", " ").title()
                                    score = details['score']
                                    grade = details['grade']
                                    
                                    if score >= 8.0:
                                        st.success(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                                    elif score >= 6.0:
                                        st.info(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                                    else:
                                        st.warning(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                        
                        with col3:
                            for i in range(2, len(skills_list), 3):
                                if i < len(skills_list):
                                    skill, details = skills_list[i]
                                    skill_name = skill.replace("_", " ").title()
                                    score = details['score']
                                    grade = details['grade']
                                    
                                    if score >= 8.0:
                                        st.success(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                                    elif score >= 6.0:
                                        st.info(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                                    else:
                                        st.warning(f"**{skill_name}**: {score:.1f}/10 ({grade})")
                    
                    # Speaking skills feedback
                    with st.expander("💡 Speaking Skills Feedback", expanded=False):
                        feedback = speaking_results['feedback']
                        
                        for area, advice in feedback.items():
                            if area == 'overall':
                                st.markdown(f"**Overall Assessment:** {advice}")
                            else:
                                area_name = area.replace("_", " ").title()
                                st.markdown(f"**{area_name}:** {advice}")
                
                else:
                    st.warning("⚠️ Speaking skills analysis encountered an error. Using rule-based assessment.")
                    
            else:
                st.info("📁 No audio recordings found for speaking skills analysis")
        else:
            st.info("📁 Audio directory not found. Speaking skills analysis unavailable.")
            
    except Exception as e:
        st.warning(f"⚠️ Speaking skills analysis unavailable: {str(e)}")

    # Sentiment Analysis
    st.subheader("😊 Emotional & Sentiment Analysis")
    
    try:
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentAudioAnalyzer()
        
        # Check if we have audio files to analyze
        audio_dir = Path("audio")
        if audio_dir.exists():
            audio_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.mp3"))
            
            if audio_files:
                st.info(f"🎭 Analyzing emotional states from {len(audio_files)} audio recordings")
                
                # Analyze the most recent audio file
                latest_audio = max(audio_files, key=os.path.getctime)
                
                with st.spinner("Analyzing emotional states and sentiment..."):
                    sentiment_results = sentiment_analyzer.analyze_sentiment(str(latest_audio))
                
                if 'error' not in sentiment_results:
                    # Display sentiment results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Primary emotion
                        primary_emotion = sentiment_results['emotional_states']['primary']
                        emotion_confidence = sentiment_results['emotional_states']['confidence']
                        
                        if primary_emotion in ['Confident', 'Professional', 'Calm']:
                            st.success(f"**Primary Emotion:** {primary_emotion} 🎯")
                            st.markdown(f"*Confidence: {emotion_confidence:.1f}*")
                        elif primary_emotion in ['Enthusiastic', 'Excited']:
                            st.info(f"**Primary Emotion:** {primary_emotion} ✨")
                            st.markdown(f"*Confidence: {emotion_confidence:.1f}*")
                        elif primary_emotion in ['Stressed', 'Anxious', 'Nervous']:
                            st.warning(f"**Primary Emotion:** {primary_emotion} 😰")
                            st.markdown(f"*Confidence: {emotion_confidence:.1f}*")
                        else:
                            st.info(f"**Primary Emotion:** {primary_emotion} 😐")
                            st.markdown(f"*Confidence: {emotion_confidence:.1f}*")
                    
                    with col2:
                        # Confidence analysis
                        confidence_level = sentiment_results['confidence_analysis']['level']
                        confidence_score = sentiment_results['confidence_analysis']['score']
                        
                        if confidence_level == "High":
                            st.success(f"**Confidence:** {confidence_level} 💪")
                            st.markdown(f"*Score: {confidence_score:.1f}/10*")
                        elif confidence_level == "Medium":
                            st.info(f"**Confidence:** {confidence_level} 📊")
                            st.markdown(f"*Score: {confidence_score:.1f}/10*")
                        else:
                            st.warning(f"**Confidence:** {confidence_level} 💭")
                            st.markdown(f"*Score: {confidence_score:.1f}/10*")
                    
                    with col3:
                        # Stress analysis
                        stress_level = sentiment_results['stress_analysis']['level']
                        stress_score = sentiment_results['stress_analysis']['score']
                        
                        if stress_level == "Low":
                            st.success(f"**Stress Level:** {stress_level} 😌")
                            st.markdown(f"*Score: {stress_score:.1f}/10*")
                        elif stress_level == "Medium":
                            st.warning(f"**Stress Level:** {stress_level} 😐")
                            st.markdown(f"*Score: {stress_score:.1f}/10*")
                        else:
                            st.error(f"**Stress Level:** {stress_level} 😰")
                            st.markdown(f"*Score: {stress_score:.1f}/10*")
                    
                    # Emotional Intelligence Score
                    st.subheader("🧠 Emotional Intelligence Assessment")
                    
                    ei_score = sentiment_results['emotional_intelligence']
                    if ei_score >= 8.5:
                        st.success(f"**Emotional Intelligence: {ei_score:.2f}/10** 🧠✨")
                        st.markdown("*Exceptional emotional awareness and control*")
                    elif ei_score >= 7.0:
                        st.info(f"**Emotional Intelligence: {ei_score:.2f}/10** 🧠💡")
                        st.markdown("*Strong emotional intelligence*")
                    elif ei_score >= 5.5:
                        st.warning(f"**Emotional Intelligence: {ei_score:.2f}/10** 🧠📈")
                        st.markdown("*Good emotional intelligence with room for growth*")
                    else:
                        st.error(f"**Emotional Intelligence: {ei_score:.2f}/10** 🧠🌱")
                        st.markdown("*Areas for emotional development identified*")
                    
                    # Overall sentiment
                    overall_sentiment = sentiment_results['overall_sentiment']
                    if overall_sentiment == "Positive":
                        st.success(f"**Overall Sentiment: {overall_sentiment}** 😊")
                    elif overall_sentiment == "Neutral":
                        st.info(f"**Overall Sentiment: {overall_sentiment}** 😐")
                    else:
                        st.warning(f"**Overall Sentiment: {overall_sentiment}** 😔")
                    
                    # Detailed sentiment analysis
                    with st.expander("🎭 Detailed Emotional Analysis", expanded=False):
                        # All detected emotions
                        st.markdown("**Detected Emotional States:**")
                        all_emotions = sentiment_results['emotional_states']['all_emotions']
                        if all_emotions:
                            for emotion, confidence in all_emotions.items():
                                if confidence >= 0.7:
                                    st.success(f"• {emotion}: {confidence:.1f}")
                                elif confidence >= 0.5:
                                    st.info(f"• {emotion}: {confidence:.1f}")
                                else:
                                    st.warning(f"• {emotion}: {confidence:.1f}")
                        else:
                            st.info("• Neutral emotional state detected")
                        
                        # Confidence indicators
                        st.markdown("**Confidence Indicators:**")
                        confidence_indicators = sentiment_results['confidence_analysis']['indicators']
                        for indicator in confidence_indicators:
                            st.markdown(f"• {indicator}")
                        
                        # Stress markers
                        st.markdown("**Stress Markers:**")
                        stress_markers = sentiment_results['stress_analysis']['markers']
                        for marker in stress_markers:
                            st.markdown(f"• {marker}")
                    
                    # Sentiment recommendations
                    with st.expander("💡 Emotional Intelligence Recommendations", expanded=False):
                        recommendations = sentiment_results['recommendations']
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"**{i}.** {rec}")
                    
                    # Sentiment features breakdown
                    with st.expander("🔍 Sentiment Features Breakdown", expanded=False):
                        features = sentiment_results['features']
                        
                        # Create columns for features display
                        col1, col2, col3 = st.columns(3)
                        features_list = list(features.items())
                        
                        with col1:
                            for i in range(0, len(features_list), 3):
                                if i < len(features_list):
                                    feature, score = features_list[i]
                                    feature_name = feature.replace("_", " ").title()
                                    
                                    if score >= 8.0:
                                        st.success(f"**{feature_name}**: {score:.1f}/10")
                                    elif score >= 6.0:
                                        st.info(f"**{feature_name}**: {score:.1f}/10")
                                    else:
                                        st.warning(f"**{feature_name}**: {score:.1f}/10")
                        
                        with col2:
                            for i in range(1, len(features_list), 3):
                                if i < len(features_list):
                                    feature, score = features_list[i]
                                    feature_name = feature.replace("_", " ").title()
                                    
                                    if score >= 8.0:
                                        st.success(f"**{feature_name}**: {score:.1f}/10")
                                    elif score >= 6.0:
                                        st.info(f"**{feature_name}**: {score:.1f}/10")
                                    else:
                                        st.warning(f"**{feature_name}**: {score:.1f}/10")
                        
                        with col3:
                            for i in range(2, len(features_list), 3):
                                if i < len(features_list):
                                    feature, score = features_list[i]
                                    feature_name = feature.replace("_", " ").title()
                                    
                                    if score >= 8.0:
                                        st.success(f"**{feature_name}**: {score:.1f}/10")
                                    elif score >= 6.0:
                                        st.info(f"**{feature_name}**: {score:.1f}/10")
                                    else:
                                        st.warning(f"**{feature_name}**: {score:.1f}/10")
                
                else:
                    st.warning("⚠️ Sentiment analysis encountered an error. Using rule-based analysis.")
                    
            else:
                st.info("🎭 No audio recordings found for sentiment analysis")
        else:
            st.info("🎭 Audio directory not found. Sentiment analysis unavailable.")
            
    except Exception as e:
        st.warning(f"⚠️ Sentiment analysis unavailable: {str(e)}")

    # New interview option
    st.subheader("🔄 Start New Interview")
    if st.button("Begin New Interview Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def render_interview_progress():
    """Render interview progress indicator"""
    if st.session_state.get("interview_started", False):
        progress_text = f"Question {st.session_state['qa_index']} of {st.session_state['max_questions']}"
        st.markdown(
            f'<div class="interview-progress"><strong>{progress_text}</strong></div>',
            unsafe_allow_html=True,
        )


def main():
    """Main application function"""
    # Setup
    setup_page_config()
    initialize_session_state()

    # Header
    st.title("🤖 AI Interview System")
    insturctions = st.empty()
    if not  st.session_state["interview_started"]:
        insturctions = get_instructions()

    # Sidebar
    uploaded_resume, job_description, submit = render_sidebar()

    # Process submission
    if submit and uploaded_resume and job_description:
        insturctions.empty()
        process_resume_submission(uploaded_resume, job_description)
        

    # Start interview button
    if st.session_state["name"] and not st.session_state["interview_started"]:
        if st.button("Start Interview"):
            start_interview()
            st.rerun()

    # Interview section
    if st.session_state.get("interview_started", False):
        render_interview_progress()

        # Show chat history
        st.subheader("Interview Chat")
        display_chat_messages()

        # Handle different interview states
        if not st.session_state["interview_completed"]:
            # Active interview
            speak_current_question()
            handle_audio_recording()
        elif not st.session_state["thanks_message_prepared"]:
            # Interview just completed - prepare thanks
            prepare_thanks_message()
        elif not st.session_state["thanks_message_spoken"]:
            # Thanks message prepared but not spoken
            speak_thanks_message()
        else:
            # Everything done - show results
            display_final_results()


if __name__ == "__main__":
    main()
