# 🤖 AI Interview System

An intelligent interview platform that conducts automated job interviews using AI. The system analyzes candidate resumes, asks relevant questions, and provides detailed feedback and scoring.

## ✨ Features

- **Resume Analysis**: Upload your PDF resume and get key highlights extracted automatically
- **Personalized Questions**: AI generates interview questions based on your resume and the job description
- **Voice Interaction**: Speak your answers naturally - the system will transcribe and analyze them
- **Real-time Chat**: Chat interface showing the conversation flow
- **Intelligent Scoring**: Get detailed feedback and scores for each answer
- **Complete Evaluation**: Receive an overall interview score and comprehensive report

## 🚀 How It Works

### 1. Setup
- Upload your resume (PDF format)
- Paste the job description you're applying for
- Select Maximum number of Questions (Optional)
- Select AI Interviewer Voice (Optional)
- Click "Submit" to process your information

### 2. Interview Process
- Click "Start Interview" to begin
- The AI will greet you and ask the first question
- Listen to each question (text-to-speech enabled)
- Record your answer using the audio recorder **(Make sure to use Chrome Browser Only)**
- The system transcribes and analyzes your response
- Receive the next question based on your previous answers

### 3. Get Results
- Complete selected number interview questions
- Receive detailed feedback for each answer
- Get an overall interview score out of 10
- Review the complete chat history and evaluation report

## 🎯 What Makes It Special

- **Adaptive Questioning**: Each question builds on your previous answers
- **Natural Conversation**: Feels like talking to a real interviewer
- **Detailed Feedback**: Understand what you did well and areas for improvement
- **Professional Interface**: Clean, easy-to-use chat-based design
- **Complete Documentation**: Full interview transcript and scoring breakdown

## 📋 Requirements
- Internet connection for AI processing
- Microphone access for recording answers
- PDF resume file
- Job description text
- LLM API key for AI processing
    - Supported Models (LiteLLM): https://docs.litellm.ai/docs/providers (Change LLM_MODEL in .env)
    - Free Experimental Model from MistralAI: https://mistral.ai/
    - Note: If you're using a different model provider such as OpenAI, be sure to update the environment variable from MISTRAL_API_KEY to OPENAI_API_KEY as per the LiteLLM guidelines.
- Speechmatics API key for speech-to-text
    - Speechmatics Platform: https://www.speechmatics.com/

## 🎨 Interface

The system features a modern chat interface similar to ChatGPT:
- **AI Interviewer** messages appear on the left (questions and instructions)
- **Your responses** appear on the right (transcribed from audio)
- **Progress tracker** shows which question you're on
- **Audio recorder** for easy voice input

## 📊 Scoring System

Each answer receives:
- Individual score (0-10)
- Detailed written feedback
- Suggestions for improvement

Final results include:
- Overall interview score
- Question-by-question breakdown
- Complete conversation history
- Personalized recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- API keys for LLM and Speech-to-Text services

### Installation Steps

1. **Clone the repository** (if not already cloned)
   ```bash
   git clone <repository-url>
   cd AI-Interview-System
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows (Git Bash):
   source .venv/Scripts/activate
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy example file
   copy env_example.txt .env    # Windows
   # OR
   cp env_example.txt .env      # Mac/Linux
   
   # Edit .env file and add your API keys:
   # - LLM_MODEL (e.g., mistral/mistral-large-latest)
   # - MISTRAL_API_KEY or OPENAI_API_KEY
   # - SPEECHMATICS_API_KEY
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   - Navigate to http://localhost:8501
   - **Important**: Use Chrome browser for microphone access

📖 For detailed setup instructions and troubleshooting, see [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)


## 🔄 Multiple Interviews

- Take multiple practice interviews
- Try different job descriptions
- Track your improvement over time
- Perfect your interview skills

## 💡 Tips for Best Results

- Speak clearly when recording answers
- Provide detailed, specific responses
- Take your time - there's no rush
- Treat it like a real interview
- Review feedback to improve

## 📚 Documentation

Additional documentation is available in the `docs/` folder:
- [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) - Detailed setup and troubleshooting guide
- [SCORING_EXPLANATION.md](docs/SCORING_EXPLANATION.md) - How the scoring system works
- [TECH_STACK_DOCUMENTATION.md](docs/TECH_STACK_DOCUMENTATION.md) - Technical architecture details
- [DATASET_GUIDE.md](docs/DATASET_GUIDE.md) - Dataset information and usage

## 📁 Project Structure

```
AI-Interview-System/
├── app.py                 # Main Streamlit application
├── utils/                 # Utility modules
│   ├── analyze_candidate.py
│   ├── evaluation.py
│   ├── llm_call.py
│   ├── text_to_speech.py
│   └── ...
├── models/                # Trained models
├── audio/                 # Audio files (generated during interviews)
├── outputs/               # Interview results (JSON files)
├── docs/                  # Documentation files
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
└── README.md              # This file
```

---

*Ready to ace your next interview? Upload your resume and get started!*
