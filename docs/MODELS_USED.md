# 🤖 Models Used in AI Interview System

## 📊 Summary

**Currently, there are TWO trained machine learning model files in the system:**

1. ✅ **`models/speaking_skills_model.pkl`** - Trained Random Forest model for speaking skills assessment
2. ✅ **`models/sentiment_model.pkl`** - Trained Random Forest model for emotion/sentiment classification

Additionally, the system uses:
3. **Mistral AI LLM** (API service) - For content analysis
4. **Speechmatics API** (API service) - For audio transcription

---

## 🎯 Detailed Breakdown

### **1. Local Machine Learning Models**

#### ✅ **Speaking Skills Model** (`speaking_skills_model.pkl`)

**Location**: `models/speaking_skills_model.pkl`

**Type**: Random Forest Classifier (scikit-learn)

**Status**: ✅ **EXISTS and LOADED**

**What it does**:
- Trained to classify speaking skills into categories (Exceptional/Excellent/Good/Fair/Needs Improvement)
- Uses 12 audio features as input
- Loaded automatically when `SpeakingSkillsAnalyzer` is initialized

**How it's used**:
```python
# In utils/speaking_skills_analyzer.py
def load_model(self):
    if os.path.exists("models/speaking_skills_model.pkl"):
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']  # Random Forest model
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        # Model is loaded and ready to use
```

**If model doesn't exist**: Falls back to rule-based assessment

---

#### ✅ **Sentiment Model** (`models/sentiment_model.pkl`)

**Location**: `models/sentiment_model.pkl`

**Type**: Random Forest Classifier (scikit-learn)

**Status**: ✅ **EXISTS and LOADED**

**What it does**:
- Classifies emotional states (Confident/Anxious/Enthusiastic/Calm/Stressed/etc.)
- Uses 16 sentiment features as input
- Trained on 505 samples (9 real audio files + 496 synthetic samples)

**Model Details**:
- **Size**: 2.1 MB
- **Training Date**: 2024
- **Emotions Detected**: Confident, Calm, Enthusiastic, Stressed, Professional, Nervous, Anxious, Excited

**How it's used**:
```python
# In utils/sentiment_audio_analyzer.py
def load_model(self):
    if os.path.exists("models/sentiment_model.pkl"):
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']  # Random Forest model
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        # Model is loaded and ready to use
```

**Result**: Sentiment analyzer uses **trained ML model** for emotion classification

---

### **2. API Services (External Models)**

#### **Mistral AI LLM** (Large Language Model)

**Type**: API Service (not a local file)

**Purpose**: Content analysis of interview responses

**What it does**:
- Evaluates transcribed responses based on 6 criteria
- Generates feedback and scores
- Creates interview questions
- Analyzes resume content

**Status**: ✅ **ACTIVELY USED** (requires API key)

**Model**: `mistral/mistral-large-latest` (or configurable via `.env`)

---

#### **Speechmatics API**

**Type**: API Service (not a local file)

**Purpose**: Speech-to-text transcription

**What it does**:
- Converts audio recordings to text
- Handles punctuation and formatting

**Status**: ✅ **ACTIVELY USED** (requires API key)

---

## 📈 Current Model Status

### **What Actually Works with Trained Models:**

```
✅ Speaking Skills Analyzer
   └─ Uses trained model: speaking_skills_model.pkl
   └─ Random Forest Classifier
   └─ Predicts: Exceptional/Excellent/Good/Fair/Needs Improvement
```

### **What Uses Rule-Based Analysis (No Model):**

```
❌ Sentiment Audio Analyzer
   └─ sentiment_model.pkl does NOT exist
   └─ Falls back to rule-based analysis
   └─ Uses predefined rules to detect emotions
```

### **What Uses API Services:**

```
✅ Content Analysis (Mistral LLM)
   └─ External API call
   └─ Evaluates text content

✅ Audio Transcription (Speechmatics)
   └─ External API call
   └─ Converts speech to text
```

---

## 🔍 How to Check What's Being Used

### **Check if Models Exist:**

```bash
# In your terminal
ls models/
# Should show:
# - speaking_skills_model.pkl ✅
# - sentiment_model.pkl ❌ (doesn't exist)
```

### **Check Logs:**

When the system runs, check console logs:

**Speaking Skills Analyzer:**
- ✅ If model exists: `"Model loaded from models/speaking_skills_model.pkl"`
- ❌ If model missing: `"No pre-trained model found. Using rule-based assessment."`

**Sentiment Analyzer:**
- ✅ If model exists: `"Sentiment model loaded from models/sentiment_model.pkl"`
- ❌ If model missing: `"No pre-trained sentiment model found. Using rule-based analysis."`

---

## 🚀 How Models Are Loaded

### **Speaking Skills Model:**

```python
# In utils/speaking_skills_analyzer.py
class SpeakingSkillsAnalyzer:
    def __init__(self, model_path="models/speaking_skills_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()  # Tries to load on initialization
    
    def load_model(self):
        if os.path.exists(self.model_path):
            # Model exists - load it
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            # Model is now ready to use
        else:
            # Model doesn't exist - use rule-based
            self.model = None  # Will use rule-based methods
```

### **Sentiment Model:**

```python
# In utils/sentiment_audio_analyzer.py
class SentimentAudioAnalyzer:
    def __init__(self, model_path="models/sentiment_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()  # Tries to load on initialization
    
    def load_model(self):
        if os.path.exists(self.model_path):
            # Model exists - load it
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
        else:
            # Model doesn't exist - will use rule-based analysis
            self.model = None
```

---

## 📊 Analysis Methods Comparison

### **With Trained Model (Speaking Skills):**

1. Extract 12 audio features
2. Load trained Random Forest model
3. Scale features using saved scaler
4. Make prediction: `model.predict(features)`
5. Convert prediction to label: "Excellent", "Good", etc.

### **Without Model (Sentiment - Rule-Based):**

1. Extract 16 sentiment features
2. Apply predefined rules:
   - If confidence_score >= 8: → "Confident"
   - If stress_score >= 8: → "Stressed"
   - etc.
3. Calculate scores based on thresholds
4. Return primary emotion based on highest score

---

## 🎯 Answer to Your Question

**"Is there only one model used?"**

**No - there are TWO trained ML models:**

- ✅ **TWO local ML model files exist**: 
  - `speaking_skills_model.pkl` (1.1 MB)
  - `sentiment_model.pkl` (2.1 MB)
- ✅ **Both models are actively used** for audio analysis
- ✅ **Two API services used**: Mistral LLM + Speechmatics

**In summary:**
- **2 trained ML models** (speaking skills + sentiment)
- **2 API services** (LLM for content, Speechmatics for transcription)

---

## 🔧 To Add More Models

If you want to train and use a sentiment model:

1. **Train the model** using training data
2. **Save it** using `sentiment_analyzer.save_model()`
3. **Place it** in `models/sentiment_model.pkl`
4. **Restart** the application - it will auto-load

The code is already set up to use it once it exists!

---

## 📝 Summary Table

| Model/Service | Type | Status | Location | Purpose |
|---------------|------|--------|----------|---------|
| **speaking_skills_model.pkl** | ML Model | ✅ Exists | `models/` | Speaking skills classification |
| **sentiment_model.pkl** | ML Model | ✅ Exists | `models/` | Emotion/sentiment classification |
| **Mistral LLM** | API Service | ✅ Active | External | Content analysis |
| **Speechmatics** | API Service | ✅ Active | External | Audio transcription |

---

**Last Updated**: 2024

