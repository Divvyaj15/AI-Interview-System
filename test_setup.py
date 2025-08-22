#!/usr/bin/env python3
"""
Test script to verify AI Interview System setup
Run this to check if your environment is configured correctly
"""

import os
import sys
from dotenv import load_dotenv

def test_environment():
    """Test if environment variables are set correctly"""
    print("🔍 Testing Environment Setup...")
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        load_dotenv()
        print("✅ .env file found and loaded")
    else:
        print("⚠️  .env file not found")
        print("   Create a .env file with your API keys")
    
    # Check required environment variables
    required_vars = {
        'LLM_MODEL': 'LLM model to use (e.g., mistral/mistral-large-latest)',
        'MISTRAL_API_KEY': 'Mistral AI API key (or OPENAI_API_KEY for OpenAI)',
        'SPEECHMATICS_API_KEY': 'Speechmatics API key for speech-to-text'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"✅ {var}: {value[:20]}...")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your .env file")
        return False
    
    print("\n✅ Environment variables are set correctly!")
    return True

def test_dependencies():
    """Test if required packages are installed"""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        'streamlit',
        'litellm', 
        'edge-tts',
        'pygame',
        'speechmatics-python',
        'pypdf',
        'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def test_llm_connection():
    """Test LLM API connection"""
    print("\n🤖 Testing LLM Connection...")
    
    try:
        from utils.llm_call import get_response_from_llm
        
        # Test with a simple prompt
        test_prompt = "Hello! Please respond with just 'OK' and nothing else."
        response = get_response_from_llm(test_prompt)
        
        if response:
            print(f"✅ LLM API working! Response: {response[:50]}...")
            return True
        else:
            print("❌ LLM API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ LLM API test failed: {str(e)}")
        print("   Check your API key and model configuration")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Interview System - Setup Test")
    print("=" * 50)
    
    env_ok = test_environment()
    deps_ok = test_dependencies()
    
    if not env_ok or not deps_ok:
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return
    
    # Only test LLM if environment is set up
    llm_ok = test_llm_connection()
    
    print("\n" + "=" * 50)
    if env_ok and deps_ok and llm_ok:
        print("🎉 All tests passed! Your system is ready to run.")
        print("\nTo start the app:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    
    print("\n📝 Next steps:")
    print("1. Ensure your .env file has valid API keys")
    print("2. Run: streamlit run app.py")
    print("3. Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    main()
