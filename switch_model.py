#!/usr/bin/env python3
"""
Script to switch to a smaller Mistral model to avoid rate limits
"""

import os

def switch_to_smaller_model():
    """Switch to a smaller model that's less likely to hit rate limits"""
    
    print("🔄 Switching to smaller model to avoid rate limits...")
    
    # Read current .env file
    env_content = ""
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
    except FileNotFoundError:
        print("❌ .env file not found")
        return
    
    # Replace the model
    if "mistral/mistral-large-latest" in env_content:
        new_content = env_content.replace(
            "LLM_MODEL=mistral/mistral-large-latest",
            "LLM_MODEL=mistral/mistral-small-latest"
        )
        
        # Write back to .env
        with open('.env', 'w') as f:
            f.write(new_content)
        
        print("✅ Switched to mistral/mistral-small-latest")
        print("💡 This model is smaller and less likely to hit rate limits")
        
    elif "mistral/mistral-small-latest" in env_content:
        print("ℹ️  Already using mistral/mistral-small-latest")
        
    else:
        print("⚠️  Could not find model to switch. Current content:")
        print(env_content)

def show_current_model():
    """Show the current model being used"""
    try:
        with open('.env', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('LLM_MODEL='):
                    print(f"Current model: {line}")
                    return
    except FileNotFoundError:
        print("❌ .env file not found")

if __name__ == "__main__":
    print("🤖 Model Switcher for AI Interview System")
    print("=" * 50)
    
    show_current_model()
    print()
    
    switch_to_smaller_model()
    
    print("\n📝 Next steps:")
    print("1. Restart the Streamlit app")
    print("2. Try uploading your resume again")
    print("3. If still having issues, wait a few minutes and try again")

