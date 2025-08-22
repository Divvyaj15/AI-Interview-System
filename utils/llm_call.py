from litellm import completion
import os
import json

LLM_MODEL = os.environ.get("LLM_MODEL", "mistral/mistral-large-latest")


def get_response_from_llm(prompt):
    """
    Calls the LLM and returns the response.

    Args:
        prompt (str): The string to prompt the LLM with.

    Returns:
        str: The response from the LLM.
    """
    try:
        response = completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        print(f"LLM API Error: {error_msg}")
        print(f"Model: {LLM_MODEL}")
        
        # Handle specific error types
        if "RateLimitError" in error_msg or "capacity exceeded" in error_msg:
            print("⚠️  Rate limit exceeded. The free tier has usage limits.")
            print("💡 Try again in a few minutes or switch to a different model.")
            raise Exception(f"Rate limit exceeded: {error_msg}")
        elif "401" in error_msg or "authentication" in error_msg.lower():
            print("❌ Authentication failed. Check your API key.")
            raise Exception(f"Authentication failed: {error_msg}")
        elif "quota" in error_msg.lower():
            print("❌ Quota exceeded. Check your API usage limits.")
            raise Exception(f"Quota exceeded: {error_msg}")
        else:
            print("❌ Unknown API error. Check your configuration.")
            raise Exception(f"LLM API call failed: {error_msg}")


def parse_json_response(response):
    # Parse the JSON response
    try:
        if not response:
            print("Warning: Empty response from LLM")
            return None
            
        # Clean up the response
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        print(f"Attempting to parse JSON response: {cleaned_response[:200]}...")
        
        parsed = json.loads(cleaned_response)
        print(f"Successfully parsed JSON with keys: {list(parsed.keys())}")
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {str(e)}")
        print(f"Raw response: {response}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing response: {str(e)}")
        print(f"Raw response: {response}")
        return None
