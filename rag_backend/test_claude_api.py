import os
import anthropic
from anthropic import Anthropic

def test_anthropic_api():
    # Print version information
    print(f"Anthropic SDK version: {anthropic.__version__}")
    
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Check if API key has the right format (starts with sk-ant-)
    if not api_key or not api_key.startswith("sk-ant-"):
        print(f"Warning: API key has invalid format or not found. Should start with 'sk-ant-'")
        print(f"API key: {api_key[:10]}...{api_key[-4:]}" if api_key else "None")
        return
    
    # Print API key pattern for debugging (first few and last few characters)
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
    print(f"API key pattern: {masked_key}")
    
    # Initialize the client with minimal parameters
    try:
        client = Anthropic(api_key=api_key)
        print("Successfully created Anthropic client")
        
        # Make a simple API call
        try:
            print("Attempting to send a request to Claude...")
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Say hello"}
                ]
            )
            print(f"Success! Claude responded: {response.content[0].text}")
        except Exception as e:
            print(f"Error making API call: {e}")
            
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}")

if __name__ == "__main__":
    test_anthropic_api()