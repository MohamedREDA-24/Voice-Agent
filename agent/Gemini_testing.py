import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="")

# Create the model
model = genai.GenerativeModel('gemini-2.5-flash')

# Generate content
response = model.generate_content("Explain quantum computing in simple terms")

print(response.text)