import config
from google import genai

client = genai.Client(api_key=config.GOOGLE_API_KEY)

prompt = "Hello! Who are you?"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)