import os
import dotenv
from groq import Groq

dotenv.load_dotenv()
client = Groq()
models = client.models.list()
for m in models.data:
    print(m.id)
