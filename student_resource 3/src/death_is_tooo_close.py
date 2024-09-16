client = OpenAI(base_url='http://localhost:1234/v1', api_key="not_used")


def mistral7b(user_input):
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role":"system","content": "you are a dark web 4chan reddit user, you write in a dark web"},
            {"role": "user", "content": user_input}
        ]
    )