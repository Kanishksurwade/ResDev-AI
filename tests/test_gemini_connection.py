import os

from google import genai


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY was not found.")
        return

    print("Gemini API key found.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: GEMINI_OK",
    )

    print("Gemini response:")
    print(response.text)


if __name__ == "__main__":
    main()