from rapidfuzz import fuzz
import os
import re
import openai
import logging
import os

# text patterns to remove from the transcription
text_patterns = [
            "Tell me everything you see going on in this picture",
            "What's happening in this picture",
            "What's going on in this picture",
            "What's going on",
            "Anything else you see going on.",
            "And I want you to tell me everything that you see happening there.",
            "I'd like you to tell me all the things you see going on in the picture.",
            "Just tell me all",
            "Everything that's going on there",
            "I want you to tell me everything that you see happening in that picture",
            "everything that you see going on there",
            "Just look at the picture and tell me everything that you see"

    ]
threshold=70

def remove_text_patterns(original_text):
    #print(f"target: {target}")
    # split the target text into sentences via '.', '!', '?' or '\n'
    # need a regex to split on multiple delimiters of '.', '!', '?' or '\n'
    sentences = re.split(r'[.!?]', original_text)
    filtered = []
    for sentence in sentences:
        keep = True
        # look through all the text patterns
        for text in text_patterns:
            # compare the sentence to the text pattern
            if fuzz.ratio(sentence.strip().lower(), text.lower()) > threshold:
                keep = False
                break
        if keep:
            filtered.append(sentence)
    return_string = '. '.join(filtered).strip()
    #print(f"return_string: {return_string}\n")
    return return_string


def post_process_whisper_text(text, openai_api_key):
    system_prompt = """
    You are a experienced linguistic assistant working in Alzheimer's clinic.
    The following text is transcribed from Spontanous speech recordings, describing the Boston cookie theft picture.
    There should be two speakers inside the recorded transcripts.
    Please identify sentences spoken by the speaker who prompts the other user to describe the picture and remove those sentences.
    Keep all other words, and sentences, even the meaningless utterances like uh, hmm, etc.
    """
    openai.api_key = openai_api_key
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    clean_text = response.choices[0].message.content
    print(f"text: {text}\n\n")
    print(f"clean_text: {clean_text}")
    return clean_text
