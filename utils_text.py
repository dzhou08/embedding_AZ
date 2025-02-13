from rapidfuzz import fuzz
import os
import re

# text patterns to remove from the transcription
text_patterns = [
            "Tell me everything you see going on in this picture",
            "What's happening in this picture",
            "What's going on in this picture",
            "What's going on",
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
    print(f"return_string: {return_string}\n")
    return return_string
