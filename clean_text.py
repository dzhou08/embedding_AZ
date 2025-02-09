from rapidfuzz import fuzz
import os
import re

# text patterns to remove from the transcription
text_patterns = [
            "Tell me everything you see going on in this picture",
            "What's happening in this picture",
            "What's going on in this picture",
            "What's going on"
    ]


# read text from a file
transcription_folder = "ADReSSo_2020_WAV_transcription/train"

def remove_similar_sentences(text, target, threshold=70):
    #print(f"target: {target}")
    # split the target text into sentences via '.', '!', '?' or '\n'
    # need a regex to split on multiple delimiters of '.', '!', '?' or '\n'
    sentences = re.split(r'[.!?]', target)
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

# loop in the folder and read the text from the each file
for file in os.listdir(transcription_folder):
    with open(os.path.join(transcription_folder, file), "r") as f:
        target = f.read()
        result = remove_similar_sentences(text_patterns, target)
        print(result)
