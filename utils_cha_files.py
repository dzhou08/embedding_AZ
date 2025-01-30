import pylangacq


def get_char_transcript(file_path, participant_type):
    """
    Extracts and returns the transcript of a specified participant type from a CHAT transcript file.

    https://dementia.talkbank.org/access/English/Pitt.html


    Args:
        file_path (str): The path to the CHAT file.
        participant_type (str): The type of participant whose transcript is to be extracted (e.g., 'CHI' for child, 'MOT' for mother).

    Returns:
        str: A string containing the concatenated words spoken by the specified participant type.

    Example:
        transcript = get_char_transcript('/path/to/chatfile.cha', 'CHI')
    """
    print(file_path)
    data = pylangacq.read_chat(file_path)

    # Access basic information
    print(data.participants())
    print(data.utterances())
    print(data.words())

    par_words = data.words(participants=participant_type)
    par_words_str = ' '.join(par_words)
    return par_words_str