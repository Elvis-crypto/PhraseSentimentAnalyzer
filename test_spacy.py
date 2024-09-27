import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
print("Model loaded successfully.")
