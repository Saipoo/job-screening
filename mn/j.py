import nltk
# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')  # Adding the missing resource
nltk.download('words')
print("NLTK data downloaded successfully")
print("NLTK data path:", nltk.data.path)
