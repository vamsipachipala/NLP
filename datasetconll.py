import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.metrics import accuracy

nltk.download('state_union')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('conll2003')

def get_Entities(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    i = 1
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entText = (' '.join(c[0] for c in chunk))
            print('{:5}'.format(i), entText.ljust(50, '-') + "{:<30}".format(chunk.label()))
            i = i + 1

# Load the CoNLL 2002 dataset for testing
test_sentences = open(r"C:\Users\ACER\Documents\NLP\Project\train.txt").read()

s = "-"
print(s.center(80, '-'))
s = "NLTK Named Entity Recognition"
print(s.center(80, '-'))

# Extract entities from the CoNLL 2002 dataset
predicted_entities = []
true_entities = []

for sentence in test_sentences:
    words, true_labels, _ = zip(*sentence)
    text = ' '.join(words)
    
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    iob_tags = tree2conlltags(chunked)
    _, predicted_labels, _ = zip(*iob_tags)
    
    true_entities.extend(true_labels)
    predicted_entities.extend(predicted_labels)

# Calculate accuracy
accuracy_score = accuracy(true_entities, predicted_entities)
print("Accuracy:", accuracy_score)

s = "-"
print(s.center(80, '-'))
