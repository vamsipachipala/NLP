import nltk
from nltk.tag import StanfordNERTagger
nltk.download('state_union')
nltk.download('punkt')
from nltk.metrics.scores import accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from prettytable import PrettyTable


# Path to the Stanford NER model and jar file
stanford_ner_model = '/Library/stanford-ner-2020-11-17/classifiers/english.muc.7class.distsim.crf.ser.gz'
stanford_ner_jar = '/Library/stanford-ner-2020-11-17/stanford-ner.jar'

# Initialize Stanford NER Tagger
st = StanfordNERTagger(stanford_ner_model, stanford_ner_jar, encoding='utf-8')

categories = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'PERCENT', 'TIME', 'MONEY']

def calculate_metrics(reference_annotations, predicted_annotations):
    reference_labels = [label for _, label in reference_annotations]
    predicted_labels = [label for _, label in predicted_annotations]

    precision = precision_score(reference_labels, predicted_labels, average='weighted')
    recall = recall_score(reference_labels, predicted_labels, average='weighted')
    f1 = f1_score(reference_labels, predicted_labels, average='weighted')
    return precision, recall, f1

def get_entities_table(text):
    # Tokenize the text
    tokenized_text = nltk.word_tokenize(text)

    # Run Stanford NER on the tokenized text
    stanford_entities = st.tag(tokenized_text)
    unique_words = set()  # To store unique words


    table = PrettyTable()
    table.field_names = ["Index", "Word", "Entity"]

    i = 1
    for word, tag in stanford_entities:
        if tag != 'O'and word not in unique_words:
            unique_words.add(word)
            table.add_row([i, word, tag])
            i += 1

    print(table)

# Load your text
text = open(r"/Users/vamsipachipala/Documents/NLP/NLP/test.txt", 'r', encoding='utf-8').read()

s = "-"
print(s.center(80, '-'))
s = "Stanford class 7 Named Entity Recognition"
print(s.center(80, '-'))
get_entities_table(text)
s = "-"
print(s.center(80, '-'))

raw_annotations = open(r"/Users/vamsipachipala/Documents/NLP/NLP/train.txt").read()
split_annotations = raw_annotations.split()

# Group NE data into tuples
def group(lst, n):
  for i in range(0, len(lst), n):
      val = lst[i:i+n]
      if len(val) == n:
            yield tuple(val)

reference_annotations = list(group(split_annotations, 2))

pure_tokens = split_annotations[::2]

stanford_prediction = st.tag(pure_tokens)
stanford_accuracy = accuracy(reference_annotations, stanford_prediction)
precision, recall, f1 = calculate_metrics(reference_annotations, stanford_prediction)

# Calculate precision, recall, and F1-score
print(f"Accuracy:{stanford_accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

