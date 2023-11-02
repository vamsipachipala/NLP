import nltk
from nltk.tag import StanfordNERTagger
nltk.download('state_union')
nltk.download('punkt')
from nltk.metrics.scores import accuracy


# Path to the Stanford NER model and jar file
stanford_ner_model = 'c:\Program Files\stanford-ner-2020-11-17\classifiers\english.muc.7class.distsim.crf.ser.gz'
stanford_ner_jar = 'c:\Program Files\stanford-ner-2020-11-17\stanford-ner.jar'

# Initialize Stanford NER Tagger
st = StanfordNERTagger(stanford_ner_model, stanford_ner_jar, encoding='utf-8')

categories = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'PERCENT', 'TIME', 'MONEY']

def calculate_metrics(reference_annotations, stanford_prediction, category):
    true_positives = sum(1 for r, p in zip(reference_annotations, stanford_prediction) if r == category and p == category)
    false_positives = sum(1 for r, p in zip(reference_annotations, stanford_prediction) if r != category and p == category)
    false_negatives = sum(1 for r, p in zip(reference_annotations, stanford_prediction) if r == category and p != category)

    if true_positives == 0 and false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives == 0 and false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def get_Entities(text):
    # Tokenize the text
    tokenized_text = nltk.word_tokenize(text)

    # Run Stanford NER on the tokenized text
    stanford_entities = st.tag(tokenized_text)

    i = 1
    for word, tag in stanford_entities:
        if tag != 'O':
            print('{:5}'.format(i), word.ljust(50, '-') + "{:<30}".format(tag))
            i = i + 1

# Load your text
text = open(r"C:\Users\ACER\Documents\NLP\Project\NLP\socialmediatext.txt", 'r', encoding='utf-8').read()

s = "-"
print(s.center(80, '-'))
s = "Stanford class 7 Named Entity Recognition"
print(s.center(80, '-'))
get_Entities(text)
s = "-"
print(s.center(80, '-'))

raw_annotations = open(r"C:\Users\ACER\Documents\NLP\Project\NLP\annotatedtext.txt").read()
split_annotations = raw_annotations.split()

# Amend class annotations to reflect Stanford's NERTagger
for n,i in enumerate(split_annotations):
	if i == "I-PER":
		split_annotations[n] = "PERSON"
	if i == "I-ORG":
		split_annotations[n] = "ORGANIZATION"
	if i == "I-LOC":
		split_annotations[n] = "LOCATION"
	if i == "I-DATE":
		split_annotations[n] = "DATE"
	if i == "I-PERCENT":
		split_annotations[n] = "PERCENT"
	if i == "I-TIME":
		split_annotations[n] = "TIME"
	if i == "I-MONEY":
		split_annotations[n] = "MONEY"

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

# Calculate precision, recall, and F1-score
print(f"Accuracy:{stanford_accuracy}")
for category in categories:
    precision, recall, f1_score = calculate_metrics(reference_annotations, stanford_prediction, category)
    print(f"Category: {category}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print("-" * 80)