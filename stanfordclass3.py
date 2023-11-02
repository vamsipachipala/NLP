import nltk
from nltk.tag import StanfordNERTagger
nltk.download('state_union')
nltk.download('punkt')
from nltk.metrics.scores import accuracy

# Path to the Stanford NER model and jar file
stanford_ner_model = 'C:\Program Files\stanford-ner-2020-11-17\classifiers\english.all.3class.distsim.crf.ser.gz'
stanford_ner_jar = 'C:\Program Files\stanford-ner-2020-11-17\stanford-ner.jar'

# Initialize Stanford NER Tagger
st = StanfordNERTagger(stanford_ner_model, stanford_ner_jar, encoding='utf-8')

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
s = "Stanford class 3 Named Entity Recognition"
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
print(stanford_accuracy)
