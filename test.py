from nltk import ne_chunk, pos_tag, word_tokenize
import nltk
import numpy
nltk.download('state_union')
from nltk.corpus import state_union
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.metrics.scores import accuracy

def get_Entities(text):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        i = 1
        for chunk in chunked:
              if hasattr(chunk, 'label'):
                 entText= (' '.join(c[0] for c in chunk))
                 
                 print('{:5}'.format(i), entText.ljust(50, '-')+"{:<30}".format(chunk.label()))
                 i = i + 1
                
Media = state_union.raw(r"C:\Users\ACER\Documents\NLP\Project\socialmediatext.txt") 


s="-"
print(s.center(80, '-'))
s="NLTK Named Entity Recognition"
print(s.center(80, '-'))
get_Entities(Media)
s="-"
print(s.center(80, '-'))

raw_annotations = open(r"C:\Users\ACER\Documents\NLP\Project\annotatedtext.txt").read()
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

tagged_words = nltk.pos_tag(pure_tokens)
nltk_unformatted_prediction = nltk.ne_chunk(tagged_words)

multiline_string = nltk.chunk.tree2conllstr(nltk_unformatted_prediction)
listed_pos_and_ne = multiline_string.split()

# Delete pos tags and rename
del listed_pos_and_ne[1::3]
listed_ne = listed_pos_and_ne

# Amend class annotations for consistency with reference_annotations
for n,i in enumerate(listed_ne):
	if i == "B-PERSON":
		listed_ne[n] = "PERSON"
	if i == "I-PERSON":
		listed_ne[n] = "PERSON"    
	if i == "B-ORGANIZATION":
		listed_ne[n] = "ORGANIZATION"
	if i == "I-ORGANIZATION":
		listed_ne[n] = "ORGANIZATION"
	if i == "B-LOCATION":
		listed_ne[n] = "LOCATION"
	if i == "I-LOCATION":
		listed_ne[n] = "LOCATION"
	if i == "B-GPE":
		listed_ne[n] = "LOCATION"
	if i == "I-GPE":
		listed_ne[n] = "LOCATION"

# Group prediction into tuples
nltk_formatted_prediction = list(group(listed_ne, 2))

nltk_accuracy = accuracy(reference_annotations, nltk_formatted_prediction)
print(nltk_accuracy)