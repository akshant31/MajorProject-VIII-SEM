import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim
import pyLDAvis.gensim
import pyLDAvis

stemmer = PorterStemmer()



#def stem_tokens(tokens, stemmer):
#    stemmed = []
#    for item in tokens:
#        stemmed.append(stemmer.stem(item))
#    return stemmed


def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)  # Removing numbers and punctuation
    text = re.sub(" +", " ", text)  # Removing extra white space
    text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b", " ", text)  # Removing very long words above 10 characters
    text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b", " ", text)  # Removing single characters (e.g k, K)
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text)
    tokens = nltk.word_tokenize(text.strip())
    tokens = nltk.pos_tag(tokens)
    # Uncomment next line to use stemmer
    # tokens = stem_tokens(tokens, stemmer)
    return tokens


stopset = stopwords.words('english')
freq_words = ['http', 'https', 'amp', 'com', 'co', 'th', 'ji', 'all', 'sampla', 'pQ', 'lot', 'sir', 'pxr', 'ncbn', 'plz', 'qnI', 'way', 'sEkq', 'iAvvaPhp', 'zYGLz', 'tHMJdha']
for i in freq_words:
    stopset.append(i)


def analyze(fileObj, Uname):
    text_corpus = []
    for doc in fileObj:
        temp_doc = tokenize(doc.strip())
        current_doc = []
        for word in range(len(temp_doc)):
            if temp_doc[word][0] not in stopset and temp_doc[word][1] == 'NN':
                current_doc.append(temp_doc[word][0])

        text_corpus.append(current_doc)

    dictionary = corpora.Dictionary(text_corpus)
    corpus = [dictionary.doc2bow(text) for text in text_corpus]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=60)
    print ('Topics for ', Uname, '\n')
    for topics in ldamodel.print_topics(num_topics=5, num_words=7):
        print (topics, "\n")

    vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.display(vis_data)
    pyLDAvis.save_html(vis_data, Uname+'.html')


manojsinha = open('manojsinha.txt')
analyze(manojsinha, 'manoj sinha')

others = open('others.txt')
analyze(others, 'others')


