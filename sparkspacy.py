from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import HashingTF

#sc = SparkContext()
#sc.setLogLevel('WARN')

class SpacyLoader(object):

	_spacys = {}

	@classmethod
	def get(cls, lang):
		if lang not in cls._spacys:
			import spacy
			cls._spacys[lang] = spacy.load(lang, disable=['parser', 'tagger', 'ner'])
		return cls._spacys[lang]


def lemmatize(doc):
	nlp = SpacyLoader.get('es')
	return [token.lemma_ for token in nlp(doc) if token.is_stop == False]
	

def process(sc, docDir):
	corpus = sc.textFile(docDir).map(lemmatize)
	print corpus.collect()
	h = HashingTF(32)
	t = h.transform(corpus)
	return RowMatrix(t)

#data = [u'puigdemont puigdemont es detenido en alemania',u'spark funciona con spacy en alemania',u'me voy de vacaciones en un rato con puigdemont']

#r1 = sc.parallelize(data).map(lemmatize)
#r1 = sc.textFile('docs/').map(lemmatize)
#print r1.collect()
#h = HashingTF(32)
#t = h.transform(r1)
#print t.collect()
#mat = RowMatrix(t)
#print mat.rows.collect()
