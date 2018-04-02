import numpy as np
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import SparseVector

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

	corpusFreq = corpus.zipWithIndex()\
			.flatMap(lambda x: [(str(x[1])+' '+e, 1) for e in x[0]])\
			.reduceByKey(lambda x, y: x+y)\
			.map(lambda x: (int(x[0].split()[0]), [(x[0].split()[1], x[1])]))\
			.reduceByKey(lambda x, y: x+y)\
			.map(lambda x: x[1])

	match = corpusFreq.flatMap(lambda x: x)\
			.reduceByKey(lambda x, y: x+y)\
			.map(lambda x: x[0])\
			.zipWithIndex()\
			.collect()

	dictionary = dict()
	for word, id in match:
		dictionary[word] = id

	mat = RowMatrix(corpusFreq.map(lambda x: line2SparseVector(x, dictionary)))

	print corpus.collect()
	print corpusFreq.collect()
	print '---'
	print dictionary
	print '---'
	print np.array(mat.rows.map(lambda x: list(x.toArray())).collect())

	return mat


def line2SparseVector(line, dictionary):
	lineDict = dict()
	for word, freq in line:
		lineDict[dictionary[word]] = freq
	return SparseVector(len(dictionary), lineDict)

#data = [u'puigdemont puigdemont es detenido en alemania',u'spark funciona con spacy en alemania',u'me voy de vacaciones en un rato con puigdemont']

#r1 = sc.parallelize(data).map(lemmatize)
#r1 = sc.textFile('docs/').map(lemmatize)
#print r1.collect()
#h = HashingTF(32)
#t = h.transform(r1)
#print t.collect()
#mat = RowMatrix(t)
#print mat.rows.collect()
