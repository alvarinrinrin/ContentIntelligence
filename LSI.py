import numpy as np
from pyspark import SparkContext
from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, RowMatrix
import sparkspacy as ss

sc = SparkContext()
sc.setLogLevel('WARN')

class LSI(object):
	'''Manages LSI factorization

	Attributes:
		u: RowMatrix (document x features)
		v: DenseMatrix (term x features)
		s: numpy array of eigenvalues^2 (features)
		vsinv: numpy array vs^(-1)
		unorm: numpy array applied norm to u

	Methods:
		compute
		retrieve
		index
	'''

	def __init__(self,  uPath=None, vPath=None, sPath=None):
		'''
		'''

		if uPath <> None and vPath <> None and sPath <> None:
			pass
		else:
			self.u = None
			self.v = None
			self.s = None
			self.vsinv = None


	def compute(self, dataPath, nTopics):
		'''Computes SVD factorization

		Args:
			dataset: RowMatrix containing data
			nFeatures: number of resulting features
	
		Returns:
			SVD object with 
		'''

		#data = RowMatrix(sc.textFile(dataPath).map(lambda line: [x for x in line.split()]))
		data = ss.process(sc, 'docs/')
		print data.rows.collect()
		svd = data.computeSVD(nTopics, computeU=True)

		self.u = svd.U
		self.v = svd.V
		self.s = svd.s
		
		# V x S^(-1)
		self.vsinv = self.v.toArray().dot(np.diag(1./self.s))

		# norm(U) with all-0 column to multiply	
		unorm = np.array(self.u.rows.map(lambda x: x.norm(2)).collect())
		self.unorm = np.insert(np.asmatrix(unorm).transpose(), 1, 0, axis=1)


	def transform(self, d):
		'''Transforms a document into SVD space.
		d' = d x v x s^(-1)

		Args:
			d: numpy array with freqs. --> each position a term
		'''
	
		return d.dot(self.vsinv)
		

	def index(self, d):
		'''Indexes a document into the system (appending to U matrix)

		Args:
			d: numpy array with freqs. --> each position a term
		'''

		w = RowMatrix(sc.parallelize(self.transform(d)))
		self.u = RowMatrix(self.u.rows.union(w.rows))


	def retrieve(self, w, nResults):
		'''
		'''

		print '----- query de entrada:'
		print w
		print '-----'
		wspace = self.transform(w)
		print '----- query transformada:'
		print wspace
		print '-----'
		distances = self._getCosineDistances(wspace)
		print distances.flatMap(lambda x: [(i, [x[0], y]) for i, y in enumerate(x[1])])\
				.sortBy(lambda x: 2 * x[0] + x[1][1], ascending=False)\
				.reduceByKey(lambda x, y: x + y)\
				.map(lambda x: _groupInPairs(x[1])[:nResults])\
				.collect()
		
				

	def _getCosineDistances(self, w):
		'''
		'''

		wt = w.transpose()
		print '----- query transpuesta:'
		print wt
		print '-----'
		print '----- query transpuesta flatten:'
		print wt.flatten()
		print wt.shape
		print '-----'
		wt = DenseMatrix(wt.shape[0], wt.shape[1], wt.flatten(), isTransposed=True)
		print '----- query transpuesta y dense matrix:'
		print wt
		print '-----'
		print '----- U:'
		print self.u.rows.collect()
		print '-----'
		num = self.u.multiply(wt)
		print '----- num;'
		print num.rows.collect()
		print '-----'

		wnorm = np.linalg.norm(w, axis=1)
		wnorm = np.insert(np.asmatrix(wnorm), 1, 0, axis=0)
		print '----- U norm:'
		print self.unorm	
		print '-----'
		print '----- W norm:'
		print wnorm
		print '-----'
		den = np.dot(self.unorm, wnorm)
		print '----- den:'
		print den
		print '-----'
		distances = num.rows.zipWithIndex().map(lambda x: _getDivisionByNorm(x, den))
		print '----- distances:'
		print distances
		print '-----'
		return distances


def _getDivisionByNorm(num, den):
	'''
	'''

	#print '+++'
	#print num[0].toArray()
	#print np.squeeze(np.asarray(den[num[1],:]))
	#print '+++'
	#print '---'
	#print '---'
	return (num[1], np.divide(num[0].toArray(), np.squeeze(np.asarray(den[num[1],:][0]))))


def _groupInPairs(l):
	'''
	'''

	out = []
	first = True
	for i, e in enumerate(l):
		if i % 2 == 0:
			ePrev = e
		else:
			out.append([ePrev, e])
	return out
		


lsi = LSI()
lsi.compute('dataset.csv', 2)
print '----- U matrix -----'
print lsi.u.rows.collect()
print '----- S vector -----'
print lsi.s
print '----- V matrix -----'
print lsi.v
print '----- VS^(-1) matrix -----'
print lsi.vsinv
print '----- |U| vector -----'
print lsi.unorm
similarities = lsi.retrieve(np.array([[1,0,1,0,0,0,0]]), 2)
print '----- similarities:'
print similarities.collect()
#lsi.index(np.array([[2,0,0,0],[0,0,0,6],[1,2,0,0],[6,0,0,6]]))
#print '----- New U Matrix -----'
#print lsi.u.rows.collect()
