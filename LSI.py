import numpy as np
from pyspark import SparkContext
from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, RowMatrix


sc = SparkContext()
sc.setLogLevel('WARN')

class LSI(object):
	'''Manages LSI factorization

	Attributes:
		u: RowMatrix (document x features)
		v: DenseMatrix (term x features)
		s: list of eigenvalues^2
		vsinv: vs^(-1)

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

		data = RowMatrix(sc.textFile(dataPath).map(lambda line: [x for x in line.split()]))
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


	def retrieve(self, q, nResults):
		'''
		'''

		qp = self.transform(q)
		

	def getSimilarities(self, w):
		'''
		'''

		wspace = self.transform(w)
		print '---'
		print wspace
		distances = self._getCosineDistances(wspace)
				

	def _getCosineDistances(self, w):
		'''
		'''

		wt = w.transpose()
		wt = DenseMatrix(wt.shape[0], wt.shape[1], wt.flatten())
		num = self.u.multiply(wt)

		wnorm = np.linalg.norm(w, axis=1)
		wnorm = np.insert(np.asmatrix(wnorm), 1, 0, axis=0)
		print '---'
		print self.unorm	
		print '---'
		print self.unorm		
		print '---'
		print wnorm
		print '---'
		den = np.dot(self.unorm, wnorm)
		print den
		distances = num.rows.zipWithIndex().map(lambda x: _getDivisionByNorm(x, den))
		print '---'
		print distances.collect()
		return distances


def _getDivisionByNorm(num, den):
	'''
	'''

	return np.divide(num[0], den[num[1],:]).flatten()


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
print '----- similarities -----'
print lsi.getSimilarities(np.array([[2,0,0,0],[0,0,0,6]]))
#lsi.index(np.array([[2,0,0,0],[0,0,0,6],[1,2,0,0],[6,0,0,6]]))
#print '----- New U Matrix -----'
#print lsi.u.rows.collect()
