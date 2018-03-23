import numpy as np
from pyspark import SparkContext
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
		
		# v x s^(-1)
		self.vsinv = self.v.toArray().dot(np.diag(1./self.s))

	def transform(self, q):
		'''Transforms a query into SVD space.
		q2 = q x v x s^(-1)

		Args:
			q: list of terms
		'''
	
		return q.dot(self.vsinv)
		

	def index(self, q):
		'''
		'''

		w = RowMatrix(sc.parallelize(self.transform(q)))
		self.u = RowMatrix(self.u.rows.union(w.rows))


	def retrieve(self, nResuts):
		'''
		'''

		pass

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
lsi.index(np.array([[2,0,0,0],[0,0,0,6],[1,2,0,0],[6,0,0,6]]))
print '----- New U Matrix -----'
print lsi.u.rows.collect()
