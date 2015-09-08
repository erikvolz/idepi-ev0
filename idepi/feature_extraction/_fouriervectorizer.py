
from numpy import zeros
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from idepi.constants import GAPS
from idepi.labeledmsa import LabeledMSA

import csv


__all__ = ['FourierVectorizer']
__KIDERA_FEATURES__ = 10


class FourierVectorizer(BaseEstimator, TransformerMixin):
	def __init__(self, waveNum = 10, name=''):
		self.name = name
		self.waveNum = waveNum
		self.feature_names_ = []
		self.vocabulary_ = {}
		
		# load chemical properties table for each AA; store in dict keyed by AA
		f_kidera = open('../data/kideraTable3.1.tsv', 'r') 
		r = csv.reader( f_kidera, delimiter='\t')
		self.kidera = dict()
		for rr in r:
			self.kidera[rr[0]] = [ float(x) for x in rr[1:] ]
		f_kidera.close()
		
		self.feature_names_ = [ 'wn%i.property%i.cosine' % (wn, prop) for prop in range(1,__KIDERA_FEATURES__+1) for wn in range(1,self.waveNum+1)  ] + 
		  [ 'wn%i.property%i.sine' % (wn, prop) for prop in range(1,__KIDERA_FEATURES__+1) for wn in range(2,self.waveNum+1)  ]
	#
	
	def fit(self, alignment):
		if not isinstance(alignment, LabeledMSA):
			raise ValueError("MSAVectorizers require a LabeledMSA")
		#
		return self
	#
	
	def transform(self, aa_alignment ):
		# TODO ensure AA alphabet
		#if alignment.type!='_amino':
		#	raise ...
		
		# output features
		data = zeros((len(alignment), self.waveNum * __KIDERA_FEATURES__), dtype=int)
		
		for i, seq in enumerate(aa_alignment):
			# do NOT convert to encoder coords
			cols = []
			ltrs = []
			# remove gaps
			for col, ltr in enumerate(str(seq.seq)):
				if ltr not in GAPS:
					cols.append(col)
					ltrs.append(ltr)
			#
			# tabulate chem properties
			propmat = zeros((__KIDERA_FEATURES__, len(ltrs)))
			for ic, c in enumerate(ltrs): #for each character
				for l in range(__KIDERA_FEATURES__): # for each property
					propmat[l,ic] = self.kidera[c][l]
			#
			
			coefA_mat =  zeros((__KIDERA_FEATURES__, self.waveNum))
			coefB_mat =  zeros((__KIDERA_FEATURES__, self.waveNum))
			for l in range(__KIDERA_FEATURES__):
				pmi = propmat[l,:]
				pmi = np.pmi[-isnan(pmi)]
				dft = np.fft(pmi)
				coefA_mat[l,:] = dft.real[:self.waveNum]
				coefB_mat[l,:] = dft.imag[1:self.waveNum]
			#
			# note unravels matrices by row
			data[i,:] = np.concatenate(( np.ravel(coefA_mat) , np.ravel(coefB_mat[:,1:]) ))
		#
		return data
#
