import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

class Line():
	def __init__(self,n):
		self.n=n
		self.detected =False

		#Polynomial coefficients of the lines

		self.A=[]
		self.B=[]
		self.C=[]

		#Running average of coefficients

		self.A_avg=0.
		self.B_avg=0.
		self.C_avg=0.

	def obtain_fit(self):
		return (self.A_avg,self.B_avg,self.C_avg)	


	def update_fit(self,fit_coeffs):

		"""Obtain the fit coefficients from the latest frame and apply over each of 2nd polynomial coefficients
		for the purpose of smoothing
		"""

		full_Q= len(self.A) >= self.n


		#Append line fit coefficients

		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		if full_Q:
			_=self.A.pop(0)
			_=self.B.pop(0)
			_=self.C.pop(0)


		# Compute the average  of the polynomial coefficients 

		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)


		return (self.A_avg,self.B_avg,self.C_avg)

