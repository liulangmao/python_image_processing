from pylab import imshow
import numpy as np
import mahotas

wally = mahotas.imread('waldo.jpg')
imshow(wally)

wfloat = wally.astype(float)
r,g,b = wfloat.transpose((2,0,1)) # split into rgb channels, better to use floats
w = wfloat.mean(2)	# w is the white channel

pattern = np.ones((24,16),float)
for i in xrange(2):
	pattern[i::4] = -1 # build a pattern of +1,+1,-1,-1 on vertical axis -> Wally's shirt.

v = mahotas.convolve(r-w, pattern) # convolve red-white, will give strong response where shirt
mask = (v == v.max())
mask = mahotas.dilate(mask,np.ones((48,24))) # look for max and dilate it to make it visible

wally -= .8*wally * ~mask[:,:,None]
imshow(wally)# imshow(wally)