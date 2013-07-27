import numpy as np
import scipy
from scipy import ndimage
import pylab
import mahotas
import pymorph

dna = mahotas.imread('dna.png')

############ STEP ONE -- Initial Image Processing

# Shows as heat map
pylab.imshow(dna)
pylab.show()

# Gray-scale
pylab.imshow(dna)
pylab.gray()
pylab.show()

print dna.shape # (1024, 1344) -- 1024 px by 1344 px
print dna.dtype # uint8 	   -- unsigned 8-bit int
print dna.max() # 252		   -- max val = 252
print dna.min() # 0			   -- min val = 0

# What happens when you divide by 2? -- It's the same!
pylab.imshow(dna // 2)
pylab.show()

# Threshold the image and count the number of objects
T = mahotas.thresholding.otsu(dna)
pylab.imshow(dna > T)
pylab.show()
# Results in numpy array of bools -> b/w image

# Smooth image using gaussian filter
dnaf = ndimage.gaussian_filter(dna, 8) # -- Image and stdev of image
T = mahotas.thresholding.otsu(dnaf)
pylab.imshow(dnaf > T)
pylab.show()

# Deal with merged/touching nuclei
labeled,nr_objects = ndimage.label(dnaf > T)
print nr_objects # prints 18
pylab.imshow(labeled)
pylab.jet()	# resets to jet from gray-scale
pylab.show()

############ STEP TWO -- Segmenting Image/Finding seeds
# Smooth image->find regional maxima->use maxima as seeds for watershed

# First try:
dnaf = ndimage.gaussian_filter(dna, 8)
rmax = pymorph.regmax(dnaf)
pylab.imshow(pymorph.overlay(dna, rmax)) # Overlay returns a color image with gray level component in first argument, second arg is red
pylab.show()

# Second try - Increase sigma:
dnaf = ndimage.gaussian_filter(dna, 16)
rmax = pymorph.regmax(dnaf)
pylab.imshow(pymorph.overlay(dna, rmax))

seeds,nr_nuclei = ndimage.label(rmax)
print nr_nuclei # prints 22

# Watershed to distance transform of threshold
T = mahotas.thresholding.otsu(dnaf)
dist = ndimage.distance_transform_edt(dnaf > T)
dist = dist.max() - dist
dist -= dist.min()
dist = dist/float(dist.ptp()) * 255
dist = dist.astype(np.uint8)
pylab.imshow(dist)
pylab.show()

nuclei = pymorph.cwatershed(dist, seeds)
pylab.imshow(nuclei)
pylab.show()

# Extend segmentation to whole plane using generalized voronoi - each px -> nearest nucleus
whole = mahotas.segmentation.gvoronoi(nuclei)
pylab.imshow(whole)
pylab.show()

# Quality control - remove cells whose nucleus touches border
borders = np.zeros(nuclei.shape, np.bool)	# builds an array of zeroes with nuclei shape and np.bool
borders[ 0,:] = 1
borders[-1,:] = 1
borders[:, 0] = 1
borders[:,-1] = 1 							# Sets borders of array to 1/True
at_border = np.unique(nuclei[borders])		# nuclei[borders] gets True borders, unique returns only unique values
for obj in at_border:
    whole[whole == obj] = 0					# Iterate over border objects and everywhere whole takes that value, set to 0
pylab.imshow(whole)
pylab.show()