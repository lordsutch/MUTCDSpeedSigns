#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how you can use dlib to make an object
#   detector for things like faces, pedestrians, and any other semi-rigid
#   object.  In particular, we go though the steps to train the kind of sliding
#   window object detector first published by Dalal and Triggs in 2005 in the
#   paper Histograms of Oriented Gradients for Human Detection.
#
# COMPILING THE DLIB PYTHON INTERFACE
#   Dlib comes with a compiled python interface for python 2.7 on MS Windows. If
#   you are using another python version or operating system then you need to
#   compile the dlib python interface before you can use this file.  To do this,
#   run compile_dlib_python_module.bat.  This should work on any operating
#   system so long as you have CMake and boost-python installed.
#   On Ubuntu, this can be done easily by running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install -U scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

from __future__ import print_function

from multiprocessing import Pool

import os
import sys
import glob
import warnings

import dlib
import skimage

NCPUS=6

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
TRAINING='speedlimits.svm'
DEBUG=False

detector = dlib.simple_object_detector(TRAINING)

if DEBUG:
    # We can look at the HOG filter we learned.
    win_det = dlib.image_window()
    win_det.set_image(detector)
    dlib.hit_enter_to_continue()

# from skimage.transform import rescale, pyramid_expand
from skimage import io
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import img_as_ubyte
    
def process_file(f):
    # Cheat a bit to improve upsampling speed here...
    # img = imread(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        img = img_as_ubyte(rgb2gray(imread(f)), 2.0)
    #print(dir(img))
    dets = detector(img, 1) # Upsampling improves detection IME
    if dets: # Found a sign
        print(f, len(dets))
    else:
        print('No signs in', f, file=sys.stderr)

if __name__ == '__main__':
    filenames = []
    for bit in sys.argv[1:]:
        filenames.extend( glob.glob(bit) )

    p = Pool(NCPUS) ## Number of parallel processes to run
    status = p.map(process_file, filenames)
