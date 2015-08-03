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
from functools import partial

import multiprocessing
import os
import sys
import glob
import warnings

import dlib
import skimage

import argparse
import itertools
import traceback
import uuid

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
TRAINING='speedlimits.svm'

detector = dlib.simple_object_detector(TRAINING)

# from skimage.transform import rescale, pyramid_expand
from skimage import io
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import img_as_ubyte
    
def process_file(f, verbose=False, link=None):
    if not os.path.exists(f):
        # Skip if not readable
        return []
    # Cheat a bit to improve upsampling speed here... grayscale is faster
    img = imread(f)
    img = rgb2gray(img)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        img = img_as_ubyte(img)
    #print(dir(img))
    dets = detector(img, 1) # Upsampling improves detection IME
    if dets: # We found a sign (or more!)
        if verbose:
            print('Found', len(dets), 'sign(s) in', f, 
                  [str(x) for x in dets], file=sys.stderr)
        if link:
            if not os.path.exists(link) and not os.path.isdir(link):
                os.makedirs(link, 0o700)
            fnamebase = os.path.basename(f)
            linkname = os.path.join(link, fnamebase)
            # Add a random prefix to avoid clashes, if needed
            while os.path.exists(linkname):
                linkname = os.path.join(link, str(uuid.uuid4())+'-'+fnamebase)
            try:
                os.link(f, linkname)
            except OSError as e:
                try:
                    os.symlink(f, linkname)
                except OSError as e:
                    print('Unable to link',f,'to',link, file=sys.stderr)
                    trackback.print_exc()
        return dets
    if verbose:
        print('No signs in', f, file=sys.stderr)
    return []

if __name__ == '__main__':
    NCPUS = multiprocessing.cpu_count()/2
    parser = argparse.ArgumentParser(description='detect images matching pattern')
    parser.add_argument('-s', '--show-filter', dest='showfilter',
                        default=False, action='store_true',
                        help='show the filter that will be applied')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False,
                        action='store_true', help='include extra output to stderr')
    parser.add_argument('-l', '--link', dest='link', default=None,
                        action='store', type=str, metavar='DIR',
                        help='link found files to DIR')
    parser.add_argument('-j', '--parallel', dest='ncpus', default=NCPUS,
                        action='store', type=int, metavar='JOBS',
                        help='how many parallel tasks to run')
    parser.add_argument('files', metavar='FILE', type=str, nargs="*",
                        help='files to scan')
    args = parser.parse_args()

    if args.showfilter:
        # We can look at the HOG filter we learned.
        win_det = dlib.image_window()
        win_det.set_image(detector)
        dlib.hit_enter_to_continue()
        sys.exit(0)
    
    filenames = []
    for bit in args.files:
        filenames.extend( glob.glob(bit) )

    partial_process = partial(process_file, verbose=args.verbose, link=args.link)
        
    p = Pool(args.ncpus) ## Number of parallel processes to run
    status = p.map(partial_process, filenames)
    print("\n".join(itertools.compress(filenames, status)))
