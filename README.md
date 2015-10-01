# MUTCDSpeedSigns
bits and pieces for detecting North American-style (MUTCD) speed limit signs in images

So far there isn't too much here:

`train_signmatch.py` - train the object detector using photos with signs
in them

`signmatch.py` - use the training data to find signs in other photos

`speedlimits.svm` - training data derived from a corpus of sign photos

To use this, you'll need to build
[`dlib`](https://github.com/davisking/dlib/) with Python support. You
will need Python 2.7, `cmake`, `libjpeg`, `boost-python`, and
`scikit-image` (aka `python-skimage`). The code will probably work with
Python 3.x as well, but is as-yet untested.

To build the Python `dlib` module,
`sh compile_dlib_python_module.bat` under the `python_examples`
directory. Then copy `dlib.so` into your Python path or this directory
so the Python scripts can use it.

If you want to train the classifier with your own signs, follow the
instructions in Ian Dees' Gist listed below through step 5. Then, use
the `train_signmatch.py` script (for some reason, the classifiers from
DLib are incompatible between C++ and Python).

# Speedups

On x86, you probably will want to enable AVX. This should be the
default in dlib after version 18.17; until then, you'll need to do
this manually by editing `compile_dlib_python_module.bat` to add
`-DUSE_AVX_INSTRUCTIONS=ON` to the first `cmake` command so it reads:

`cmake ../../tools/python -DUSE_AVX_INSTRUCTIONS=ON`

# Credits

All this depends on the awesome DLib project at
https://github.com/davisking/dlib/

Idea shamelessly stolen from Ian Dees' Gist at
https://gist.github.com/iandees/f773749c47d088705199

Photos used for training were mostly sourced from Wikimedia projects;
the rest are my own.
