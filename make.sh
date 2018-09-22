#!/usr/bin/env bash



ASDF=`python3 <<END
import numpy

ASDF = numpy.get_include()
print (ASDF)
END`

export CFLAGS=-I$ASDF
python3 setup.py build_ext --inplace
