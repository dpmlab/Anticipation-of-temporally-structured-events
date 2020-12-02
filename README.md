# Code for "Anticipation of temporally structured events in the brain"

The code in this repository can be used to reproduce the results of (Lee, Aly, and Baldadssano. (2020). "Anticipation of temporally structured events in the brain")[https://doi.org/10.1101/2020.10.14.338145]

Data from ["Learning Naturalistic Temporal Structure in the Posterior Medial Network"](https://openneuro.org/datasets/ds001545/versions/1.1.1) was preprocessed using FSL as specified in preproc01.fsf. All the results reported in the manuscript can be reproduced by running main.py. Note that running all the bootstraps will be take substantial time (days), and you may want to modify these loops to take advantage of parallel processing resources available on your system.

This code was originally run with:
Python version: 3.6.1
Anaconda version: 4.5.5
Backend/analysis dependencies: numpy, brainiak, scipy, nibabel



MIT License

Copyright (c) 2020, Caroline Lee and Christopher Baldassano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.