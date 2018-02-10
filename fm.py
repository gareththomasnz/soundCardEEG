""" FM demodulator for OpenEEG project. 

(C) 2003 Oren Tirosh <oren@hishome.net>
Released under the GNU General Public License

Requires:

Python 2.2 or higher 
  http://www.python.org

Numerical Python extension 
  http://www.pfdubois.com/numpy/

"""

from Numeric import *
from FFT import real_fft as fft, inverse_real_fft as ifft
import RNG


hilbcoeffs = 161    # determines accuracy of hilbert approximation
diffcoeffs = 74     # determines accuracy of derivative approximation
lpcoeffs = 1023     # size of low pass filter (total overkill)


def memoize(func):
    """ Speed up expensive functions by caching results """
    memo = {}

    def memoized(*args, **kw):
        if args not in memo:
            memo[args] = func(*args, **kw)
        return memo[args]

    return memoized


def trim(x, targetlen):
    """ Trim leftovers from convolution """
    trim = int((len(x)-targetlen)/2)
    return x[trim:][:targetlen]


def hilb(x):
    """ Apply Hilbert transform to x """
    y = convolve(x, make_hilb(hilbcoeffs))
    return trim(y, len(x))


def diff(x):
    """ Differentiator - approximation of derivative """
    y = convolve(x, make_diff(diffcoeffs))
    return trim(y, len(x))


def lowpass(x, bwfraction):
    """ Low pass filter. bwfraction is between 0 and 1 """
    y = convolve(x, make_filter(0, bwfraction, lpcoeffs))
    return trim(y, len(x))


def hanning_window(N):
    """ Compute the Von Hann window for FFTs and other filters """
    return 0.5 - 0.5*cos(2*pi*arange(N)/N)

hanning_window = memoize(hanning_window)


def make_hilb(N):
    """ Compute coefficients for a hilbert transforma of length N"""
    assert N%2 == 1, "Hilbert transform must have odd length"
    N = N+1
    allpass = zeros(N, Float)
    allpass[int(N/2)] = 1.0
    
    result = ifft(fft(allpass)*-1j) * hanning_window(N) 
    result = result[1:]
    return normalizeat(result, 0.5)   # normalize at middle of band

make_hilb = memoize(make_hilb)


def make_diff(N):
    """ Compute coefficients for a differentiator of length N"""
    assert N%2 == 0, "Differentiator must have even length"
    diff = zeros(N, Float)
    diff[int(N/2)-1] = -1.0
    diff[int(N/2)] = 1.0
   
    f = fft(diff)
    f[0] = 1
    f = f/absolute(f)               # leave just phase information
    f = f*arange(len(f))/(len(f)-1) # set linearly increasing amplitude
    result = ifft(f) * hanning_window(N) 
    return normalizeat(result, 0.5) * 0.5

make_diff = memoize(make_diff)


def make_filter(minfreq, maxfreq, N):
    """ Compute coefficients for a bandpass filter. Frequencies relative
    to Fs/2. Can also be used for low/high pass """
    assert N%2 == 1, "lowpass filter must have odd length"
    N = N+1
    allpass = zeros(N, Float)
    allpass[int(N/2)] = 1.0

    response = zeros(1+N/2, Float)
    response[ int(minfreq*len(response)) : int(maxfreq*len(response)) ] = 1

    result = ifft(fft(allpass)*response) * hanning_window(N)

    if minfreq == 0:
        normfreq = 0
    else:
        normfreq = (minfreq+maxfreq)/2
    return normalizeat(result[1:], normfreq)

make_filter = memoize(make_filter)


def normalizeat(x, freq=0.0):
    """ Normalize a filter for gain of 1 at target frequency """
    fx = fft(x)
    gain = absolute(fx[int(freq*len(fx))])
    return x/gain


def limiter(x):
    """ Clip a complex signal to amplitude 1 """
    a = absolute(x)
    mask = (a <= 1e-6)          # Avoid division by zero
    return (x+mask)/(a+mask)


def demodulate(i, iterations=3, bwfraction=1):
    """ Demodulate FM signal i. Output range is 0..1. Data near beginning 
    and end of vector is not accurate. """

    # Multiple iterations help reduce effects of square wave
    for n in range(iterations):
        iq = limiter(i + 1j*hilb(i))     # create complex signal, limit it
        i = iq.real

    return lowpass(absolute(diff(iq)), bwfraction)
