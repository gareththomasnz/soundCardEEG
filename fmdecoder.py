#!/usr/bin/env python
""" FM demodulator for OpenEEG project. 

(C) 2003 Oren Tirosh <oren@hishome.net>
Released under the GNU General Public License

Requires Numerical Python extension 
  http://www.pfdubois.com/numpy/
"""

from fm import *

class WAVError(Exception):
    pass


# struct module format for WAVE, fmt and data chunks:
wav_hdr_fmt = "<"+"4sI4s"+"4sIHHIIHH"+"4sI"   

def load_wav(fname):
    """ Load WAV file. Returns tuple of (sampling frequency, channels) """
    import struct
    f = open(fname, 'rb')
    header = f.read(44)
    (
        RIFFID,
        RIFFSize,
        WAVEID,

        fmtID,
        fmtSize,
        AudioFormat,
        NumChannels,
        SampleRate,
        ByteRate,
        BLockAlign,
        BitsPerSample,

        dataID,
        dataSize
    ) = struct.unpack(wav_hdr_fmt, header)
    if not (RIFFID == 'RIFF' and WAVEID == 'WAVE'):
        raise WAVError, "File is not in WAV format."
    if not (fmtID == 'fmt ' and dataID == 'data' and 
            fmtSize == 16 and AudioFormat == 1 and 
            NumChannels in (1,2) and BitsPerSample == 16):
        raise WAVError, "Unsupported WAV subformat"
    data = fromstring(f.read(dataSize), Int16)

    if NumChannels == 1:
        channels = [data]
    elif NumChannels == 2:
        channels = [data[0::2], data[1::2]]

    return SampleRate, channels

def save_wav(fname, fs, channels):
    import struct

    samples = len(channels)*len(channels[0])

    header = struct.pack(
        wav_hdr_fmt,
        'RIFF',
        36+samples*2,
        'WAVE',
        'fmt ',
        16,
        1,
        len(channels),
        fs,
        fs*2*len(channels),
        2*len(channels),
        16,
        'data',
        samples*2
    )
    f = open(fname, 'w')
    f.write(header)

    if len(channels) == 1:
        data = channels[0].astype(Int16)
    elif len(channels) == 2:
        data = zeros(samples, Int16)
        data[0::2] = channels[0].astype(Int16)
        data[1::2] = channels[1].astype(Int16)
    else:
        raise WAVError, 'Unsupported number of channels'
   
    f.write(data)
    f.close()


def demodulate_file(fnamei, fnameo, targetrate, bandwidth):
    samplerate, channels = load_wav(fnamei)

    if targetrate == 0:
        targetrate = 256

    if bandwidth == 0:
        bandwidth = 99999

    ratio = int(samplerate/targetrate)
    bw = min(bandwidth, 0.45*targetrate)
    bwfraction = bw/(0.5*samplerate)

    for n in range(len(channels)):
        channels[n] = demodulate(channels[n], 3, bwfraction)[::ratio]*32767.0

    save_wav(fnameo, targetrate, channels)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit("Usage: fm.py input.wav output.wav [target sampling rate, [bandwidth]]")
    args = sys.argv + ['0', '0']
    demodulate_file(
        args[1], args[2], int(args[3]), int(args[4]) )

