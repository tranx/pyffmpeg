# -*- coding: utf-8 -*-
###############################################################
###############################################################
###############################################################
# Example of spectrogram computations from sound/video file 
###############################################################

import sys
import numpy 
import pylab
import Image

def NumPy2PIL(input):
    """Converts a numpy array to a PIL image.

    Supported input array layouts:
       2 dimensions of numpy.uint8
       3 dimensions of numpy.uint8
       2 dimensions of numpy.float32
    """
    if not isinstance(input, numpy.ndarray):
        raise TypeError, 'Must be called with numpy.ndarray!'
    # Check the number of dimensions of the input array
    ndim = input.ndim
    if not ndim in (2, 3):
        raise ValueError, 'Only 2D-arrays and 3D-arrays are supported!'
    if ndim == 2:
        channels = 1
    else:
        channels = input.shape[2]
    # supported modes list: [(channels, dtype), ...]
    modes_list = [(1, numpy.uint8), (3, numpy.uint8), (1, numpy.float32), (4,numpy.uint8)]
    mode = (channels, input.dtype)
    if not mode in modes_list:
        raise ValueError, 'Unknown or unsupported input mode'
    return Image.fromarray(input)



def hamming(lw):
   return 0.54-0.46*numpy.cos(numpy.pi*2.*numpy.arange(0,1.,1./lw))


from pyffmpeg import *
frate=44100.
freq=8
df=2048
do=df-(df/freq)
di=df-do
nx=df//di

TS_AUDIO={ 'audio1':(1, -1, {'hardware_queue_len':1000,'dest_frame_size':df, 'dest_frame_overlap':do} )}

class Observer():
  def __init__(self):
     self.ctr=0
     self.ark=[]
     self.arp=[]     
  def observe(self,x):
     self.ctr+=1
     fftsig=numpy.fft.fft(x[0].mean(axis=1)/32768.)
     spectra=numpy.roll(fftsig,fftsig.shape[0]//2,axis=0)
     spect=(20*numpy.log10(0.0001+numpy.abs(spectra)))*4
     specp=numpy.angle(spectra)
#     print spect.min() , spect.max()
     spect=spect.clip(0,255)
     spect=spect.astype(numpy.uint8)
     self.arp.append(((specp/numpy.pi)*127).astype(numpy.int8))
     self.ark.append(spect.copy('C'))

observer=Observer()


for f in sys.argv[1:]:
  observer.ark=[]
  print "processing ",f
  rdr=FFMpegReader()
  rdr.open(f,track_selector=TS_AUDIO)    
  track=rdr.get_tracks()
  track[0].set_observer(observer.observe)
  try:
    rdr.run()
  except IOError:
    pass
  arim=numpy.vstack(observer.ark)
  arip=numpy.vstack(observer.arp) 
  for i in range(0,arim.shape[0],5000):
     xcmap=(pylab.cm.hsv(arip[i:(i+5000),:].astype(numpy.float)/255.)*arim[i:(i+5000),:].reshape(arim[i:(i+5000),:].shape+(1,)).repeat(4,axis=2)).astype(numpy.uint8)[:,:,:3].copy('C')
     NumPy2PIL(xcmap).save("spectrogram-%s-%d.png"%(f.split('/')[-1].split('.')[0] ,i,))
