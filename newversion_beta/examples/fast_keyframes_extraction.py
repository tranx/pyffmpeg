# -*- coding: utf-8 -*-
## Simple demo for pyffmpegb 
## 
## Copyright -- Bertrand Nouvel 2009

## import your modules

from pyffmpeg import *
from PyQt4 import QtCore
from PyQt4 import QtGui

import sys, numpy
#import alsaaudio

try:
    LazyDisplayQt__imgconvarray={
                      1:QtGui.QImage.Format_Indexed8,
                      3:QtGui.QImage.Format_RGB888,
                      4:QtGui.QImage.Format_RGB32
                      }
except:
    LazyDisplayQt__imgconvarray={
                      1:QtGui.QImage.Format_Indexed8,
                      4:QtGui.QImage.Format_RGB32
                      }

qapp = QtGui.QApplication(sys.argv)
qapp.processEvents()

class LazyDisplayQt(QtGui.QMainWindow):
        imgconvarray=LazyDisplayQt__imgconvarray
        def __init__(self, *args):
            QtGui.QMainWindow.__init__(self, *args)
            self._i=numpy.zeros((1,1,4),dtype=numpy.uint8)
            self.i=QtGui.QImage(self._i.data,self._i.shape[1],self._i.shape[0],self.imgconvarray[self._i.shape[2]])
            self.show()
        def __del__(self):
            self.hide()
        def f(self,thearray):
            self._i=thearray.astype(numpy.uint8).copy('C')
            self.i=QtGui.QImage(self._i.data,self._i.shape[1],self._i.shape[0],self.imgconvarray[self._i.shape[2]])
            self.update()
            qapp.processEvents()
        def paintEvent(self, ev):
            self.p = QtGui.QPainter()
            self.p.begin(self)
            self.p.drawImage(QtCore.QRect(0,0,self.width(),self.height()),
                             self.i,
                             QtCore.QRect(0,0,self.i.width(),self.i.height()))
            self.p.end()


TS_VIDEO_RGB24={ 'video1':(0, -1, {'pixel_format':PixelFormats.RGB24,'videoframebanksz':1, 'skip_frame':32})}#, 'audio1':(1,-1,{}) }
TS_VIDEO_BGR24={ 'video1':(0, -1, {'pixel_format':PixelFormats.BGR24,'videoframebanksz':1, 'skip_frame':32})}#, 'audio1':(1,-1,{})}
TS_VIDEO_GRAY8={ 'video1':(0, -1, {'pixel_format':PixelFormats.GRAY8,'videoframebanksz':1, 'skip_frame':32})}#, 'audio1':(1,-1,{})}


## create the reader object
mp=FFMpegReader()


## open an audio video file
vf=sys.argv[1]
mp.open(vf,TS_VIDEO_RGB24)
tracks=mp.get_tracks()

## connect video and audio to their respective device
ld=LazyDisplayQt()
tt=0

def obs(x):
   global tt
   #print tracks[0].get_current_frame_type()
   tt+=1
   if (tt%1000==0):
     print tracks[0].get_cur_pts()
     ld.f(x.reshape(x.shape[0],x.shape[1],1).repeat(3,axis=2))

   
 
tracks[0].set_observer(obs)

import time
print time.clock()
dur=mp.duration()/1000000
print "Duration = ",dur
try:
   mp.run()
except Exception, e:
  print "Exception e=", e
  print "Processing time=",time.clock()
  print tt," keyframes"
  print (dur*30.)/tt ," frames per keyframe"
  print (dur)/time.clock() ," times faster than rt"


