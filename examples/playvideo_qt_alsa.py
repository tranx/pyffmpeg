# -*- coding: utf-8 -*-
## Simple demo for pyffmpegb 
## 
## Copyright -- Bertrand Nouvel 2009

## import your modules

from pyffmpeg import *
from PyQt4 import QtCore
from PyQt4 import QtGui

import sys, numpy
import alsaaudio

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

class AlsaSoundLazyPlayer:
    def __init__(self,rate=44100,channels=2,fps=25):
        self._rate=rate
        self._channels=channels
        self._d = alsaaudio.PCM()
        self._d.setchannels(channels)
        self._d.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self._d.setperiodsize((rate*channels)//fps)
        self._d.setrate(rate)
    def push_nowait(self,stamped_buffer):
        self._d.write(stamped_buffer[0].data)


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


TS_VIDEO_RGB24={ 'video1':(0, -1, {'pixel_format':PixelFormats.RGB24}), 'audio1':(1,-1,{})}
TS_VIDEO_BGR24={ 'video1':(0, -1, {'pixel_format':PixelFormats.BGR24}), 'audio1':(1,-1,{})}


## create the reader object
mp=FFMpegReader()


## open an audio video file
vf=sys.argv[1]
mp.open(vf,TS_VIDEO_RGB24)
tracks=mp.get_tracks()

## connect video and audio to their respective device
ld=LazyDisplayQt()
tracks[0].set_observer(ld.f)

ap=AlsaSoundLazyPlayer(tracks[1].get_samplerate(),tracks[1].get_channels(),tracks[0].get_fps())
tracks[1].set_observer(ap.push_nowait)

tracks[0].seek_to_seconds(10)
## play the movie !

mp.run()


