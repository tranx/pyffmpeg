# -*- coding: utf-8 -*-
import numpy,random,re,sys,os
from pyffmpeg import *

from PyQt4 import QtCore
from PyQt4 import QtGui


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


# select your database
directory=sys.argv[1]

# instantiate the display
ld=LazyDisplayQt()


# set parameters
display_sz=(600,800)
n=(len(sys.argv)>2) and int(sys.argv[2]) or 8
subdisplay_nb=(n,n)


#compute the size of each video
shp=(display_sz[0]//subdisplay_nb[0], display_sz[1]//subdisplay_nb[1])

# initials buffers
img=numpy.zeros(display_sz+(3,),dtype=numpy.uint8)
subdisplay=numpy.zeros(subdisplay_nb,dtype=object)

# look for videofiles
files=filter(lambda x:re.match("(.*).(mpg|avi|flv)",x),os.listdir(directory))

# specify to open only video at the correct size
TS={ 'video1':(0, -1, {'pixel_format':PixelFormats.RGB24,'videoframebanksz':1, 'dest_width':shp[1], 'dest_height':shp[0] })}


# do play, and reinstantiate players in case of error
while True:
  ld.f(img)
  for xx in numpy.ndindex(subdisplay_nb):
    try:
      subdisplay[xx].step()
    except:
        mp=FFMpegReader()
        maxtries=4
        def do_display(subimg):
          x=shp[1]*xx[1]
          y=shp[0]*xx[0]
          dy,dx=shp
          img[y:(y+dy),x:(x+dx) ]=subimg

        while (maxtries>0):
          try:
             mp.open(directory+"/"+random.choice(files),TS)
             mp.seek_to(random.randint(1,1024))
             mp.get_tracks()[0].set_observer(do_display)
             mp.step()
             maxtries=0
          except:
             maxtries-=1
        subdisplay[xx]=mp
        



