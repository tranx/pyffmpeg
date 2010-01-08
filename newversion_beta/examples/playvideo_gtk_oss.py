# -*- coding: utf-8 -*-
## Simple demo for pyffmpegb and pygtk
## 
## Copyright -- Sebastien Campion

## import your modules

import sys, numpy, time, StringIO, Image, threading
from pyffmpeg import *

try:
  import ossaudiodev as oss
except:
  import oss
  
import pygtk, gtk
pygtk.require('2.0')
gtk.gdk.threads_init()




class play(threading.Thread):
    def run(self):
      global mp
      mp.run()


class pyffplay:
    def __init__(self,width,height):
        self.builder = gtk.Builder()
        self.builder.add_from_file("example2.xml")
        self.window = self.builder.get_object("window")
        self.screen = self.builder.get_object("screen")
        self.builder.connect_signals(self)
        self.size = (width,height)
        
    def image2pixbuf(self,im):
      """
      convert a PIL image to pixbuff
      """
      file1 = StringIO.StringIO()  
      im.save(file1, "ppm")  
      contents = file1.getvalue()  
      file1.close()  
      loader = gtk.gdk.PixbufLoader("pnm")  
      loader.write(contents, len(contents))  
      pixbuf = loader.get_pixbuf()  
      loader.close()  
      return pixbuf  
   

    def displayframe(self,thearray):
      """
      pyffmpeg callback
      """
      _i = thearray.astype(numpy.uint8).copy('C')
      _i_height=_i.shape[0]
      _i_width = _i.shape[1]
     
      frame = Image.fromstring("RGB",(_i_width,_i_height),_i.data)
      frame = frame.resize(self.size)
      self.screen.set_from_pixbuf(self.image2pixbuf(frame))

    def on_play_clicked(self,widget):
      play().start()
    
 


# create a pygtk window
pyff = pyffplay(320,240)
pyff.window.show_all()

TS_VIDEO_RGB24={ 'video1':(0, -1, {'pixel_format':PixelFormats.RGB24}), 'audio1':(1,-1,{})}
TS_VIDEO_BGR24={ 'video1':(0, -1, {'pixel_format':PixelFormats.BGR24}), 'audio1':(1,-1,{})}


## create the reader object
mp=FFMpegReader()

## open an audio video file
vf=sys.argv[1]


mp.open(vf,TS_VIDEO_RGB24)
tracks=mp.get_tracks()


## connect video and audio to their respective device
tracks[0].set_observer(pyff.displayframe)

ao=oss.open('w')

ao.speed(tracks[1].get_samplerate())
ao.setfmt(oss.AFMT_S16_LE)
ao.channels(tracks[1].get_channels())
tracks[1].set_observer(lambda x:ao.write(x[0].data))
tracks[0].seek_to_seconds(10)


gtk.main()



