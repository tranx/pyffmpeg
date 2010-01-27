# -*- coding: utf-8 -*-
import sys
import pyffmpeg

reader = pyffmpeg.FFMpegReader(False)
reader.open(sys.argv[1],pyffmpeg.TS_VIDEO_PIL)
vt=reader.get_tracks()[0]
print dir(vt)
nframes=31
try:
  rdrdur=reader.duration()
  rdrdurtime=reader.duration_time()
except:
  print "no duration information in reader"
try:
  cdcdur=vt.duration()
  cdcdurtime=vt.duration_time()
  mt=max(cdcdurtime,rdrdurtime)
  print rdrdurtime, cdcdurtime
  print "FPS=",vt.get_fps()
  nframes=min(mt*vt.get_fps(),1000)
  print "NFRAMES= (max=1000)",nframes
except KeyError: 
  print "no duration information in track"
for i in range(nframes,0,-1):
  try:
    vt.seek_to_frame(i)
    image=vt.get_current_frame()[2]
    image.save('frame-%04d.png'%(i,))
  except:
    print "missing frame %d"%(i,)
