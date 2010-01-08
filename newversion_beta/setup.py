# -*- coding: utf-8 -*-
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os.path import join as path_join
from sys import platform
try:
  import numpy.distutils.misc_util as nd
  with_numpy=True
except:
  with_numpy=False
  sys.stderr.write("Numpy does not seems to be installed on your system.\n")
  sys.stderr.write("You may still use pyffmpeg but audiosupport and numpy-bride are disabled.\n")  


##
## Try to locate source if necessary
##

if platform == 'win32':
    ffmpegpath = r'c:\ffmpeg-static'
    for x in [ r'..\ffmpeg',  r'c:\ffmpeg-static', r'c:\ffmpeg' ]:
        try:
             os.stat(x)
             ffmpegpath = x
        except:
            pass
else:
    ffmpegpath = '/opt/ffmpeg'
    for x in [ os.environ["HOME"]+"build/ffmpeg",  '/usr/local/ffmpeg',  '/opt/ffmpeg' ]:
        try:
             os.stat(x)
             ffmpegpath = x
        except:
            pass    

#
# Try to resove
# static dependencies resolution by looking into pkgconfig files
def static_resolver(libs):
    deps = []

    for lib in libs:
        try:
            pc = open(path_join(ffmpegpath, 'lib', 'pkgconfig', 'lib' + lib + '.pc'))
        except IOError:
            continue

        # we only need line starting with 'Libs:'
        l = filter(lambda x: x.startswith('Libs:'), pc).pop().strip()

        # we only need after '-lmylib' and one entry for library
        d = l.split(lib, 1).pop().split()

        # remove '-l'
        d = map(lambda x: x[2:], d)

        # empty list means no deps
        if len(d): deps += d

    # Unique list
    result = list(libs)
    map(lambda x: x not in result and result.append(x), deps)
    return result

try:
    #
    # compatibility with old FFMPEG ubuntu packages
    #
    os.stat("/usr/include/ffmpeg")
    os.mkdir("include")
    os.stat("/usr/include/ffmpeg/avcodec.h")    
    os.system("ln -s /usr/include/ffmpeg ./include/libavcodec")
    os.stat("/usr/include/ffmpeg/avformat.h")        
    os.system("ln -s /usr/include/ffmpeg ./include/libavformat")    
    os.stat("/usr/include/ffmpeg/avutil.h")        
    os.system("ln -s /usr/include/ffmpeg ./include/libavutil")
    os.stat("/usr/include/ffmpeg/swscale.h")        
    os.system("ln -s /usr/include/ffmpeg ./include/libswscale")    
except:
    pass


libs = ('avformat', 'avcodec', 'swscale')
incdir = [ path_join(ffmpegpath, 'include'), "/usr/include/ffmpeg" , "./include" ] + nd.get_numpy_include_dirs()
libinc = [ path_join(ffmpegpath, 'lib') ]

if platform == 'win32':
    libs = static_resolver(libs + ('avutil',))
    libinc += [ r'/mingw/lib' ] # why my mingw is unable to find libz2?

with_numpy=True

if with_numpy:
        ext_modules=[ Extension('pyffmpeg', [ 'pyffmpeg.pyx' ],
                       include_dirs = incdir,
                       library_dirs = libinc,
                       libraries = libs),
                      Extension('audioqueue', [ 'audioqueue.pyx' ],
                       include_dirs = incdir,
                       library_dirs = libinc,
                       libraries = libs),
                      Extension('pyffmpeg_numpybindings', [ 'pyffmpeg_numpybindings.pyx' ],
                       include_dirs = incdir,
                       library_dirs = libinc,
                       libraries = libs)
                     ]
else:
        ext_modules=[ Extension('pyffmpeg', [ 'pyffmpeg.pyx' ],
                       include_dirs = incdir,
                       library_dirs = libinc,
                       libraries = libs)
                    ]


setup(
    name = 'pyffmpeg',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
