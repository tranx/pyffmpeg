from distutils.core import setup
from distutils.extension import Extension
from Pyrex.Distutils import build_ext

import sys
if sys.platform == 'win32':
	setup(
	  name = "pyffmpeg",
	  ext_modules=[ 
	    Extension("pyffmpeg", ["pyffmpeg.pyx"],
		define_macros=[('EMULATE_INTTYPES', '1')],
		include_dirs=["/usr/include/ffmpeg"], 
		library_dirs=[r"\usr\lib"], 
		libraries = ["avutil-49","avformat-50","avcodec-51"])
	    ],
	  cmdclass = {'build_ext': build_ext}
	)
else:
	setup(
	  name = "pyffmpeg",
	  ext_modules=[ 
	    Extension("pyffmpeg", ["pyffmpeg.pyx"],
		include_dirs=["/usr/include/ffmpeg"], 
		libraries = ["avformat","avcodec"])
	    ],
	  cmdclass = {'build_ext': build_ext},
	  version = "0.2.0",
	  author = "James Evans",
	  author_email = "jaevans@users.sf.net",
	  url = "http://www.clark-evans.com/~milamber/pyffmpeg",
	)

