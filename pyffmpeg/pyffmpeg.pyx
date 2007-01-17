# pyffmpeg.pyx - read frames from video files and convert to PIL image
#
# Copyright (C) 2006-2007 James Evans <jaevans@users.sf.net>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import Image

ctypedef signed long long int64_t

cdef enum:
	SEEK_SET = 0
	SEEK_CUR = 1
	SEEK_END = 2
	
cdef extern from "Python.h":
	ctypedef int size_t
	object PyBuffer_FromMemory(	void *ptr, int size)
	object PyString_FromStringAndSize(char *s, int len)
	void* PyMem_Malloc( size_t n)
	void PyMem_Free( void *p)

cdef extern from "mathematics.h":
	int64_t av_rescale(int64_t a, int64_t b, int64_t c)
	
cdef extern from "avutil.h":
	cdef enum PixelFormat:
		PIX_FMT_NONE= -1,
		PIX_FMT_YUV420P,   #< Planar YUV 4:2:0 (1 Cr & Cb sample per 2x2 Y samples)
		PIX_FMT_YUV422,    #< Packed pixel, Y0 Cb Y1 Cr
		PIX_FMT_RGB24,     #< Packed pixel, 3 bytes per pixel, RGBRGB...
		PIX_FMT_BGR24,     #< Packed pixel, 3 bytes per pixel, BGRBGR...
		PIX_FMT_YUV422P,   #< Planar YUV 4:2:2 (1 Cr & Cb sample per 2x1 Y samples)
		PIX_FMT_YUV444P,   #< Planar YUV 4:4:4 (1 Cr & Cb sample per 1x1 Y samples)
		PIX_FMT_RGBA32,    #< Packed pixel, 4 bytes per pixel, BGRABGRA..., stored in cpu endianness
		PIX_FMT_YUV410P,   #< Planar YUV 4:1:0 (1 Cr & Cb sample per 4x4 Y samples)
		PIX_FMT_YUV411P,   #< Planar YUV 4:1:1 (1 Cr & Cb sample per 4x1 Y samples)
		PIX_FMT_RGB565,    #< always stored in cpu endianness
		PIX_FMT_RGB555,    #< always stored in cpu endianness, most significant bit to 1
		PIX_FMT_GRAY8,
		PIX_FMT_MONOWHITE, #< 0 is white
		PIX_FMT_MONOBLACK, #< 0 is black
		PIX_FMT_PAL8,      #< 8 bit with RGBA palette
		PIX_FMT_YUVJ420P,  #< Planar YUV 4:2:0 full scale (jpeg)
		PIX_FMT_YUVJ422P,  #< Planar YUV 4:2:2 full scale (jpeg)
		PIX_FMT_YUVJ444P,  #< Planar YUV 4:4:4 full scale (jpeg)
		PIX_FMT_XVMC_MPEG2_MC,#< XVideo Motion Acceleration via common packet passing(xvmc_render.h)
		PIX_FMT_XVMC_MPEG2_IDCT,
		PIX_FMT_UYVY422,   #< Packed pixel, Cb Y0 Cr Y1
		PIX_FMT_UYVY411,   #< Packed pixel, Cb Y0 Y1 Cr Y2 Y3
		PIX_FMT_NB,

cdef extern from "avcodec.h":
	# use an unamed enum for defines
	cdef enum:
		AVSEEK_FLAG_BACKWARD = 1 #< seek backward
		AVSEEK_FLAG_BYTE     = 2 #< seeking based on position in bytes
		AVSEEK_FLAG_ANY      = 4 #< seek to any frame, even non keyframes
		CODEC_CAP_TRUNCATED = 0x0008
		CODEC_FLAG_TRUNCATED = 0x00010000 # input bitstream might be truncated at a random location instead of only at frame boundaries
		AV_TIME_BASE = 1000000
		FF_I_TYPE = 1 # Intra
		FF_P_TYPE = 2 # Predicted
		FF_B_TYPE = 3 # Bi-dir predicted
		FF_S_TYPE = 4 # S(GMC)-VOP MPEG4
		FF_SI_TYPE = 5
		FF_SP_TYPE = 6

		AV_NOPTS_VALUE = <int64_t>0x8000000000000000

	enum AVDiscard:
		# we leave some space between them for extensions (drop some keyframes for intra only or drop just some bidir frames)
		AVDISCARD_NONE   = -16 # discard nothing
		AVDISCARD_DEFAULT=   0 # discard useless packets like 0 size packets in avi
		AVDISCARD_NONREF =   8 # discard all non reference
		AVDISCARD_BIDIR  =  16 # discard all bidirectional frames
		AVDISCARD_NONKEY =  32 # discard all frames except keyframes
		AVDISCARD_ALL    =  48 # discard all
		
		
	struct AVCodecContext:
		int codec_type
		int codec_id
		int flags
		int width
		int height
		int pix_fmt
		int frame_number
		int hurry_up
		int skip_idct
		int skip_frame
		
	struct AVRational:
		int num
		int den

	enum CodecType:
		CODEC_TYPE_UNKNOWN = -1
		CODEC_TYPE_VIDEO = 0
		CODEC_TYPE_AUDIO = 1
		CODEC_TYPE_DATA = 2
		CODEC_TYPE_SUBTITLE = 3

	struct AVCodec:
		char *name
		int type
		int id
		int priv_data_size
		int capabilities
		AVCodec *next
		AVRational *supported_framerates #array of supported framerates, or NULL if any, array is terminated by {0,0}
		int *pix_fmts       #array of supported pixel formats, or NULL if unknown, array is terminanted by -1

	struct AVPacket:
		int64_t pts                            #< presentation time stamp in time_base units
		int64_t dts                            #< decompression time stamp in time_base units
		char *data
		int   size
		int   stream_index
		int   flags
		int   duration                      #< presentation duration in time_base units (0 if not available)
		void  *priv
		int64_t pos                            #< byte position in stream, -1 if unknown

	struct AVFrame:
		char *data[4]
		int linesize[4]
		int64_t pts
		int pict_type
		int key_frame

	struct AVPicture:
		pass
	AVCodec *avcodec_find_decoder(int id)
	int avcodec_open(AVCodecContext *avctx, AVCodec *codec)
	int avcodec_decode_video(AVCodecContext *avctx, AVFrame *picture,
                         int *got_picture_ptr,
                         char *buf, int buf_size)
	int avpicture_fill(AVPicture *picture, void *ptr,
                   int pix_fmt, int width, int height)
	AVFrame *avcodec_alloc_frame()
	int avpicture_get_size(int pix_fmt, int width, int height)
	int avpicture_layout(AVPicture* src, int pix_fmt, int width, int height,
                     unsigned char *dest, int dest_size)
	int img_convert(AVPicture *dst, int dst_pix_fmt,
                AVPicture *src, int pix_fmt,
                int width, int height)
				
	void avcodec_flush_buffers(AVCodecContext *avctx)



cdef extern from "avformat.h":
	struct AVFrac:
		int64_t val, num, den

	void av_register_all()

	struct AVCodecParserContext:
		pass

	struct AVIndexEntry:
		pass

	struct AVStream:
		int index    #/* stream index in AVFormatContext */
		int id       #/* format specific stream id */
		AVCodecContext *codec #/* codec context */
		# real base frame rate of the stream.
		# for example if the timebase is 1/90000 and all frames have either
		# approximately 3600 or 1800 timer ticks then r_frame_rate will be 50/1
		AVRational r_frame_rate
		void *priv_data
		# internal data used in av_find_stream_info()
		int64_t codec_info_duration
		int codec_info_nb_frames
		# encoding: PTS generation when outputing stream
		AVFrac pts
		# this is the fundamental unit of time (in seconds) in terms
		# of which frame timestamps are represented. for fixed-fps content,
		# timebase should be 1/framerate and timestamp increments should be
		# identically 1.
		AVRational time_base
		int pts_wrap_bits # number of bits in pts (used for wrapping control)
		# ffmpeg.c private use
		int stream_copy   # if TRUE, just copy stream
		int discard       # < selects which packets can be discarded at will and dont need to be demuxed
		# FIXME move stuff to a flags field?
		# quality, as it has been removed from AVCodecContext and put in AVVideoFrame
		# MN:dunno if thats the right place, for it
		float quality
		# decoding: position of the first frame of the component, in
		# AV_TIME_BASE fractional seconds.
		int64_t start_time
		# decoding: duration of the stream, in AV_TIME_BASE fractional
		# seconds.
		int64_t duration
		char language[4] # ISO 639 3-letter language code (empty string if undefined)
		# av_read_frame() support
		int need_parsing                  # < 1->full parsing needed, 2->only parse headers dont repack
		AVCodecParserContext *parser
		int64_t cur_dts
		int last_IP_duration
		int64_t last_IP_pts
		# av_seek_frame() support
		AVIndexEntry *index_entries # only used if the format does not support seeking natively
		int nb_index_entries
		int index_entries_allocated_size
		int64_t nb_frames                 # < number of frames in this stream if known or 0

	struct ByteIOContext:
		pass

	struct AVFormatContext:
		int nb_streams
		AVStream **streams
		int64_t timestamp
		int64_t start_time
		AVStream *cur_st
		AVPacket cur_pkt
		ByteIOContext pb
		# decoding: total file size. 0 if unknown
		int64_t file_size
		int64_t duration
		# decoding: total stream bitrate in bit/s, 0 if not
		# available. Never set it directly if the file_size and the
		# duration are known as ffmpeg can compute it automatically. */
		int bit_rate
		# av_seek_frame() support
		int64_t data_offset    # offset of the first packet
		int index_built
		

	struct AVInputFormat:
		pass

	struct AVFormatParameters:
		pass

	int av_open_input_file(AVFormatContext **ic_ptr, char *filename,
                       AVInputFormat *fmt,
                       int buf_size,
                       AVFormatParameters *ap)
	int av_find_stream_info(AVFormatContext *ic)

	void dump_format(AVFormatContext *ic,
                 int index,
                 char *url,
                 int is_output)
	void av_free_packet(AVPacket *pkt)
	int av_read_packet(AVFormatContext *s, AVPacket *pkt)
	int av_read_frame(AVFormatContext *s, AVPacket *pkt)
	int av_seek_frame(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)
	int av_seek_frame_binary(AVFormatContext *s, int stream_index, int64_t target_ts, int flags)

	void av_parser_close(AVCodecParserContext *s)

	int av_index_search_timestamp(AVStream *st, int64_t timestamp, int flags)


cdef extern from "avio.h":
	int url_ferror(ByteIOContext *s)
	int url_feof(ByteIOContext *s)
	
cdef __registered
__registered = 0

def py_av_register_all():
	if __registered:
		return
	__registered = 1
	av_register_all()

cdef AVRational AV_TIME_BASE_Q
AV_TIME_BASE_Q.num = 1
AV_TIME_BASE_Q.den = AV_TIME_BASE
cdef class VideoStream:
	cdef AVFormatContext *FormatCtx
	cdef AVCodecContext *CodecCtx
	cdef AVCodec *Codec
	cdef AVPacket packet
	cdef int videoStream
	cdef AVFrame *frame
	cdef int frameno
	cdef object filename
	cdef object index
	cdef object keyframes
	
	def __new__(self):
		self.FormatCtx = NULL
		self.frame = avcodec_alloc_frame()
		self.frameno = 0
		self.videoStream = -1
		self.Codec = NULL
		self.filename = None
		self.index = None
		self.keyframes = None

	def dump(self):
		dump_format(self.FormatCtx,0,self.filename,0)

	def open(self,char *filename):
		cdef AVFormatContext *pFormatCtx
		cdef int ret
		cdef int i

		py_av_register_all()
		ret = av_open_input_file(&self.FormatCtx,filename,NULL,0,NULL)
		pFormatCtx = <AVFormatContext *>self.FormatCtx
		if ret != 0:
			raise IOError("Unable to open file %s" % filename)

		ret = av_find_stream_info(pFormatCtx)
		if ret < 0:
			raise IOError("Unable to find stream info: %d" % ret)

		self.videoStream = -1
		for i from 0 <= i < pFormatCtx.nb_streams:
			if pFormatCtx.streams[i].codec.codec_type == CODEC_TYPE_VIDEO:
				self.videoStream = i
				break
		if self.videoStream == -1:
			raise IOError("Unable to find video stream")

		self.CodecCtx = pFormatCtx.streams[self.videoStream].codec
		self.Codec = avcodec_find_decoder(self.CodecCtx.codec_id)

		if self.Codec == NULL:
			raise IOError("Unable to get decoder")

		# Inform the codec that we can handle truncated bitstreams -- i.e.,
		# bitstreams where frame boundaries can fall in the middle of packets
		if self.Codec.capabilities & CODEC_CAP_TRUNCATED:
			self.CodecCtx.flags = self.CodecCtx.flags & CODEC_FLAG_TRUNCATED
		# Open codec
		ret = avcodec_open(self.CodecCtx, self.Codec)
		if ret < 0:
			raise IOError("Unable to open codec")
		self.filename = filename
		
	cdef AVFrame *ConvertToRGBA(self,AVPicture *frame,AVCodecContext *pCodecCtx):
		cdef AVFrame *pFrameRGBA
		cdef int numBytes
		cdef char *rgb_buffer
		cdef int width,height

		pFrameRGBA = avcodec_alloc_frame()
		if pFrameRGBA == NULL:
			raise MemoryError("Unable to allocate RGB Frame")

		width = pCodecCtx.width
		height = pCodecCtx.height
		# Determine required buffer size and allocate buffer
		numBytes=avpicture_get_size(PIX_FMT_RGBA32, width,height)
		# Hrm, how do I figure out when to release the old one....
		rgb_buffer = <char *>PyMem_Malloc(numBytes)
		avpicture_fill(<AVPicture *>pFrameRGBA, rgb_buffer, PIX_FMT_RGBA32,
				width, height)

		img_convert(<AVPicture *>pFrameRGBA, PIX_FMT_RGBA32,
					<AVPicture *>frame, pCodecCtx.pix_fmt, width,
					height)
		return pFrameRGBA

	cdef AVFrame *ConvertToRGB24(self,AVPicture *frame,AVCodecContext *pCodecCtx):
		cdef AVFrame *pFrameRGB24
		cdef int numBytes
		cdef char *rgb_buffer
		cdef int width,height

		pFrameRGB24 = avcodec_alloc_frame()
		if pFrameRGB24 == NULL:
			raise MemoryError("Unable to allocate RGB Frame")

		width = pCodecCtx.width
		height = pCodecCtx.height
		# Determine required buffer size and allocate buffer
		numBytes=avpicture_get_size(PIX_FMT_RGB24, width,height)
		# Hrm, how do I figure out how to release the old one....
		rgb_buffer = <char *>PyMem_Malloc(numBytes)
		avpicture_fill(<AVPicture *>pFrameRGB24, rgb_buffer, PIX_FMT_RGB24,
				width, height)

		img_convert(<AVPicture *>pFrameRGB24, PIX_FMT_RGB24,
					<AVPicture *>frame, pCodecCtx.pix_fmt, width,
					height)
		return pFrameRGB24

	def SaveFrame(self):
		cdef int i
		cdef void *p
		cdef AVFrame *pFrameRGB
		cdef int width,height

		width = self.CodecCtx.width
		height = self.CodecCtx.height

		# I haven't figured out how to write RGBA data to an ppm file so I use a 24 bit version
		pFrameRGB = self.ConvertToRGB24(<AVPicture *>self.frame,self.CodecCtx)
		filename = "frame%04d.ppm" % self.frameno
		f = open(filename,"wb")

		f.write("P6\n%d %d\n255\n" % (width,height))
		f.flush()
		for i from 0 <= i < height:
			f.write(PyBuffer_FromMemory(pFrameRGB.data[0] + i * pFrameRGB.linesize[0],width * 3))
		f.close()
		PyMem_Free(pFrameRGB.data[0])

	def GetCurrentFrame(self):
		cdef AVFrame *pFrameRGB
		cdef object buf_obj
		cdef int numBytes

		pFrameRGB = self.ConvertToRGBA(<AVPicture *>self.frame,self.CodecCtx)
		numBytes=avpicture_get_size(PIX_FMT_RGBA32, self.CodecCtx.width, self.CodecCtx.height)
		buf_obj = PyBuffer_FromMemory(pFrameRGB.data[0],numBytes)

		img_image = Image.frombuffer("RGBA",(self.CodecCtx.width,self.CodecCtx.height),buf_obj,"raw","BGRA",pFrameRGB.linesize[0],1)
		PyMem_Free(pFrameRGB.data[0])
		return img_image
		
		
	def __next_frame(self):
		cdef int ret
		cdef int frameFinished
		cdef int64_t pts,pts2
		cdef AVStream *stream

		frameFinished = 0
		while frameFinished == 0:
			self.packet.stream_index = -1
			while self.packet.stream_index != self.videoStream:
				ret = av_read_frame(self.FormatCtx,&self.packet)
				if ret < 0:
					raise IOError("Unable to read frame: %d" % ret)
			ret = avcodec_decode_video(self.CodecCtx,self.frame,&frameFinished,self.packet.data,self.packet.size)
			if ret < 0:
				raise IOError("Unable to decode video picture: %d" % ret)

		if self.packet.pts == AV_NOPTS_VALUE:
			pts = self.packet.dts
		else:
			pts = self.packet.pts
		stream = self.FormatCtx.streams[self.videoStream]
		return av_rescale(pts,AV_TIME_BASE * <int64_t>stream.time_base.num,stream.time_base.den)

	def GetNextFrame(self):
		self.__next_frame()
		return self.GetCurrentFrame()                  

	def build_index(self,fast = True):
		if fast == True:
			return self.build_index_fast()
		else:
			return self.build_index_full()
			
	def build_index_full(self):
		cdef int ret,ret2
		
		cdef int frameFinished
		cdef AVStream *stream
		cdef int64_t myPts,pts,time_base
		cdef int frame_no
		
		if self.index is not None:
			# already indexed
			return
		self.index = {}
		self.keyframes = []
		stream = self.FormatCtx.streams[self.videoStream]
		time_base = AV_TIME_BASE * <int64_t>stream.time_base.num
		ret = av_seek_frame(self.FormatCtx,self.videoStream, 0, AVSEEK_FLAG_BACKWARD)
		if ret < 0:
			raise IOError("Error rewinding stream for full indexing: %d" % ret)
		avcodec_flush_buffers(self.CodecCtx)
		
		frame_no = 0
		while True:
			frameFinished = 0
			while frameFinished == 0:
				ret = av_read_frame(self.FormatCtx,&self.packet)
				if ret < 0:
					# check for eof condition
					ret2 = url_feof(&self.FormatCtx.pb)
					if ret2 == 0:
						raise IOError("Error reading frame for full indexing: %d" % ret)
					else:
						frameFinsished = 1
						break
				if self.packet.stream_index != self.videoStream:
					# only looking for video packets
					continue
				if self.packet.pts == AV_NOPTS_VALUE:
					pts = self.packet.dts
				else:
					pts = self.packet.pts
				myPts = av_rescale(pts,time_base,stream.time_base.den)
				
				ret = avcodec_decode_video(self.CodecCtx,self.frame,&frameFinished,self.packet.data,self.packet.size)
				if ret < 0:
					raise IOError("Unable to decode video picture: %d" % ret)
			if self.frame.pict_type == FF_I_TYPE:
				myType = 'I'
			elif self.frame.pict_type == FF_P_TYPE:
				myType = 'P'
			elif self.frame.pict_type == FF_B_TYPE:
				myType = 'B'
			elif self.frame.pict_type == FF_S_TYPE:
				myType = 'S'
			elif self.frame.pict_type == FF_SI_TYPE:
				myType = 'SI'
			elif self.frame.pict_type == FF_SP_TYPE:
				myType = 'SP'
			else:
				myType = 'U'
			self.index[frame_no] = (myPts,myType)
			frame_no = frame_no + 1
			if self.frame.key_frame:
				self.keyframes.append(myPts)
				
		ret = av_seek_frame(self.FormatCtx,self.videoStream, 0, AVSEEK_FLAG_BACKWARD)
		if ret < 0:
			raise IOError("Error rewinding stream after full indexing: %d" % ret)		
		avcodec_flush_buffers(self.CodecCtx)
		
	def build_index_fast(self):
		cdef int ret,ret2
		cdef int64_t myPts,pts,time_base
		cdef AVStream *stream
		
		if self.keyframes is not None:
			# already fast indexed
			return
		self.keyframes = []
		stream = self.FormatCtx.streams[self.videoStream]
		ret = av_seek_frame(self.FormatCtx,self.videoStream, 0, AVSEEK_FLAG_BACKWARD)
		if ret < 0:
			raise IOError("Error rewinding stream for fast indexing: %d" % ret)
		
		
		avcodec_flush_buffers(self.CodecCtx)
		time_base = AV_TIME_BASE * <int64_t>stream.time_base.num
		frame_no = 0

		self.CodecCtx.skip_idct = AVDISCARD_NONKEY
		self.CodecCtx.skip_frame = AVDISCARD_NONKEY
		while True:
			ret = av_read_frame(self.FormatCtx,&self.packet)
			if ret < 0:
				ret2 = url_feof(&self.FormatCtx.pb)
				if  ret2 == 0:
					raise IOError("Error reading frame for fast indexing: %d" % ret)
				else:
					break
			if self.packet.stream_index != self.videoStream:
				continue
			if self.packet.pts == AV_NOPTS_VALUE:
				pts = self.packet.dts
			else:
				pts = self.packet.pts
			myPts = av_rescale(pts,time_base,stream.time_base.den)
			self.keyframes.append(myPts)
		self.CodecCtx.skip_idct = AVDISCARD_ALL
		self.CodecCtx.skip_frame = AVDISCARD_DEFAULT

		ret = av_seek_frame(self.FormatCtx,self.videoStream, 0, AVSEEK_FLAG_BACKWARD)
		if ret < 0:
			raise IOError("Error rewinding stream after fast indexing: %d" % ret)		
		avcodec_flush_buffers(self.CodecCtx)
		
	def GetFrameTime(self,float timestamp):
		cdef int64_t targetPts
		targetPts = timestamp * AV_TIME_BASE
		return self.GetFramePts(targetPts)
		
	def GetFramePts(self,int64_t pts):
		cdef int ret
		cdef int64_t myPts
		cdef AVStream *stream
		cdef int64_t targetPts,scaled_start_time
		
		stream = self.FormatCtx.streams[self.videoStream]

		scaled_start_time = av_rescale(stream.start_time,AV_TIME_BASE * <int64_t>stream.time_base.num,stream.time_base.den)
		targetPts = pts + scaled_start_time

		# why doesn't this work? It should be possible to seek only the video stream
		#ret = av_seek_frame(self.FormatCtx,self.videoStream,targetPts, AVSEEK_FLAG_BACKWARD)
		ret = av_seek_frame(self.FormatCtx,-1,targetPts, AVSEEK_FLAG_BACKWARD)
		if ret < 0:
			raise IOError("Unable to seek: %d" % ret)
		avcodec_flush_buffers(self.CodecCtx)
		
		# if we hurry it we can get bad frames later in the GOP
		self.CodecCtx.skip_idct = AVDISCARD_BIDIR
		self.CodecCtx.skip_frame = AVDISCARD_BIDIR
		
		#self.CodecCtx.hurry_up = 1
		hurried_frames = 0
		while True:
			myPts = self.__next_frame()
			if myPts >= targetPts:
				break

		#self.CodecCtx.hurry_up = 0
		
		self.CodecCtx.skip_idct = AVDISCARD_DEFAULT
		self.CodecCtx.skip_frame = AVDISCARD_DEFAULT
		return self.GetCurrentFrame()
			
	def GetFrameNo(self, int frame_no):
		cdef int ret,steps,i
		cdef int64_t myPts
		cdef float my_timestamp
		cdef float frame_rate
		cdef AVStream *stream
		
		stream = self.FormatCtx.streams[self.videoStream]
		#if self.keyframes is None:
			# no index at all, so figure out the pts from the frame rate and frame_no
			# this seems to be accurate enough for my MPEGs, so I'm not sure its worth
			# it to index the stream at all
			# 
		# OK, I'm going to go with this as the only implementation until I find
		# a reason to do it anyother way
			
		frame_rate = (<float>stream.r_frame_rate.num / <float>stream.r_frame_rate.den)
		my_timestamp = frame_no / frame_rate
		return self.GetFrameTime(my_timestamp)
			
		
		if self.index is None:
			# we don't have a full index, so we'll have to fake it from the keyframes
			index = frame_no
			steps = 0
			while index not in self.keyframes:
				index = index - 1
				steps = steps + 1
				
			ret = av_seek_frame(self.FormatCtx, self.videoStream, self.keyframes[index], AVSEEK_FLAG_BACKWARD)
			avcodec_flush_buffers(self.CodecCtx)
			for i from 0 <= i < steps:
				myPts = self.__next_frame()
			return self.GetCurrentFrame() 
		else:
			# use the full index here, I deleted the code but don't seem to need it anyway
			pass
