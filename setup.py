##
##
##

import sys
import os

sys.stderr.write("A complete rewrite of pyffmpeg is available as beta, you may choose to using the setup.py of the newversion_beta directory.\n")
sys.stderr.write("This setup is using pyffmpeg stable by default\n")
os.chdir("stable")

import stable.setup

