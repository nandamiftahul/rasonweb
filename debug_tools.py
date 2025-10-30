# rason_backend/debug_tools.py

from core.bufr_parser import parse_bufr, decode_bufr
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(initialdir="uploads/test") # show an "Open" dialog box and return the path to the selected file
print(filename)
datas = decode_bufr(filename)
#print(datas)
pd = parse_bufr(datas, site="terrindo")

