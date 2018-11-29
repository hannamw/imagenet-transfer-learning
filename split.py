import os
import shutil
import numpy as np
import pathlib


source = pathlib.Path("scorpion")
s1 = pathlib.Path("Train", source) 

for f in s1.iterdir():
    if(not f.is_file()):
        continue
    tr = np.random.rand(1)
    if tr < 0.15:
        dest1 = pathlib.Path("data","test", source, f.name)

    elif tr < 0.3:
        dest1 = pathlib.Path("data","val", source, f.name)

    else: 
        dest1 = pathlib.Path("data","train", source, f.name)
    
    if not os.path.exists(dest1.parent):
        os.makedirs(dest1.parent)
    shutil.copy(f, dest1)
        
        
source = pathlib.Path("spider")
s1 = pathlib.Path("Train", source)
for f in s1.iterdir():
    if(not f.is_file()):
        continue
    tr = np.random.rand(1)
    if tr < 0.15:
        dest1 = pathlib.Path("data","test", source, f.name)
    elif tr < 0.3:
        dest1 = pathlib.Path("data","val", source, f.name)
    else: 
        dest1 = pathlib.Path("data","train", source, f.name)
        
    if not os.path.exists(dest1.parent):
        os.makedirs(dest1.parent)
    shutil.copy(f, dest1)
