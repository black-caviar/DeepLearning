import os
import hashlib
import argparse
import pathlib
import numpy as np  
    
def load_meta(filename):
    try:
        #f = open(filename, "rw")
        return np.load(filename, 'rw') 
    except IOError as e:
        print("Unable to open metadata file")
        print(e)
        exit()
        
def gen_hashes(data):
    md5 = hashlib.md5(data).hexdigest()
    #add in more hash types later
    return md5, md5
    
def make_unique(metafile, datapath):
    meta = {}
    if metafile:
        meta = load_meta(metafile)
    
    for fname in os.listdir(datapath):
        #print(fname)
        if fname in meta:
            continue
        else:
            try:
                path = os.path.join(datapath, fname);
                #print(path)
                img = open(path, 'rb')
                #print(gen_hashes(img.read()))
                meta[fname] = [fname, *gen_hashes(img.read())]
                img.close()
            except IOError as e:
                print("Unable to open image file")
                print(e)
                
    files = list(meta.values())
    print(files)
    print(np.array(files).shape)
    md5clean = np.unique(files, axis=1)
    print(md5clean)
    print(len(md5clean))
    print(len(files))
    
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-meta', 
        help='Path to meta file',
        type=str,
        required=False,)
    parse.add_argument('-img', 
        help='Path to image directory',
        type=str,
        required=True,)
    args = parse.parse_args()
    print(args.meta)
    print(args.img)
    make_unique(args.meta, args.img);
