import os
import hashlib
import argparse
import pathlib
import numpy as np
import sqlite3
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_db(filename):
    if filename:
        try:
            return sqlite3.connect(filename)
        except Exception as e:
            print(e)
            exit(1)
    else:
        try:
            return sqlite3.connect(':memory:')
        except Exception as e:
            print(e)
            exit(1)

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
                meta[fname] = gen_hashes(img.read())
                img.close()
            except IOError as e:
                print("Unable to open image file")
                print(e)
                
    files = pd.DataFrame.from_dict(meta, orient='index', columns=['md5','else'])
    #list(meta.values())
    #print(files)
    md5clean = files.drop_duplicates(subset='md5')
    #print(md5clean)
    print(len(files), 'Files counted')
    print(len(md5clean), 'unique md5 hashes')
    #print(md5clean.index.values)
    with open("unique.txt", "w") as f:
        f.write('\n'.join(md5clean.index.values))

def show_dupes(db):
    idk = db.execute('SELECT md5 FROM (SELECT md5, COUNT(md5) FROM images GROUP BY md5 HAVING COUNT(md5)>1)')
    for row in idk.fetchall():
        db.execute('SELECT name FROM images WHERE md5=?', row)
        dupes = db.fetchall()
        print(dupes)
        for i in dupes:
            img = Image.open(os.path.join(args.img, i[0]))
            #img.show()

def main(args):
    # purely to uniqify a single folder of images 
    con = get_db(args.meta)
    db = con.cursor()
    db.execute('''CREATE TABLE images (name TEXT, md5 TEXT UNIQUE, hash TEXT)''')
    flist = os.listdir(args.img)
    with tqdm(total=len(flist)) as pbar:
        for fname in flist:
            path = os.path.join(args.img, fname);
            try:
                img = Image.open(path)
                fimg = open(path, 'rb')
            except IOError as e:
                print("Failed to open file: %s" % path)
                continue
            img_hash = imagehash.dhash(img)
            md5 = hashlib.md5(img.tobytes()).hexdigest()
            #md5 = hashlib.md5(fimg.read()).hexdigest()
            try:
                db.execute('INSERT INTO images VALUES (?, ?, ?)', (fname, md5, str(img_hash)))
            except sqlite3.IntegrityError as e:
                pass
            pbar.update(1)
    con.commit()
    db.execute('SELECT COUNT(*) FROM images')
    n_img = db.fetchone()
    print(n_img[0], 'Different MD5')
    if args.out:
        outf = open(args.out, 'w')
        db.execute('SELECT * FROM images')
        for row in db:
            print(row[0], file=outf)
            
    
    
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
    parse.add_argument('-out', 
        help='Path to output file',
        type=str,
        required=False)
    args = parse.parse_args()
    #print(args.meta)
    #print(args.img)
    main(args)
    #make_unique(args.meta, args.img)
