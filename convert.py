#! /usr/bin/env python
import nibabel.gifti.giftiio as gio
from nibabel import gifti
from plyfile import PlyData
import numpy as np
import argparse

def convert(inputfp, outputfp):
    print 'converting %s to %s'%(inputfp, outputfp)
    plydata = PlyData.read(inputfp)
    vertex = np.array([each for each in plydata.elements[0].data.tolist()], dtype=np.float32)
    faces = plydata.elements[1].data.tolist()
    faces = np.array([each[0] for each in faces], dtype=np.int32)
    gv = gifti.GiftiDataArray.from_array(vertex, intent=1008)
    gf = gifti.GiftiDataArray.from_array(faces, intent=1009)
    g = gifti.GiftiImage()
    g.add_gifti_data_array(gv)
    g.add_gifti_data_array(gf)
    gio.write(g, outputfp)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert a ply to mesh')
    parser.add_argument("-i", dest='inputfp', type=str, required=True)
    parser.add_argument("-o", dest='outputfp', type=str, required=True)

    args = parser.parse_args()
    convert(args.inputfp, args.outputfp)
