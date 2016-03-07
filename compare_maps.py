#!/usr/bin/env python
import nibabel
import numpy as np
from numpy import linalg
import nibabel.gifti.giftiio as gio
import pickle, string
import copy
from nibabel import gifti


def compare_node(n1, m1, m2, t1, t2):
    n2 = m1.closest_node(n1, m2)
    return t1[n1] - t2[n2], linalg.norm(m1.vertex[n1] - m2.vertex[n2])


def main(args):
    ''' Reads both meshes, finds associations between nodes from both meshes
    and writes :
    - a distance map between each pair of associated nodes
    - a difference map between texture values for each pair of nodes
    If n is provided, stops the process after n random nodes'''
    import thickness as t
    import random
    m1 = t.Mesh(args.m1)
    m2 = t.Mesh(args.m2)
    t1 = gio.read(args.t1).darrays[0].data
    t2 = gio.read(args.t2).darrays[0].data

    diff = [-1] * len(m1.vertex)
    dist = [-1] * len(m1.vertex)

    if args.n:
       for i in xrange(int(args.n)):
          if i%1000==0: print i, '/', int(args.n)
          r = random.randint(0, len(m1.vertex))
          diff[r], dist[r] = compare_node(r, m1, m2, t1, t2)
    else:
       for r in xrange(len(m1.vertex)):
          if r%1000==0: print r, '/', len(m1.vertex)
          diff[r], dist[r] = compare_node(r, m1, m2, t1, t2)

    gda = gifti.GiftiDataArray.from_array(np.array(diff), intent=1001)
    g = gifti.GiftiImage(darrays=[gda])
    gio.write(g, args.difffp)

    gda = gifti.GiftiDataArray.from_array(np.array(dist), intent=1001)
    g = gifti.GiftiImage(darrays=[gda])
    gio.write(g, args.distfp)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
      description='''Reads both meshes, finds associations between nodes (closest neighbours)
      from both meshes and writes :
    - a distance map between each pair of associated nodes
    - a difference map between texture values for each pair of nodes
    Both maps have to be visualized over the first mesh.
    If n is provided, stops the process after n random nodes (and fills the rest with -1)''')
    parser.add_argument("--m1", help="First mesh", dest='m1', type=str, required=True)
    parser.add_argument("--m2", help="Second mesh", dest='m2', type=str, required=True)
    parser.add_argument("--t1", help="First texture", dest='t1', type=str, required=True)
    parser.add_argument("--t2", help="Second texture", dest='t2', type=str, required=True)
    parser.add_argument("-n", help="Number of nodes", dest='n', type=int, required=False)

    parser.add_argument("--dist", help="(output) Distance map between both meshes", dest='distfp', type=str, required=True)
    parser.add_argument("--diff", help="(output) Difference map between both textures", dest='difffp', type=str, required=True)

    args = parser.parse_args()
    main(args)
