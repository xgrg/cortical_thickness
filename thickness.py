#!/usr/bin/env python

import nibabel
import numpy as np
from numpy import linalg
import os.path as osp
import nibabel.gifti.giftiio as gio
from nibabel import gifti
import string
import copy

class Mesh():
    def __init__(self, fp):
        s = gio.read(fp)
        self.surface = s
        self.vertex = s.darrays[0].data
        self.face = s.darrays[1].data
        self.compute_neighbours()
        self.compute_normals()

    def write(self, fp):
        gv = gifti.GiftiDataArray.from_array(self.vertex, intent=1008)
        gf = gifti.GiftiDataArray.from_array(self.face, intent=1009)
        g = gifti.GiftiImage()
        g.add_gifti_data_array(gv)
        g.add_gifti_data_array(gf)
        gio.write(g, fp)

    def compute_area(self):
        area = 0
        for f in self.face:
           a = self.vertex[f[1]] - self.vertex[f[0]]
           b = self.vertex[f[2]] - self.vertex[f[0]]
           c = np.cross(a, b)
           area = area + np.sqrt((c ** 2).sum())
        return area

    def compute_neighbours(self):
        self.neighbours = {}
        for each in self.face:
            assert(len(each) == 3)
            for v in each:
                c = set(each)
                c.remove(v)
                self.neighbours.setdefault(v, set()).update(c)

    def neighbours_order(self, order=4, save_to_file=None, load_from_file=None):
        if load_from_file:
            import pickle
            print 'loading neighbours from %s'%load_from_file
            return pickle.load(open(load_from_file))
        n = self.neighbours
        n4 = copy.deepcopy(self.neighbours)
        for j in range(1, order):
            print 'computing %sth order'%(str(j+1))
            n5 = copy.deepcopy(n4)
            for i in xrange(len(n4.items())):
                for e1 in n4[i]:
                    n5[i] = n5[i].union(n[e1])

        if save_to_file:
            import pickle
            print 'saving to file %s'%save_to_file
            pickle.dump(n5, open(save_to_file, 'w'))
        return n5

    def neighbours_maxdist(self, maxdist=5, save_to_file=None, load_from_file=None):
        if load_from_file:
            import pickle
            print 'loading neighbours from %s'%load_from_file
            return pickle.load(open(load_from_file))

        import gdist
        print 'computing neighbours'
        n = []
        src = len(self.vertex) * [0]
        j = -1
        for i, v in enumerate(self.vertex):
            if i%1000 == 0: print i, '/', len(self.vertex)
            if j != -1:
                src[j] = 0
            src[i]  = 1
            j = i
            distmap = gdist.compute_gdist(np.array(self.vertex, dtype=np.float64), self.face, np.array(src, dtype=np.int32), max_distance=maxdist)
            n.append(list(np.where(distmap<maxdist)[0]))

        if save_to_file:
            import pickle
            print 'saving to file %s'%save_to_file
            pickle.dump(n, open(save_to_file, 'w'))

        return n

    def compute_normals(self):
        # compute the normal for each triangle
        norms = np.zeros((len(self.vertex), 3))
        for triangle in self.face:
            sa, sb, sc = triangle
            a = self.vertex[sb] - self.vertex[sa]
            b = self.vertex[sc] - self.vertex[sa]
            norm = np.cross(a, b)
            norms[sa] += norm
            norms[sb] += norm
            norms[sc] += norm

        # normalize the normal at each vertex
        eps = 1.e-15
        self.normal = (norms.T / np.sqrt(eps + np.sum(norms ** 2, 1))).T


    def closest_node(self, i, mesh):
        ''' Returns the closest node (euclidean distance) on a mesh from node i.'''
        a1 = np.array(len(mesh.vertex) * list(self.vertex[i])).reshape((len(mesh.vertex),3))
        dist = linalg.norm(mesh.vertex - a1, axis=1)
        return np.argmin(dist)


    def matching_node(self, i, mesh, searchzone, dw=1, nw=1):
        ''' Looks for the closest point on a target surface 'mesh' from a given node index 'i'.
        A searchzone made of nodes helps speeding up the operation.
        The difference with self.closest_node() is that there is a small constraint
        which drives the matching node to be following the normal direction
        dw and nw allow to weigh preferably on distance (dw) or normal (nw) criterion.
         '''

        def dist_w(d, mindist, maxdist):
            w = 1.0 - (d - mindist) / (maxdist - mindist)
            return w

        def scal_w(s, minscal, maxscal):
            w = (s - minscal) / (maxscal - minscal)
            return w

        searchzone = list(searchzone)
        dots = [np.dot(self.normal[i], mesh.normal[each]) for each in searchzone]

        dist = [linalg.norm(self.vertex[i] - mesh.vertex[each]) for each in searchzone]

        minscal, maxscal = min(dots), max(dots)
        mindist, maxdist = min(dist), max(dist)
        weights = [dw * dist_w(dist[e], mindist, maxdist) + nw * scal_w(dots[e], minscal, maxscal) for e in xrange(len(searchzone))]
        best = searchzone[weights.index(max(weights))]

        return best

    def closest_point_on_triangle(self, i, mesh, searchzone, dw=1, nw=1):

        index = self.matching_node(i, mesh, searchzone, dw=dw, nw=nw)
        proj = []
        for i1 in mesh.neighbours[index]:
            for i2 in mesh.neighbours[index]:
                if i1 > i2:
                    p, is_inside = is_inside_triangle(self.vertex[i], [index, i1, i2], mesh)
                    if is_inside:
                        proj.append(p)

        if len(proj) != 0:
            dist = [linalg.norm(self.vertex[i] - e) for e in proj]
            best = proj[dist.index(min(dist))]
            return (best, index)
        else:
            return (mesh.vertex[index], index)



def vertices_around(p, mesh, maxdist=10.0):
    ''' Returns a set of vertices around a given position.'''
    a1 = np.array(len(mesh.vertex) * list(p)).reshape((len(mesh.vertex),3))
    dist = linalg.norm(mesh.vertex - a1, axis=1)
    return list(np.where(dist<maxdist)[0])

def is_inside_triangle(p, face, mesh):
    u = mesh.vertex[face[1]] - mesh.vertex[face[0]]
    v = mesh.vertex[face[2]] - mesh.vertex[face[0]]
    w = p - mesh.vertex[face[0]]

    uu = np.dot(u,u)
    uv = np.dot(u,v)
    vv = np.dot(v,v)
    wu = np.dot(w,u)
    wv = np.dot(w,v)
    d = uv * uv - uu * vv
    invD = 1.0 / d
    s = (uv * wv - vv * wu) * invD
    t = (uv * wu - uu * wv) * invD

    if s < 0 or s > 1:
        return ([0,0,0], False)
    if t < 0 or (s+t) > 1:
        return ([0,0,0], False)

    res = np.array(mesh.vertex[face[0]])
    u *= s
    v *= t
    res += u
    res += v
    return (res, True)




def build_median(im, em, order=4, neighbours_file=None, dw=1, nw=1):
    import gdist

    print 'computing neighbours list'
    att = {'order': order}
    if neighbours_file:
       if osp.isfile(neighbours_file):
          att['load_from_file'] = neighbours_file
       else:
          att['save_to_file'] = neighbours_file
    n4_ext = em.neighbours_order(**att)

    intcorr = {} # matching vertices from inner mesh to outer mesh
    mm = copy.deepcopy(im)
    matching_mesh = copy.deepcopy(im)
    curr = 0
    processed = set()
    current = [curr]

    sz = vertices_around(im.vertex[curr], em, maxdist=20.0)
    matching_pt, intcorr[curr] = im.closest_point_on_triangle(curr, em, searchzone=sz, dw=dw, nw=nw)

    # compute distance map
    print 'computing distance map'
    src = len(im.vertex) * [0]
    src[curr]  = 1
    src = np.array(src, dtype=np.int32)
    distmap = gdist.compute_gdist(np.array(im.vertex, dtype=np.float64), im.face, src)

    # init thickness map
    matching_mesh.vertex[curr] = matching_pt #em.vertex[intcorr[curr]]
    thickness = len(im.vertex) * [-1.0]
    thickness[curr] = linalg.norm(im.vertex[curr] - em.vertex[intcorr[curr]])

    print 'propagating front...'
    nofound = 0
    while len(processed) < len(im.vertex):
        if len(processed) % 100 == 0:
            print len(processed), '/', len(im.vertex), '(%s)'%len(current), 'nofound:', nofound

        dm = [distmap[e] for e in current]
        best = current[dm.index(min(dm))]
        processed.add(best)
        curr = best
        current.remove(best)

        neighbours = [e for e in im.neighbours[curr] if not (e in current or e in processed)]

        for e in neighbours:
            matching_pt, res = im.closest_point_on_triangle(e, em, searchzone = n4_ext[intcorr[curr]], dw=dw, nw=nw)

            dist = linalg.norm(im.vertex[e] - em.vertex[res])
            dot = np.dot(em.vertex[res] - im.vertex[e], im.normal[e])

            # recompute if the closest point is too distant with the initial searchzone
            if dist > 5.0 or dot < 0.0:
                nofound += 1
                sz = vertices_around(im.vertex[e], em, maxdist=20.0)
                matching_pt, res = im.closest_point_on_triangle(e, em, searchzone = sz, dw=dw, nw=nw)

            matching_mesh.vertex[e] = matching_pt
            thickness[e] = linalg.norm(im.vertex[e] - matching_pt)
            intcorr[e] = res
            current.append(e)

    mm.vertex = (im.vertex + matching_mesh.vertex)*0.5
    return mm, thickness, matching_mesh



def main(args):
    im = Mesh(args.int)
    em = Mesh(args.ext)
    mm, thickness, matching_mesh = build_median(im, em, args.order, neighbours_file = '/tmp/%s.neighbours.pickle'%osp.basename(args.ext))
    ea = em.compute_area()
    ia = im.compute_area()
    print 'area: int:', ia, 'ext:', ea
    print 'ratio e/i:', ea/ia

    gda = gifti.GiftiDataArray.from_array(np.array(thickness), intent=1001)
    g = gifti.GiftiImage(darrays=[gda])
    gio.write(g, args.thickness)

    mm.write(args.mid)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute cortical thickness between two meshes without prior node correspondence')
    parser.add_argument("--int", help="internal cortical surface", dest='int', type=str, required=True)
    parser.add_argument("--ext", help="external cortical surface", dest='ext', type=str, required=True)
    parser.add_argument("--thickness", help="thickness map", dest='thickness', type=str, required=True)
    parser.add_argument("--mid", help="central surface", dest='mid', type=str, required=True)
    parser.add_argument("-O", help="max order for search zones", dest='order', type=int, default=4, required=False)
    parser.add_argument("--dw", help="weight on distances", dest='dw', type=float, default=1, required=False)
    parser.add_argument("--nw", help="weight on normals", dest='nw', type=float, default=1, required=False)

    args = parser.parse_args()
    main(args)
