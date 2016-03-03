# Cortical Thickness

### Usage

```sh
$ git clone git@github.com:xgrg/cortical_thickness.git
$ cd cortical_thickness
$ ./thickness.py --int /path/to/innersurf.gii --ext /path/to/outersurf.gii --thickness /path/to/thicknessmap.gii
$ ./compare_maps.py --m1 /path/to/firstmesh.gii --m2 /path/to/secondmesh.gii --t1 /path/to/firstmap.gii --t2 /path/to/secondmap.gii --diff /path/to/diffmap.gii --dist /path/to/distmap.gii
```
### Example

To run it on the provided dataset :

```sh
$ ./thickness.py --int data/lh.r.aims.white.gii --ext data/lh.r.aims.pial.gii --thickness data/lh.r.aims.thickness.gii --mid data/lh.r.aims.mid.gii
$ ./thickness.py --int data/subject1_scan01_Lwhite.gii --ext data/subject1_scan01_Lhemi.gii --thickness data/subject1_scan01_Lthickness.gii --mid data/subject1_scan01_Lmid.gii
$ ./compare_maps.py --m1 data/subject1_scan01_Lwhite.gii --m2 data/lh.r.aims.white.gii --t1 data/subject1_scan01_Lthickness.gii --t2 data/lh.r.aims.thickness.gii --diff data/diff.gii --dist data/dist.gii
```

### Requirements
- nibabel, gdist, numpy, pickle (optional)

### Version
still in development




