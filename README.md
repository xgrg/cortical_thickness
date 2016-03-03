# Cortical Thickness

### Usage

```sh
$ git clone git@github.com:xgrg/cortical_thickness.git
$ cd cortical_thickness
$ ./thickness.py --int /path/to/innersurf.gii --ext /path/to/outersurf.gii --thickness /path/to/thicknessmap.gii
$ ./compare_maps.py --m1 /path/to/firstmesh.gii --m2 /path/to/secondmesh.gii --t1 /path/to/firstmap.gii --t2 /path/to/secondmap.gii --diff /path/to/diffmap.gii --dist /path/to/distmap.gii
```

### Requirements
- nibabel, gdist, numpy, pickle (optional)
### Version
still in development




