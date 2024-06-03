**McGill University COMP557 ray tracing project**

A ray tracer which uses the json format to define scenes

**To Run**
Running ```python provided/main.py``` will render a default scene
of two reflective spheres on a checkerboard plane to ```provided/out.png```

All of the scenes are located in ```scenes/```. Feel free to modify ```provided/main.py```
to render an alternative scene.

**Features**
- Ray generation
- Sphere intersection
- Lighting and shading
- Plane intersection
- Shadow rays
- Box intersection
- Heirarchy intersection
- Triangle mesh intersection
- Anti-aliasing and super-sampling
- Mirror reflection
    - Illustrated in mirror.png
    - Demonstrated in arealighting_antialiasing.png
- Area lights (increase sample size)
    - Demostrated in custom.png
- Depth blur
    - Demonstrated by depth_blur.png
- Hierarchichal Bounding Boxes
    - Speeds up mesh rendering significantly.
    Take my custom scene for example.
