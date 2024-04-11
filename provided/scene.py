import math

import glm
import numpy as np

import geometry as geom
import helperclasses as hc
import random
from tqdm import tqdm

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

shadow_epsilon = 10**(-6)


class Scene:

    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 ambient: glm.vec3,
                 lights: list[hc.Light],
                 materials: list[hc.Material],
                 objects: list[geom.Geometry]
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.position = position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.materials = materials  # all materials of objects in the scene
        self.objects = objects  # all objects in the scene

    def render(self):

        image = np.zeros((self.width, self.height, 3))

        cam_dir = self.position - self.lookat
        d = 1.0
        top = d * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        inf = float('inf')
        minpos = glm.vec3(inf, inf, inf)
        maxpos = glm.vec3(-inf, -inf, -inf)
        # SETUP HBV
        for obj in self.objects:
            if obj.gtype == "mesh":
                print("num triangles: " + str(len(obj.faces)))
                # calculate root bounding box
                for face in obj.faces:
                    for i in face:
                        minpos = glm.min(minpos, obj.verts[i])
                        maxpos = glm.max(maxpos, obj.verts[i])
                # have minpos and maxpos, create bbox
                bbox = geom.AABB(None, None, None, None, None)
                bbox.minpos = minpos
                bbox.maxpos = maxpos
                obj.bbox = bbox
                obj.setupHBV(0, 10)

        max_i = 0
        max_j = 0
        max_z = 0
        total_pixels = self.width * self.height
        with tqdm(total=total_pixels, desc='Rendering', unit='pixel') as pbar:
            for i in range(self.width):
                for j in range(self.height):
                    colour = glm.vec3(0, 0, 0)

                    # TODO: Generate rays
                    # perspective
                    x = left + (right - left) * (i + 0.5) / self.width
                    y = bottom + (top - bottom) * (j + 0.5) / self.height
                    ray_dir = glm.normalize(-w*d + x * u + y * v)
                    ray = hc.Ray(self.position, ray_dir)
                
                    closest = hc.Intersection.default()
                    colour = glm.vec3(0, 0, 0)
                    for cast in range(self.samples):
                        distance = float('inf')
                        # TODO: Test for intersection
                        for obj in self.objects:
                            bound = 1.0
                            #color = glm.vec3(0.0)
                            off = 0.0
                            aperture_off = 0.0
                            jitter = 1.0
                            if self.jitter:
                                jitter = random.uniform(1, 3)
                            if self.samples > 1:
                                off = jitter * (0.5 - random.uniform(-bound, bound))/2000
                                #aperture_off = (0.5 - random.uniform(-bound, bound))/20
                            sample_ray = hc.Ray(ray.origin + aperture_off, ray.direction + off)
                            intersection = obj.intersect(sample_ray, hc.Intersection.default())
                            if intersection.time < distance and intersection.hit:
                                distance = intersection.time
                                closest = intersection
                                # reflection code
                                # r = i - 2(i.dot(n))n
                                # cast the reflected ray to all objects again
                                # get closest hit and avg the colors of closest and 
                                # closest_reflection
                        # TODO: Perform shading computations on the intersection point
                        if closest.hit:
                            La = self.ambient
                            Ld = glm.vec3(0, 0, 0)
                            Ls = glm.vec3(0, 0, 0)
                            for light in self.lights:
                                # compute shadow shading
                                bound = 2
                                l = light.vector - closest.position
                                # AREA LIGHTS
                                #lv = glm.vec3(l.x + random.uniform(-bound, bound), l.y, l.z + random.uniform(-bound, bound))
                                lv = l
                                shadRay = hc.Ray(closest.position, lv)
                                for obj in self.objects:
                                    shadIntersection = obj.intersect(shadRay, hc.Intersection.default())
                                    if shadIntersection.hit:
                                        break
                                if not shadIntersection.hit:
                                    l = glm.normalize(lv)
                                    Ld += light.power * light.colour * closest.mat.diffuse * max(0, glm.dot(closest.normal, l))
                                    h = (-ray.direction + l)/np.linalg.norm(-ray.direction + l)
                                    Ls += light.power * light.colour * closest.mat.specular * max(0, glm.dot(closest.normal, h))**closest.mat.hardness
                            colour = colour + La + Ld + Ls
                        else:
                            break
                    colour = colour/self.samples
                    image[i, j, 0] = max(0.0, min(1.0, colour.x))
                    image[i, j, 1] = max(0.0, min(1.0, colour.y))
                    image[i, j, 2] = max(0.0, min(1.0, colour.z))
                    pbar.update(1)
            return image
