import math

import glm
import numpy as np

import geometry as geom
import helperclasses as hc

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

        #print("num rays: " + str(self.samples))
        #print("num lights: " + str(len(self.lights)))
        #print("num objects: " + str(len(self.objects)))
        
        #ray = hc.Ray(glm.vec3(self.position), w)
        max_i = 0
        max_j = 0
        max_z = 0
        for i in range(self.width):
            for j in range(self.height):
                colour = glm.vec3(0, 0, 0)

                # TODO: Generate rays
                # perspective
                x = left + (right - left) * (i + 0.5) / self.width
                y = bottom + (top - bottom) * (j + 0.5) / self.height
                ray_dir = -w*d + x * u + y * v
                ray = hc.Ray(self.position, ray_dir)

                # TODO: Test for intersection
                for obj in self.objects:
                    intersection = obj.intersect(ray, hc.Intersection.default())
                    # deal w/ intersection
                    # GET THE SHADING, USE intersection.mat:
                    # has diffuse and specular lighting components
                    # and has the shininess --> called hardness
                    colour = glm.vec3(0, 0, 0)
                    if (intersection.hit):
                        La = self.ambient
                        Ld = glm.vec3(0, 0, 0)
                        Ls = glm.vec3(0, 0, 0)
                        for light in self.lights:
                            Ld += 0.13 * intersection.mat.diffuse * max(0, glm.dot(intersection.normal, light.vector))
                            h = (cam_dir + light.vector)/np.linalg.norm(cam_dir + light.vector)
                            Ls += light.power * intersection.mat.specular * max(0, glm.dot(intersection.normal, h)) ** intersection.mat.hardness
                        colour = La + Ld + Ls
                # TODO: Perform shading computations on the intersection point

                image[i, j, 0] = max(0.0, min(1.0, colour.x))
                image[i, j, 1] = max(0.0, min(1.0, colour.y))
                image[i, j, 2] = max(0.0, min(1.0, colour.z))
        return image
