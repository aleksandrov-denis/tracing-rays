import math
import helperclasses as hc
import glm
import igl
import numpy as np

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

epsilon = 10 ** (-4)


class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        d = ray.direction/np.linalg.norm(ray.direction)
        o = ray.origin
        t = glm.dot(self.center - o, d)
        p = ray.getPoint(t)
        y = glm.length(self.center - p)
        if (y < self.radius):
            #print("HERE")
            x = math.sqrt(self.radius**2 - y**2)
            t1 = t - x
            t2 = t + x
            if (ray.getDistance(ray.getPoint(t1)) < ray.getDistance(ray.getPoint(t2))):
                t = t1
            else:
                t = t2
            intersect.time = t
            intersect.position = ray.getPoint(t)
            intersect.normal = (intersect.position - self.center)/np.linalg.norm(intersect.position - self.center)
            intersect.mat = self.materials[0]
            intersect.hit = True
        return intersect


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # TODO: Create intersect code for Plane
        epsilon = 0.01
        p = ray.origin
        d = ray.direction
        n = self.normal/np.linalg.norm(self.normal)
        a = self.point
        # checks for near-parallelism
        #if (abs(glm.dot(d, n)) < epsilon):
            #return intersect
        t = abs(glm.dot(a - p, n)/glm.dot(d, n))
        x = ray.getPoint(t)
        on_plane = glm.dot(x - a, n)
        # checks if on plane with epsilon margin of error
        if (abs(on_plane) <= epsilon):
            intersect.time = t
            intersect.position = x
            intersect.normal = n
            # deal with material
            x = math.ceil(intersect.position.x)
            z = math.ceil(intersect.position.z)
            if (x + z) % 2 == 0:
                intersect.mat = self.materials[0]
            else:
                intersect.mat = self.materials[1]
            intersect.hit = True
        return intersect


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Cube


class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Mesh


class Hierarchy(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], t: glm.vec3, r: glm.vec3, s: glm.vec3):
        super().__init__(name, gtype, materials)
        self.t = t
        self.M = glm.mat4(1.0)
        self.Minv = glm.mat4(1.0)
        self.make_matrices(t, r, s)
        self.children: list[Geometry] = []

    def make_matrices(self, t: glm.vec3, r: glm.vec3, s: glm.vec3):
        self.M = glm.mat4(1.0)
        self.M = glm.translate(self.M, t)
        self.M = glm.rotate(self.M, glm.radians(r.x), glm.vec3(1, 0, 0))
        self.M = glm.rotate(self.M, glm.radians(r.y), glm.vec3(0, 1, 0))
        self.M = glm.rotate(self.M, glm.radians(r.z), glm.vec3(0, 0, 1))
        self.M = glm.scale(self.M, s)
        self.Minv = glm.inverse(self.M)
        self.t = t

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Hierarchy
