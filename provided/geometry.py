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
        self.epsilon = 0.005

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        #epsilon = 0.005
        d = ray.direction
        e = ray.origin
        discriminant = glm.dot(d, e - self.center)**2 - glm.dot(d, d)*(glm.dot(e - self.center, e - self.center) - self.radius**2)
        if (discriminant < 0):
            return intersect
        t1 = (glm.dot(-d, e - self.center) + math.sqrt(discriminant))/glm.dot(d, d)
        t2 = (glm.dot(-d, e - self.center) - math.sqrt(discriminant))/glm.dot(d, d)
        if (ray.getDistance(ray.getPoint(t1)) < ray.getDistance(ray.getPoint(t2))):
            t = t1
        else:
            t = t2
        point = ray.getPoint(t)
        on_sphere = glm.dot(d, d)*(t**2) + glm.dot(2*d, e - self.center)*t + glm.dot(e - self.center, e - self.center) - self.radius**2
        if (abs(on_sphere) < self.epsilon):
            intersect.time = t
            intersect.position = point
            intersect.normal = (point - self.center)/self.radius
            intersect.mat = self.materials[0]
            if t > 0 + self.epsilon:
                intersect.hit = True
        return intersect


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # TODO: Create intersect code for Plane
        #epsilon = 0.01
        e = ray.origin
        d = ray.direction
        n = self.normal
        p1 = self.point
        t = abs(glm.dot(p1 - e, n)/glm.dot(d, n))
        p = ray.getPoint(t)
        on_plane = glm.dot(p - p1, n)
        # checks if on plane with epsilon margin of error
        if (abs(on_plane) <= self.epsilon):
            intersect.time = t
            intersect.position = p
            intersect.normal = n
            # deal with material
            x = math.ceil(p.x)
            z = math.ceil(p.z)
            if (x + z) % 2 == 0:
                intersect.mat = self.materials[0]
            else:
                intersect.mat = self.materials[1]
            if t > 0 + self.epsilon:
                intersect.hit = True
        return intersect


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def getLowHigh(self, _min_, _max_, p, d):
        if d == 0:
            return 0.0, 0.0
        tvarmin = (_min_ - p)/d
        tvarmax = (_max_ - p)/d
        return min(tvarmin, tvarmax), max(tvarmin, tvarmax)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # TODO: Create intersect code for Cube
        txlow, txhigh = self.getLowHigh(self.minpos.x, self.maxpos.x, ray.origin.x, ray.direction.x)
        tylow, tyhigh = self.getLowHigh(self.minpos.y, self.maxpos.y, ray.origin.y, ray.direction.y)
        tzlow, tzhigh = self.getLowHigh(self.minpos.z, self.maxpos.z, ray.origin.z, ray.direction.z)
        tmin = max(txlow, tylow, tzlow)
        tmax = min(txhigh, tyhigh, tzhigh)
        if abs(tmin) < 0 + self.epsilon or abs(tmax) < 0 + self.epsilon:
            return intersect
        if tmax > tmin + self.epsilon:
            intersect.time = tmin
            p = ray.getPoint(tmin)
            intersect.position = p
            nx = glm.vec3(1, 0, 0)
            ny = glm.vec3(0, 1, 0)
            nz = glm.vec3(0, 0, 1)
            if abs(p.x - self.maxpos.x) < self.epsilon:
                n = nx
            elif abs(p.x - self.minpos.x) < self.epsilon:
                n = -nx
            elif abs(p.y - self.maxpos.y) < self.epsilon:
                n = ny
            elif abs(p.y - self.minpos.y) < self.epsilon:
                n = -ny
            elif abs(p.z - self.maxpos.z) < self.epsilon:
                n = nz
            elif abs(p.z - self.minpos.z) < self.epsilon:
                n = -nz
            intersect.normal = n
            intersect.mat = self.materials[0]
            intersect.hit = True
        return intersect

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

    def flatten(self):
        children = []
        new_M = self.M
        new_Minv = self.Minv
        for geometry in self.children:
            if geometry.gtype != "node":
                children.append(geometry)
            else:
                next_children, next_M, next_Minv = geometry.flatten()
                for child in next_children:
                    new_M = next_M @ self.M
                    new_Minv = next_Minv @ self.Minv
                    children.append(child)
        """self.children = children
        self.M = new_M
        self.Minv = new_Minv"""
        return children, new_M, new_Minv

    def intersect(self, ray: hc.Ray, closest: hc.Intersection):
        # flatten the tree
        self.children, self.M, self.Minv = self.flatten()
        local_ray = hc.Ray(self.Minv @ ray.origin, self.Minv @ ray.direction)
        distance = float('inf')
        for child in self.children:
            if child.gtype != "node":
                # do intersection
                local_intersect = child.intersect(local_ray, hc.Intersection.default())
                # and keep track of closest intersection
                if local_intersect.hit:
                    new_distance = ray.getDistance(local_intersect.position)
                    # transform the normal and position
                    local_intersect.normal = glm.transpose(self.Minv) @ local_intersect.normal
                    local_intersect.position = self.M @ local_intersect.position
                    if new_distance < distance:
                        distance = new_distance
                        closest = local_intersect
            else:
                print("SHOULD NOT BE ANY NODES")
        return closest

    """def intersect(self, ray: hc.Ray, closest: hc.Intersection):
        # generate new ray
        local_ray = hc.Ray(self.Minv @ ray.origin, self.Minv @ ray.direction)
        distance = float('inf')
        for geometry in self.children:
            if geometry.gtype != "node":
                # do ray transformation on non-node obj
                local_intersect = geometry.intersect(local_ray, hc.Intersection.default())
                # and keep track of closest intersection
                new_distance1 = ray.getDistance(local_intersect.position)
                if local_intersect.hit:
                    # transform the normal and position
                    local_intersect.normal = glm.transpose(self.Minv) @ local_intersect.normal
                    local_intersect.position = self.M @ local_intersect.position
                    if new_distance1 < distance:
                        distance = new_distance1
                        closest = local_intersect
            else:
                child_intersect = geometry.intersect(local_ray, hc.Intersection.default())
                if child_intersect.hit:
                    # if intersection point is closer than existing, use it
                    child_intersect.normal = glm.transpose(self.Minv) @ child_intersect.normal
                    child_intersect.position = self.M @ child_intersect.position
                    new_distance2 = ray.getDistance(child_intersect.position)
                    if new_distance2 < distance:
                        distance = new_distance2
                        closest = child_intersect
                        break
        # account for material
        if closest.hit and closest.mat is None:
            closest.mat = self.materials[0]
        return closest"""
