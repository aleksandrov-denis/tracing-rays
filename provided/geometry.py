import math
import helperclasses as hc
import glm
import igl
import numpy as np
import random

# Integrated by Denis Aleksandrov
# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

epsilon = 10 ** (-5)


class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect

class Quadric(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: glm.vec3):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius
    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        return intersection

class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
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
        if (abs(on_sphere) < epsilon):
            intersect.time = t
            intersect.position = point
            intersect.normal = (point - self.center)/self.radius
            intersect.mat = self.materials[0]
            if t > 0 + epsilon:
                intersect.hit = True
        return intersect


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # TODO: Create intersect code for Plane
        e = ray.origin
        d = ray.direction
        n = self.normal
        p1 = self.point
        t = abs(glm.dot(p1 - e, n)/glm.dot(d, n))
        p = ray.getPoint(t)
        on_plane = glm.dot(p - p1, n)
        # checks if on plane with epsilon margin of error
        if (abs(on_plane) <= epsilon):
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
            if t > 0 + epsilon:
                intersect.hit = True
        return intersect


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        if center != None and dimension != None:
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
        if tmin < 0 + epsilon:
            return intersect
        if tmax > tmin:
            intersect.time = tmin
            p = ray.getPoint(tmin)
            intersect.position = p
            nx = glm.vec3(1, 0, 0)
            ny = glm.vec3(0, 1, 0)
            nz = glm.vec3(0, 0, 1)
            if abs(p.x - self.maxpos.x) < epsilon:
                n = nx
            elif abs(p.x - self.minpos.x) < epsilon:
                n = -nx
            elif abs(p.y - self.maxpos.y) < epsilon:
                n = ny
            elif abs(p.y - self.minpos.y) < epsilon:
                n = -ny
            elif abs(p.z - self.maxpos.z) < epsilon:
                n = nz
            elif abs(p.z - self.minpos.z) < epsilon:
                n = -nz
            intersect.normal = n
            if self.materials != None:
                intersect.mat = self.materials[0]
            else:
                
                intersect.mat = hc.Material("bbox", glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 0.0), 0.0, -1)
            intersect.hit = True
        return intersect

class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        self.verts = []
        self.norms = []
        self.faces = []
        if (translate != None and scale != None and filepath != None):
            verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
            for v in verts:
                self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
            for n in norms:
                self.norms.append(glm.vec3(n[0], n[1], n[2]))
        # HBV implementation
        self.bbox = None
        self.left = None
        self.right = None
        self.depth = None

    def setupHBV(self, axis, depth):
        self.depth = depth
        if depth == 0:
            return
        self.left = Mesh(self.name, self.gtype, self.materials, None, None, None)
        self.right = Mesh(self.name, self.gtype, self.materials, None, None, None)
        self.left.name = "left"
        self.right.name = "right"
        midpoint = (self.bbox.minpos + self.bbox.maxpos)/2.0
        l_index = 0
        r_index = 0
        inf = float('inf')
        l_minpos = glm.vec3(inf, inf, inf)
        l_maxpos = glm.vec3(-inf, -inf, -inf)
        r_minpos = glm.vec3(inf, inf, inf)
        r_maxpos = glm.vec3(-inf, -inf, -inf)
        for face in self.faces:
            new_l_face = []
            new_r_face = []
            on_left = False
            num_left = 0
            for i in face:
                if self.verts[i][axis] < midpoint[axis]:
                    num_left += 1
            rand = random.randint(0, 1)
            if num_left > 0 and num_left % 2 == rand:
                on_left = True
            for i in face:
                if on_left:
                    # if vert is on left side, put whole face on the left
                    l_minpos = glm.min(l_minpos, self.verts[i])
                    l_maxpos = glm.max(l_maxpos, self.verts[i])
                    self.left.verts.append(self.verts[i])
                    new_l_face.append(l_index)
                    l_index += 1
                else:
                    r_minpos = glm.min(r_minpos, self.verts[i])
                    r_maxpos = glm.max(r_maxpos, self.verts[i])
                    self.right.verts.append(self.verts[i])
                    new_r_face.append(r_index)
                    r_index += 1
            if on_left:
                self.left.faces.append(new_l_face)
            else:
                self.right.faces.append(new_r_face)
        bbox_left = AABB(None, None, None, None, None)
        bbox_left.minpos = l_minpos
        bbox_left.maxpos = l_maxpos
        self.left.bbox = bbox_left
        bbox_right = AABB(None, None, None, None, None)
        bbox_right.minpos = r_minpos
        bbox_right.maxpos = r_maxpos
        self.right.bbox = bbox_right
        # rotate to next axis
        axis = (axis + 1) % 3
        depth -= 1
        # need a stop point
        self.left.setupHBV(axis, depth)
        self.right.setupHBV(axis, depth)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        if self.depth != 0:
            bbox_intersect = self.bbox.intersect(ray, hc.Intersection.default())
            if not bbox_intersect.hit:
                return intersect
            left_intersect = self.left.intersect(ray, intersect)
            right_intersect = self.right.intersect(ray, intersect)
            if left_intersect.hit and right_intersect.hit:
                if left_intersect.time < right_intersect.time:
                    return left_intersect
                return right_intersect
            elif left_intersect.hit:
                return left_intersect
            elif right_intersect.hit:
                return right_intersect
            else:
                return hc.Intersection.default()
        # TODO: Create intersect code for Mesh
        for face in self.faces:
            # do triangle intersection
            A = glm.vec3(self.verts[face[0]])
            B = glm.vec3(self.verts[face[1]])
            C = glm.vec3(self.verts[face[2]])
            triangle_p = (A + B + C)/3.0
            normal = glm.normalize(glm.cross(A - C, B - C))
            t = glm.dot(triangle_p - ray.origin, normal)/glm.dot(ray.direction, normal)
            ray_point = ray.getPoint(t)
            edge0 = glm.dot(glm.cross(B - A, ray_point - A), normal)
            edge1 = glm.dot(glm.cross(C - B, ray_point - B), normal)
            edge2 = glm.dot(glm.cross(A - C, ray_point - C), normal)
            if edge0 > 0 and edge1 > 0 and edge2 > 0 and t > epsilon and t < intersect.time:
                intersect.time = t
                intersect.position = ray_point
                intersect.normal = normal
                intersect.mat = self.materials[0]
                intersect.hit = True
        return intersect



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

    def intersect(self, ray: hc.Ray, past_intersect: hc.Intersection):
        local_ray = hc.Ray(glm.vec3(self.Minv * glm.vec4(ray.origin, 1.0)), glm.vec3(self.Minv * glm.vec4(ray.direction, 0.0)))
        #distance = float('inf')
        closest = hc.Intersection.default()
        local_intersect = hc.Intersection.default()
        for child in self.children:
            if child.gtype != "node":
                local_intersect = child.intersect(local_ray, hc.Intersection.default())
                if local_intersect.hit:
                    if local_intersect.time < past_intersect.time:
                        local_intersect.normal = glm.normalize(glm.vec3(glm.transpose(self.Minv) * glm.vec4(local_intersect.normal, 0.0)))
                        local_intersect.position = glm.vec3(self.M * glm.vec4(local_intersect.position, 1.0))
                        distance = local_intersect.time
                        closest = local_intersect
                    else:
                        closest = past_intersect
            else:
                if local_intersect.hit:
                    next_intersect = child.intersect(local_ray, closest)
                else:
                    next_intersect = child.intersect(local_ray, hc.Intersection.default())
                if next_intersect.hit:
                    if next_intersect.time < past_intersect.time:
                        next_intersect.normal = glm.normalize(glm.vec3(glm.transpose(self.Minv) * glm.vec4(next_intersect.normal, 0.0)))
                        next_intersect.position = glm.vec3(self.M * glm.vec4(next_intersect.position, 1.0))
                        distance = next_intersect.time
                        closest = next_intersect
                    else:
                        closest = past_intersect
        if closest.hit and closest.mat == None:
            closest.mat = self.materials[0]
        return closest
