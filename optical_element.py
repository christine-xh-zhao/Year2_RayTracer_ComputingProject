"""
Generate optical elements such as the convex, concave, or plano refracting
surfaces and the output plane

Calculate intercept when the rays incident onto the optical elements and
propagate the rays through them
"""

import numpy as np

import ray as ra


def Normalise(vec):
    """Normalise the input vector

    Arg:
        vec -- input vector in 3D Cartesian, stored as a np array

    Return:
        Normalised input vector as a 3D list
    """

    mag = np.linalg.norm(vec)  # magnitude of vec

    if len(vec) != 3:
        raise Exception('vector requires three coordinates')
    else:
        if mag != 0:  # mag = 0 cannot be divided
            return vec/mag
        else:
            return vec


def Snell(k1, surf_nor, n1, n2):
    """Refract the incident ray using Snell's law

    Args:
        k1 -- incident ray direction
        surf_nor -- surface normal vector
        n1, n2 -- refractive index on the incident and refracted side
                  respectively of the surface

    Return:
        k2_hat -- normalised direction vector of the reflected ray as a list
    """

    surf_nor_hat = Normalise(surf_nor)
    k1_hat = Normalise(k1)

    # calculate incident angle, angle1, to the value between 0 and pi/2
    cos_angle1 = np.dot(k1_hat, surf_nor_hat)
    if cos_angle1 >= 0:
        angle1 = np.arccos(cos_angle1)

        # to obey the convention of Snell's law (vector form)
        surf_nor_hat = -surf_nor_hat  # direction of surf_nor_hat is inversed
    else:
        angle1 = np.arccos(-cos_angle1)

    # check total internal reflection
    if np.sin(angle1) > (n2/n1):
        print('\nTotal internal reflection happens')
        return None
    else:
        # obtain reflected ray direction, k2, via vector form of Snell's law
        # (reference for formulae: https://en.wikipedia.org/wiki/Snell%27s_law)
        r_n = n1/n2
        c = np.cos(angle1)
        k2 = r_n*k1_hat + surf_nor_hat*(r_n*c - np.sqrt(1 - r_n**2 * (1-c**2)))

        k2_hat = Normalise(k2)
        return k2_hat


def paraxial_focus(bundle_list, element_list, print_val=True):
    """Calculate the position coordinate of the resultant paraxial focus for
       the optical elements

    By propagating a ray very close to the optical axis (e.g., 0.01mm) through
    the optical elements (but not the final output plane) and calculate its
    intersecpt with the optical axis, the position of the paraxial focus would
    be estimated

    Arg:
        element_list -- a list of objects representing the optical elements;
                        each object has SphericalRefraction() as its instance;
                        the final output plane should not be included in this
                        list
        bundle_list -- a list of ray bundles that will be propagated through
                       element_list, which has the instance of the RayBundle
                       class
        print_val -- boolean variable that controls whether printing out the
                     paraxial focal length value or not

    Return:
        The position coordinate of where the paraxial focus is at
    """

    if isinstance(element_list, list) is False:
        raise TypeError('element_list has to be a list of optical element '
                        + 'objects, e.g. [element] or [element1, element2]')

    for element in element_list:
        if isinstance(element, OutputPlane) is True:
            raise Exception('the output plane should not be included in the '
                            + 'element_list')
        else:
            pass

    # extract the surface centre coordinate of the 1st optical element
    surf_cen_1 = element_list[0].surface_centre()

    # define the inital position and direction of the possible test rays
    p_test_1 = [surf_cen_1[0]+0.01, surf_cen_1[1], surf_cen_1[2]-10]
    k_test_1 = [0, 0, 1]
    p_test_2 = [surf_cen_1[0]+0.01, surf_cen_1[1], surf_cen_1[2]+10]
    k_test_2 = [0, 0, -1]

    # use the z direction of the first ray bundle in the bundle list to
    # determine the z direction of the test ray
    bund_z_direc = bundle_list[0].k_init()[2]
    if bund_z_direc == 0:
        raise Exception('the z direction of the ray bundle has to be either a '
                        + 'positive or a negative value, not 0')
    # the test ray has to incident the lens at the same side as the ray bundle
    elif bund_z_direc > 0:
        test_ray = ra.Ray(p_test_1, k_test_1)
    else:
        test_ray = ra.Ray(p_test_2, k_test_2)

    # propagate ray through optical element
    for element in element_list:
        element.propagate_ray(test_ray)

    # extract the final position and direction coordinate of the ray
    x_final_intercept = test_ray.p()[0] - surf_cen_1[0]
    z_final_intercept = test_ray.p()[2]
    final_direction = test_ray.k()

    # calculate the focal length from basic geometry
    focal_length = x_final_intercept * (final_direction[2]
                                        / final_direction[0])

    # if the optical elements do not focus or spread out the rays
    if final_direction[0] == 0:  # according to the direction of x coordinate
        print('\nParaxial focus cannot be calculated as the rays are '
              + 'neither focused nor spreaded out')
        return None

    else:
        # if the optical elements focus the rays
        if final_direction[0] < 0:
            z_focus = z_final_intercept - focal_length

            if print_val is True:
                print('\nParaxial focus for the convex lens is %s mm' %
                      (abs(z_focus - surf_cen_1[2])))
            else:
                pass

        # if the optical elements spread out the rays
        elif final_direction[0] > 0:
            z_focus = z_final_intercept + focal_length

            if print_val is True:
                print('\nParaxial focus for the concave lens is %s mm' %
                      (abs(z_focus - surf_cen_1[2])))
            else:
                pass

        return z_focus


class SphericalRefraction:
    """Simulate a spherical refracting surface

    Args:
        surf_cen -- position coordinate of the surface centre; the default
                    position is set to the origin (i.e. [0, 0, 0])
        cur -- curvature of the surface, which is 1/radius for spherical
               surfaces; cur > 0 for convex surface, cur < 0 for concave
               surface, and cur = 0 for plano surfaces
        n1 -- the refractive index on the side that the rays incident onto the
              surface
        n2 -- the refractive index on the side that the rays leave the surface
        ap -- aperture radius, the maximum extent of the surface from the
              optical axis; its input value will enlarged by 1e-10 to avoid
              that, when determining the existance of the intercept ray on the
              optical element, some rays might be discarded due to errors
              induced by rounding up the last digit
    """

    def __init__(self, surf_cen=[0, 0, 0], cur=1.0, n1=1.0, n2=1.5, ap=np.inf):
        if len(surf_cen) != 3:
            raise Exception("surf_cen requires three elements in the list")

        self._surf_cen = np.array(surf_cen, dtype=float)
        self._cur = cur
        self._n1 = n1
        self._n2 = n2
        self._ap = ap + 1e-10

        # set aperture radius equal to sphere radius if aperture is larger
        if self._cur != 0:  # for spherical surfaces only
            if self._ap > (1/abs(self._cur)):
                self._ap = 1/abs(self._cur) + 1e-10

        # sphere centre for a spherical surface
        self._sph_cen = None  # value will be given later

    def intercept(self, ray):
        """Calculate the intercept coordinate of the incident ray on the
           refracting surface

        'None' will be returned for any unwanted solutions or conditions

        Arg:
            ray -- object containing the positions and directions of the ray

        Main parameters:
            p, k -- the current coordinate for ray position and direction
                    respectively
            sph_cen -- coordinate of the sphere centre for a spherical surface
                       in np array; its x- and y-axis coordinates are the same
                       as those of surf_cen; for a plano surface, sph_cen will
                       not be defined
            R -- radius of the spherical surface; R is undefined for a plano
                 surface
            r -- vector pointing from sph_cen to p
            L1, L2 -- length between p and the two intercepts on a spherical
                      surface
            q1, q2 -- coordinates of the two intercepts on a spherical surface;
                      q1 is closer to p, and q2 is further away from p
            L -- length between p and the intercept on a plano surface
            q -- coordinate of the intercept on a plano surface

        Return:
            The list object of the intercept coordinate q1, q2, or q depending
            on the nature of the refracting surface
        """

        if isinstance(ray, ra.Ray) is False:
            raise Exception('class Ray is not an instance of the input ray')

        p = ray.p()
        k = ray.k()

        # case 1: spherical surface
        if self._cur != 0:
            R = 1/self._cur

            # check whether the ray travels towards the surface
            compare_val = self._surf_cen[2] - p[2]
            sign = compare_val / k[2]
            if sign > 0:  # ray travels towards the surface

                # when p is on the right of the xy-plane of the surface centre
                if p[2] > self._surf_cen[2]:
                    sph_cen = np.array([self._surf_cen[0], self._surf_cen[1],
                                       self._surf_cen[2] - R])
                    self._sph_cen = sph_cen

                # when p is on the left of the xy-plane of the surface centre
                elif p[2] < self._surf_cen[2]:
                    sph_cen = np.array([self._surf_cen[0], self._surf_cen[1],
                                       self._surf_cen[2] + R])
                    self._sph_cen = sph_cen

                # when p is on the xy-plane of the surface centre
                else:
                    return None

            else:
                return None

            r = p - sph_cen

            # to calculate the value under the sqrt
            r_dot_r = np.dot(r, r)
            r_dot_k = np.dot(r, k)
            under_sqrt = r_dot_k**2 - (r_dot_r - R**2)

            # L1 & L2 can only be calculated when the value under the sqrt >= 0
            if under_sqrt >= 0:
                L1 = -r_dot_k - np.sqrt(under_sqrt)
                L2 = -r_dot_k + np.sqrt(under_sqrt)
                q1 = p + k*L1
                q2 = p + k*L2

                # select the intercept depending on the suface curvature type
                if R > 0:  # for convex surface, return q1
                    # L1 must be positive since it represents length
                    if L1 >= 0:
                        # q1 must be within the aperture
                        if ((q1[0] - self._surf_cen[0])**2
                           + (q1[1] - self._surf_cen[1])**2) <= self._ap**2:
                            return q1
                        else:
                            return None
                    else:
                        return None

                else:  # for concave surface, return q2
                    # L2 must be positive since it represents length
                    if L2 >= 0:
                        # q2 must be within the aperture
                        if ((q2[0] - self._surf_cen[0])**2
                           + (q2[1] - self._surf_cen[1])**2) <= self._ap**2:
                            return q2
                        else:
                            return None
                    else:
                        return None

            else:
                return None

        # case 2: plano surface
        else:

            # ignore cases when k has z-coordinate = 0
            if k[2] == 0:
                return None
            else:
                L = (self._surf_cen[2] - p[2])/Normalise(k)[2]

            # L must be positive since it represents length
            if L < 0:
                return None
            else:
                q = p + Normalise(k)*L

                # q must be within the aperture
                if ((q[0] - self._surf_cen[0])**2
                   + (q[1] - self._surf_cen[1])**2) <= self._ap**2:
                    return q
                else:
                    return None

    def propagate_ray(self, ray):
        """Refract the incident ray at the surface and append the resulted ray
           position and direction to the ray object

        Arg:
            ray -- object containing the positions and directions of the ray

        Return:
            The intercept coordinate and the normalised vector of the refracted
            ray, which are appended to the input object, ray
        """

        # obatain the intercept coordinate on the surface
        intercept = self.intercept(ray)
        if intercept is None:
            return None
        else:
            # calculate the surface normal vector
            if self._cur != 0:  # for spherical surface
                surf_nor = intercept - self._sph_cen
            else:  # for plano surface
                surf_nor = np.array([0, 0, -1])

            # obtain refracted ray direction from Snell's law
            k2 = Snell(ray.k(), surf_nor, self._n1, self._n2)

            # if total internal reflection happens
            if k2 is None:
                return None
            else:
                return ray.append(intercept, k2)

    def plot2D_surf(self, bundle_list):
        """Obtain the coordinate of the lens position on the plane of y = 0

        A ray bundle with radius equals the aperture radies is generated, then
        the intercept coordinates of the rays in the ray bundle with the
        refracting surface are calculated. These intercept coordinates are
        regarded as the coordinate of the lens position.

        A ray bundle with radius slightly less than the aperture radius is
        also generated to smoothen the curve representing the spherical
        refracting surface.

        Arg:
            bundle_list -- a list of ray bundles that will be propagated
                           through element_list, which has the instance of the
                           RayBundle class

        Return:
            The x, y, z coordinates of the refracting surface
        """

        ap = self._ap - 1e-10

        # z direction of the bundle generated depends on that of the first ray
        # bundle in bundle_list
        bund_z_direc = bundle_list[0].k_init()[2]
        if bund_z_direc == 0:
            raise Exception('bund_z_direc has to be either a positive or a'
                            + 'negative value')
        elif bund_z_direc > 0:
            k_init = [0, 0, 1]
        else:
            k_init = [0, 0, -1]

        if self._cur != 0:  # for spherical surfaces
            # inital ray bundle position
            p_init = [self._surf_cen[0], self._surf_cen[1],
                      (self._surf_cen[2] - k_init[2] / abs(self._cur))]

            bundle = ra.RayBundle(radius=ap, n=24, p_init=p_init,
                                  k_init=k_init)
            ray_list1 = bundle.generate_ray_bundle()

            # add more rays around the edge of the surface
            edge = ap * 0.99
            bundle_edge = ra.RayBundle(radius=edge, n=2, p_init=p_init,
                                       k_init=k_init)
            ray_list2 = bundle_edge.generate_ray_bundle()

            # combine two ray bundles
            ray_list1 = np.array(ray_list1)
            ray_list2 = np.array(ray_list2)
            ray_list = np.hstack((ray_list1, ray_list2))

        else:
            bundle = ra.RayBundle(radius=ap, n=36,
                                  p_init=[0, 0, self._surf_cen[2] - 2],
                                  k_init=k_init)
            ray_list = bundle.generate_ray_bundle()

        int_list = []
        for ray in ray_list:
            intercept = self.intercept(ray)
            if intercept is not None:
                if abs(intercept[1] - self._surf_cen[1]) < 1e-10:
                    int_list.append(intercept)
                else:
                    pass
            else:
                pass

        # sort the list with repect to the magnitude of the x coordinates
        int_list = sorted(int_list, key=lambda x: x[0])

        x, y, z = np.array(int_list).T
        return x, y, z

    def surface_centre(self):
        """Return the position coordinate of the surface centre"""
        return self._surf_cen

    def aperture(self):
        """Return the aperture radius of the surface"""
        return self._ap

    def set_ap(self, ap_new):
        """Change the value of the aperture raidus of the surface"""
        self._ap = ap_new


class OutputPlane(SphericalRefraction):
    """Produce an output plane where the rays terminate

    It is simulated as a plano refracting surface with infinte aperture
    radius, but the result of the refraction will be neglected

    Main parameters:
        z_output -- the z axis of the output plane
        lens_list -- list of objects representing the lens, which all have
                     SphericalRefraction as their instances; the lens are
                     input to check the existance of intercepts on the lens,
                     and only the rays that can intercept with the lens will
                     be propagated to the output plane
    """

    def __init__(self, z_output=0, lens_list=[]):
        SphericalRefraction.__init__(self, [0, 0, z_output], 0, 1, 1, np.inf)

        if isinstance(lens_list, list) is False:
            raise TypeError('lens_list has to be a list, e.g. [lens] or '
                            + '[lens1, lens2]')

        self._lens_list = lens_list

    def propagate_ray(self, ray):
        """Calculate the intercept of the ray with the output plane; the ray
           direction will not be changed

        Arg:
            ray -- object containing the positions and directions of the ray

        Return:
            The intercept coordinate and the incident ray direction appended to
            the ray object
        """

        # obtain the initial ray position and direction from the input ray
        ray_p0 = ray.vertices()[0]
        ray_k0 = ray.directions()[0]

        # testing if the intercept forms with the lens in lens_list
        test_ray = ra.Ray(ray_p0, ray_k0)
        for lens in self._lens_list:
            if lens.propagate_ray(test_ray) is None:
                return None
            else:
                pass

        # obatain the intercept coordinate on the output plane
        intercept = self.intercept(ray)
        if intercept is None:
            return None
        else:
            return ray.append(intercept, ray.k())
