"""
Plot ray trajectories in 3D and 2D, and plot incident and output spot diagrams
in 2D

Calculate the RMS spot radius on the output plane and the diffraction scale of
a lens with given focal length and ray wavelength

-----------------------------------------
--- Specially for a plano-convex lens ---

Plot the RMS spot radius (a measure of spherical aberration) against ray bundle
radius for both orientations
-----------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

import ray as ra
import optical_element as ele


class PropagateNPlot:
    """Propagate the input ray bundle list and plot the ray positions into 3D
       or 2D diagrams

    Arg:
        bundle_list -- a list of ray bundles, which has the instance of the
                       RayBundle class
        element_list -- a list of objects representing the optical elements;
                        each object has SphericalRefraction() as its instance
    """

    def __init__(self, bundle_list=[], element_list=[]):

        if isinstance(bundle_list, list) is False:
            raise TypeError('bundle_list has to be a list of ray bundle '
                            + 'objects, e.g. [bundle] or [bundle1, bundle2]')

        if isinstance(element_list, list) is False:
            raise TypeError('element_list has to be a list of optical element '
                            + 'objects, e.g. [element] or [element1, element2]'
                            )

        self._bundle_list = bundle_list
        self._element_list = element_list

        # number of rays on the output plane
        self._n_output = None  # value will be asigned later

    def propagate_bundle(self):
        """Propagate the input ray bundles in the bundle_list successively
           through the list of optical elements"""

        propagate_all_bundles = []
        for bundle in self._bundle_list:
            # propagate one bundle in the bundle list
            prop_one_bund = bundle.propagate_via_elements(self._element_list)

            # append propagated bundle to re-form the bundle list
            propagate_all_bundles.append(prop_one_bund)

        return propagate_all_bundles

    def RMS(self):
        """Calculate the Root-Mean-Square (RMS) spot radius for the ray
           positions of the ray bundle on the the output plane

        This RMS value is a measure of spread of the ray positions according to
        the z-axis. If the incident ray bundle is not parallel to the z-axis,
        this RMS value would physically make no sense.

        Return:
            The RMS value for the rays positions on the output plane
        """

        # obtain the ray positions for each propagated ray bundle
        bundle_list = self.propagate_bundle()

        # extract the x and y coordinates on the output plane for calculation
        RMS_list = []
        i = 0  # to slice self._bundle_list
        j = 0  # count for the number of rays on the output plane
        for bundle in bundle_list:  # extract one ray bundle from bundle_list
            xy_squared = []

            # incident rays must be parallel to z-axis for getting RMS
            if (self._bundle_list[i].k_init()[0] == 0
               and self._bundle_list[i].k_init()[1] == 0):

                for ray in bundle:  # extract one ray from the ray bundle
                    x, y, z = np.array(ray.vertices()).T

                    # avoid using the rays outside the aperture radius
                    # those rays have less coordinate values than others
                    if len(x) <= len(self._element_list):
                        pass
                    else:
                        xy_squared.append(x[-1]**2 + y[-1]**2)
                        j += 1

                RMS = np.sqrt(np.mean(xy_squared))
                RMS_list.append(RMS)
            else:
                RMS_list.append(None)
            i += 1

        self._n_output = j

        return RMS_list

    def diffraction_scale(self, focal_length, lamb=588):
        """Calculate the diffraction scale of a known wavelength of light
           through a lens with known focal length

        Args:
            focal_length -- focal length of the lens
            lamb -- wavelength in nm of the ray

        Return:
            The list of diffraction scale values related to each bundle in the
            bundle list
        """

        diff_sca_list = []
        for bundle in self._bundle_list:
            D = 2 * bundle.radius()
            diff_sca = (lamb*1e-6) * focal_length / D
            diff_sca_list.append(diff_sca)

        return diff_sca_list

    def plot3D(self, xlim=[], ylim=[], zlim=[], title=''):
        """Plot the ray propagation diagram in 3D

        Args:
            xlim, ylim, zlim -- the range of the x-axis, y-axis, and z-axis
            title -- plot title

        Return:
            The labeled 3D plot of the ray propagation diagram
        """

        # obtain the ray positions for each propagated ray bundle
        bundle_list = self.propagate_bundle()

        # plot format parameters
        plt.rcParams.update(plt.rcParamsDefault)
        params = {
            'figure.figsize': [11, 11],
            'figure.dpi': 300,
            'axes.titlesize': 30,
            'axes.labelsize': 25,
            'axes.labelpad': 35,
            'axes.xmargin': 0.1,
            'xtick.labelsize': 20,
            'xtick.major.pad': 15,
            'legend.fontsize': 20,
            }
        plt.rcParams.update(params)

        ax1 = plt.axes(projection='3d')

        # extract the ray positions for plotting
        i = 0  # for changing the plot color when necessary
        for bundle in bundle_list:  # extract one ray bundle from bundle_list
            for ray in bundle:  # extract one ray from the ray bundle
                x, y, z = np.array(ray.vertices()).T

                # determine ray colours depending on the number of ray bundles
                if len(bundle_list) == 1:  # if only one ray bundle
                    # plot each ray in different colour
                    ax1.plot(z, y, x, lw=3)

                else:
                    # keep rays of one ray bundle in one colour
                    ax1.plot(z, y, x, lw=3, color='C'+str(i)+'')
            i += 1

        ax1.set_xlabel("z (mm)")
        ax1.set_ylabel("y (mm)")
        ax1.set_zlabel("x (mm)")

        if len(xlim) != 0:
            ax1.set_xlim(xlim[0], xlim[1])
            ax1.set_ylim(ylim[0], ylim[1])
            ax1.set_zlim(zlim[0], zlim[1])

        if len(title) != 0:
            ax1.set_title(title)
        else:
            ax1.set_title("Ray propagation in 3D")

        plt.grid()

    def plot2D_rcParams(self):
        """Define the plot formatting parameters for the 2D plots"""

        # reset thhe formatting parameters to default values
        plt.rcParams.update(plt.rcParamsDefault)

        # parameter list for the plot format
        params = {
            'figure.figsize': [8, 8],
            'axes.labelsize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'axes.titlesize': 25,
            'figure.dpi': 300,
            }

        return plt.rcParams.update(params)

    def plot2D_lens(self):
        """Plot the lens on the plane of y = 0"""

        # obtain the aperture radius of the lens, excluding output plane
        ap_list = []
        for element in self._element_list:
            if isinstance(element, ele.OutputPlane) is False:
                ap = element.aperture()
                ap_list.append(ap)
            else:
                pass
        ap_min = np.min(ap_list)

        for element in self._element_list:
            if isinstance(element, ele.OutputPlane) is False:
                if element.aperture() > ap_min:
                    element.set_ap(ap_min)
                else:
                    pass

                # obtain the coordinate of the lens and plot it
                x_s, y_s, z_s = element.plot2D_surf(
                        bundle_list=self._bundle_list)
                plt.plot(z_s, x_s, '-', lw=3, color='grey')

            else:
                pass

    def plot2D_incident(self, xlim=[], ylim=[], title=''):
        """Plot the incident ray positions in xy-plane

        Args:
            xlim, ylim -- the range of the x-axis and the y-axis
            title -- plot title

        Return:
            The labeled 2D plot of the incident ray positions in xy-plane
        """

        # obtain the ray positions for each propagated ray bundle
        bundle_list = self.propagate_bundle()

        # update plot format
        self.plot2D_rcParams()

        # extract the ray positions for plotting
        i = 0  # for changing the plot color when necessary
        for bundle in bundle_list:  # extract one ray bundle from bundle_list
            for ray in bundle:  # extract one ray from the ray bundle
                x, y, z = np.array(ray.vertices()).T

                # avoid plotting the rays outside the aperture radius
                # those rays have less coordinate values than others
                if len(x) <= len(self._element_list):
                    pass
                else:
                    plt.plot(y[0], x[0], '.', ms=12, color='C'+str(i)+'')
            i += 1

        plt.xlabel("y (mm)")
        plt.ylabel("x (mm)")

        if len(xlim) != 0:
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])

        if len(title) != 0:
            plt.title(title)
        else:
            plt.title("Incident ray positions in xy-plane")

        plt.grid()

    def plot2D_output(self, xlim=[], ylim=[], title='', print_rms=True):
        """Plot the ray positions shown on the output plane

        The output plane is perpendicular to the z-axis, i.e. it is one of the
        xy-plane

        Args:
            xlim, ylim -- the range of the x-axis and the y-axis
            title -- plot title
            print_rms -- boolean variable that controls whether printing out
                         the RMS values or not

        Return:
            The labeled 2D plot of the ray positions on the output plane
        """

        # obtain the ray positions for each propagated ray bundle
        bundle_list = self.propagate_bundle()

        # update plot format
        self.plot2D_rcParams()

        # extract the ray positions for plotting
        i = 0  # for changing the plot color when necessary
        for bundle in bundle_list:  # extract one ray bundle from bundle_list
            for ray in bundle:  # extract one ray from the ray bundle
                x, y, z = np.array(ray.vertices()).T

                # avoid plotting the rays outside the aperture radius
                # those rays have less coordinate values than others
                if len(x) <= len(self._element_list):
                    pass
                else:
                    plt.plot(y[-1], x[-1], '.', ms=12, color='C'+str(i)+'')

            if print_rms is True:
                if self.RMS()[i] is None:
                    print('\nRMS cannot be calculated for bundle '+str(i+1)+' '
                          + 'due to non-parallel incidnet rays along z axis')
                else:
                    print('\nRMS spot radius for ray bundle No. '+str(i+1)+' '
                          + 'is %s mm' % (self.RMS()[i]))

            i += 1

        plt.xlabel("y (mm)")
        plt.ylabel("x (mm)")

        if len(xlim) != 0:
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])

        if len(title) != 0:
            plt.title(title)
        else:
            plt.title("Ray positions on output plane")

        plt.grid()

    def plot2D_xz_all_rays(self, xlim=[], ylim=[], title=''):
        """Plot the diagram for the propagation of all the rays as viewed in
           the xz-plane

        Args:
            xlim, ylim -- the range of the x-axis and the y-axis
            title -- plot title

        Return:
            The labeled 2D plot for the propagation of all the rays as viewed
            in the xz-plane
        """

        # obtain the ray positions for each propagated ray bundle
        bundle_list = self.propagate_bundle()

        # update plot format
        self.plot2D_rcParams()

        # extract the ray positions for plotting
        i = 0  # for changing the plot color when necessary
        for bundle in bundle_list:  # extract one ray bundle from bundle_list
            for ray in bundle:  # extract one ray from the ray bundle
                x, y, z = np.array(ray.vertices()).T

                # avoid plotting the rays outside the aperture radius
                # those rays have less coordinate values than others
                if len(x) <= len(self._element_list):
                    pass
                else:
                    plt.plot(z, x, '-', color='C'+str(i)+'')
            i += 1

        # plot the lens
        self.plot2D_lens()

        plt.xlabel("z (mm)")
        plt.ylabel("x (mm)")

        if len(xlim) != 0:
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])

        if len(title) != 0:
            plt.title(title)
        else:
            plt.title("All ray positions viewed in xz-plane")

        plt.grid()

    def plot2D_xz_ray_cen(self, xlim=[], ylim=[], title=''):
        """Plot the ray propagation diagram in xz-plane along the diameter of
           the ray bundle

        Args:
            xlim, ylim -- the range of the x-axis and the y-axis
            title -- plot title

        Return:
            The labeled 2D plot of the ray propagation diagram in xz-plane
            along the diameter of the ray bundle
        """

        # obtain the ray positions for each propagated ray bundle
        bundle_list = self.propagate_bundle()

        # update plot format
        self.plot2D_rcParams()

        # extract the y coordinate of the 1st optical element's surface centre
        y_ray_cen = self._element_list[0].surface_centre()[1]

        # extract the ray positions for plotting
        i = 0  # for changing the plot color when necessary
        for bundle in bundle_list:  # extract one ray bundle from bundle_list
            for ray in bundle:  # extract one ray from the ray bundle
                x, y, z = np.array(ray.vertices()).T

                # avoid plotting the rays outside the aperture radius
                # those rays have less coordinate values than others
                if len(x) <= len(self._element_list):
                    pass
                else:
                    # only plot the rays on the plane of the ray bundle centre
                    if abs(y[0] - y_ray_cen) <= 1e-10:
                        plt.plot(z, x, '-', color='C'+str(i)+'')
                    else:
                        pass
            i += 1

        # plot the lens
        self.plot2D_lens()

        plt.xlabel("z (mm)")
        plt.ylabel("x (mm)")

        if len(xlim) != 0:
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])

        if len(title) != 0:
            plt.title(title)
        else:
            plt.title("Rays along ray bundle diameter in xz-plane")

        plt.grid()

    def n_output(self):
        """Return the number of rays on the output plane"""
        return self._n_output


class PlanoConvex(PropagateNPlot):
    """Produce plots related to the plano-convex lens only

    Args:
        rad_max -- maximum radius of the testing ray bundles generated
        n -- number of testing ray bundles generated to form the bundle list
        c_p -- list in the form of [convex, plano] to represent the convex-
               plano lens
        p_c -- list in the form of [plano, convex] to represent the plano-
               convex lens
    """

    def __init__(self, rad_max=10, n=10, c_p=[], p_c=[]):

        PropagateNPlot.__init__(self)

        self._rad_max = rad_max
        self._n = n
        self._c_p = c_p
        self._p_c = p_c

    def plano_convex_orient(self, title='', print_foc=True):
        """Plot the RMS spot radius against beam radius for both orientations
           of a plano-convex lens

        Return:
            The plot of RMS spot radius against beam radius with convex-plano
            and plano-convex lens plotted as seperate curves
        """

        # generate a list of ray bundles
        radius_list = []
        bundle_list = []
        for i in range(self._n):
            rad = (i+1) * self._rad_max/self._n
            bundle = ra.RayBundle(radius=rad, n=6, p_init=[0, 0, -2],
                                  k_init=[0, 0, 1])
            radius_list.append(rad)
            bundle_list.append(bundle)
        self._bundle_list = bundle_list

        # obtain the RMS value list for convex-plano orientation
        z_focus_cp = ele.paraxial_focus(bundle_list, self._c_p,
                                        print_foc)  # get focal length
        output_cp = ele.OutputPlane(z_focus_cp,
                                    self._c_p)  # output at focal point
        self._element_list = [self._c_p[0], self._c_p[1],
                              output_cp]  # update element_list
        RMS_list_cp = self.RMS()  # calculate RMS

        # obtain the RMS value list for convex-plano orientation
        z_focus_pc = ele.paraxial_focus(bundle_list, self._p_c, print_foc)
        output_pc = ele.OutputPlane(z_focus_pc, self._p_c)
        self._element_list = [self._p_c[0], self._p_c[1], output_pc]
        RMS_list_pc = self.RMS()

        # plot the graph
        self.plot2D_rcParams()

        plt.plot(radius_list, RMS_list_cp, '-', label='convex-plano')
        plt.plot(radius_list, RMS_list_pc, '-', label='plano-convex')

        plt.xlabel('Beam radius (mm)')
        plt.ylabel('RMS spot radius (mm)')

        if len(title) != 0:
            plt.title(title)
        else:
            plt.title('Performance of plano-convex lens\nin both orientations')

        plt.legend()
        plt.grid()
