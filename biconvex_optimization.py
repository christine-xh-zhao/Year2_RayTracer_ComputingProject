"""
Optimise the biconvex lens for its surface curvatures and/or the distance
between its two surfaces

Plot the aberration scale against the ray bundle radius for both the
diffraction scale and the RMS spot radius of a given lens
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

import ray as ra
import optical_element as ele
import propagate_and_plot as pnp


class BiconvexOptimise(pnp.PropagateNPlot):
    """Optimise the curvature of and the distance between the two surfaces of a
       biconves lens

    Args:
        ray_bundle -- one ray bundle that used to optimise the lens; it is in
                      the form of list, or [bundle], for the consistancy with
                      other classes involved ray bundles
        focal_len -- focal length of the lens
        surf_cen -- surface centre coordinate of the surface that rays incident
                    onto the lens
        trial_cur -- trial value for the curvatures of the two spherical
                     surfaces of the lens; it has to be a list of two elements
                     that represent the curvature of the surface which rays
                     incident and leave the lens respectively
        n1 -- the refractive index of the medium outside the lens
        n2 -- the refractive index of the material of the lens
    """

    def __init__(self, ray_bundle, focal_len=10, surf_cen=[0, 0, 0], n1=1,
                 n2=1.5168):

        # for consistency with the methods in all the other classes
        if isinstance(ray_bundle, list) is False:
            raise TypeError('ray_bundle has to be a list of only one object of'
                            + 'the ray bundle, e.g. [bundle]')

        if len(ray_bundle) != 1:
            raise Exception('ray_bundle must contain only one object of the '
                            + 'ray bundle, e.g. [bundle]')

        pnp.PropagateNPlot.__init__(self, ray_bundle, [])

        self._focal_len = focal_len
        self._surf_cen = surf_cen
        self._n1 = n1
        self._n2 = n2

        # retain the radius of the ray bundle that used for optimization
        self._radius_init = ray_bundle[0].radius()

    def biconvex(self, trial_cur, trial_dis):
        """Define the two surfaces of the biconvex lens and the output plane
           at the given image distance. Then propagate the input ray bundle
           through them

        Args:
            trial_cur -- trail curvature values for optimization
            trial_dis -- trial distance values for optimization

        Return:
            The RMS spot radius value at the given image distance
        """

        # obtain the normalised z direction of the ray bundle, i.e. +1 or -1
        bund_z_norm = (self._bundle_list[0].k_init()[2]
                       / abs(self._bundle_list[0].k_init()[2]))

        # create two spherical surfaces
        surf1 = ele.SphericalRefraction(surf_cen=self._surf_cen,
                                        cur=trial_cur[0], n1=self._n1,
                                        n2=self._n2, ap=np.inf)
        surf2 = ele.SphericalRefraction(surf_cen=[self._surf_cen[0],
                                                  self._surf_cen[1],
                                                  self._surf_cen[2]
                                                  + trial_dis * bund_z_norm],
                                        cur=trial_cur[1], n1=self._n2,
                                        n2=self._n1, ap=np.inf)

        # let the aperture radii of the two surfaces to be the same
        if surf1.aperture() >= surf2.aperture():
            surf1.set_ap(surf2.aperture())
        else:
            surf2.set_ap(surf1.aperture())

        # create the output plane, its position depends on ray bundle direction
        output = ele.OutputPlane(z_output=(bund_z_norm * self._focal_len +
                                           self._surf_cen[2]),
                                 lens_list=[surf1, surf2])

        self._element_list = [surf1, surf2, output]

        # since there is only one value in the list of self.RMS()
        return self.RMS()[0]

    def define_variable(self, func, var_index=0, value=[]):
        """Define one vairable in a multi-variable function as the only
           vairable in the function

        Args:
            func -- function with multi-variables
            var_index -- variable index in the input args of func
            value -- values substitute into each of the variables of func

        Return:
            The magnitude of the partial derivative of the selected variable
            which the values were substituted into the derivative

        Reference:
            https://stackoverflow.com/questions/20708038/scipy-misc-derivative-
            for-multiple-argument-function
        """

        args = value[:]

        def wraps(x):
            args[var_index] = x
            return func(*args)

        return wraps

    def optimise_cur(self, trial_cur=[0.1, -0.01], trial_dis=5,
                     print_res=True, plot_rays=True):
        """Optimise the curvatures of the two spherical surfaces of the
           biconvex lens so that the RMS spot radius at the given image distace
           is minimised

        Plot of ray trajectories for the optimied lens would be saved to
        Plots_from_tasks folder

        Arg:
            trial_cur -- the trial values for the curvatures of the ray-
                         incident surface and the ray-leaving surface
                         respectively
            trial_dis -- the trial values for the distance between the ray-
                         incident surface and the ray-leaving surface
                         respectively
            print_res -- boolean variable that controls whether printing out
                         the optimization result or not
            plot_rays -- boolean variable that controls whether plotting the
                         ray trajectories of the optimised lens or not

        Return:
            The curvature values of the ray-incident surface and the ray-
            leaving surface as list
        """

        if isinstance(trial_cur, list) is False:
            raise TypeError('trial_cur must be a list')

        if len(trial_cur) != 2:
            raise Exception("trial_cur requires two elements in the list")

        # make the trial_cur in biconvex function the only variable
        bicon_cur = self.define_variable(self.biconvex, 0, [trial_cur,
                                                            trial_dis])

        # optimise the curvature values
        cur_list = spo.fmin(bicon_cur, trial_cur)

        # only retain the optimization result when all rays in the bundle
        # propagate to the output plane
        number_input = self._bundle_list[0].n_tot()  # number of rays generated
        number_ouput = self.n_output()  # number of rays on output plane
        if number_ouput < number_input:
            print('\nNot all rays in the bundle propagate to the output plane',
                  '\n')
            return None
        else:
            # caculate the RMS value
            RMS = self.biconvex(cur_list, trial_dis)
            if print_res is True:
                print('-'*20, '\nWhen the biconvex lens has focal length of %s'
                      + ' mm,'
                      % (self._focal_len)
                      + '\nit has the least aberration when\n'
                      + '\n- radius of the ray bundle for optimization = %s mm'
                      % (self._radius_init)
                      + '\n- distance between the two surfaces = %s mm'
                      % (trial_dis)
                      + '\n- first surface curvature = %s mm-1 '
                      % (cur_list[0])
                      + '\n  second surface curvature = %s mm-1'
                      % (cur_list[1])
                      + '\n- RMS spot radius = %s mm'
                      % (RMS))

            # plot ray trajectories along ray bundle diameter
            if plot_rays is True:

                # round focal length value for printting
                foc_len = np.around(self._focal_len, 3)

                # obtain xlim for plotting
                p_z0 = self._bundle_list[0].p_init()[2]
                foc_int = int(self._surf_cen[2] + foc_len)

                self.plot2D_xz_ray_cen(
                    xlim=[p_z0-5, foc_int+5], ylim=[-65, 65],
                    title='Rays along ray bundle diameter for\nthe optimised '
                          + f'biconvex lens with f={foc_len}mm')
                plt.savefig('Plots_from_tasks/Lens Opt. - trajectories.png')
                plt.show()

            return cur_list

    def optimise_cur_n_dis(self, trial_cur=[0.1, -0.01], trial_dis_max=5,
                           n=10, print_each=True):
        """Optimise the curvatures at a given distance between the two
           spherical surfaces of the biconvex lens, and loop for a range of
           distances to find the distance with the least RMS spot radius

        Optimised values of the curvatures and the distance are obtained when
        the RMS spot radius of the lens is minimised

        Plot of ray trajectories only for the lens with the least aberration
        would be saved to Plots_from_tests folder

        Arg:
            trial_cur -- the trial values for the curvatures of the ray-
                         incident surface and the ray-leaving surface
                         respectively
            trial_dis_max -- the max trial values for the distance between the
                             ray-incident surface and the ray-leaving surface
                             respectively
            n -- number of trial_dis for testing
            print_each -- boolean variable that controls whether printing out
                          the optimization result for each trial of
                          optimization or not

        Return:
            The list of distance values and the related list of RMS values
        """

        # a list of trial_dis is generated with min = 1 mm
        trial_dis_list = np.linspace(1, trial_dis_max, n)

        # optimise the lens curvatures for each distance values in the list
        RMS_list = []
        cur_list = []
        i = 0
        while i < len(trial_dis_list):
            # optmise the lens curvatures at given distance between surfaces
            cur = self.optimise_cur(trial_cur, trial_dis_list[i],
                                    print_res=print_each, plot_rays=False)

            # only retain the related values when cur is not None
            if cur is None:
                # delete the distance value from trial_dis_list
                trial_dis_list = np.delete(trial_dis_list, i)
                pass
            else:
                cur_list.append(cur)
                RMS_list.append(self.biconvex(cur, trial_dis_list[i]))
                i += 1

        # select the min RMS value and its related dis value from related list
        min_index = np.where(RMS_list == np.min(RMS_list))
        min_index = min_index[0][0]

        print('-'*20, '\nWhen the biconvex lens has focal length of %s mm'
              % (self._focal_len)
              + '\nit has the least aberration when\n'
              + '\n- the distance between the two surfaces = %s mm'
              % (trial_dis_list[min_index])
              + '\n- first surface curvature = %s mm-1 '
              % (cur_list[min_index][0])
              + '\n  second surface curvature = %s mm-1'
              % (cur_list[min_index][1])
              + '\n- RMS spot radius = %s mm\n'
              % (RMS_list[min_index]))

        # plot graph only for the lens with the least aberration
        self.biconvex(cur_list[min_index], trial_dis_list[min_index])
        self.plot2D_xz_ray_cen(
            xlim=[-20, 105], ylim=[-25, 25],
            title='Rays along ray bundle diameter for\nthe optimised biconvex '
                  + 'lens with f='+str(self._focal_len)+'mm')
        plt.savefig('Plots_from_tests/'
                    + 'Rays along ray bundle diameter for the optimised '
                    + 'biconvex lens with f='+str(self._focal_len)+'mm')
        plt.show()

        return cur_list, trial_dis_list, RMS_list, min_index

    def diff_vs_rms(self, op_result, rad_max=10, n=10, lamb=588, p_z0=-2,
                    out_single=False, out_all=True, test=False):
        """Plot the aberration scale values against beam radius for both the
           RMS spot radius and the diffraction scale together in one plot

        Arg:
            op_result -- the returned result from optimise_cur_n_dis function;
                         one can input the relevant curvature, distance, RMS,
                         and index of the minimum the last two to plot the
                         graph
            rad_max -- maximum radius of the testing ray bundles generated
            n -- number of testing ray bundles generated to form the bundle
                 list
            lamb -- wavelength in nm of the input ray for diffraction scale
                    calculation
            p_z0 -- initial z-coordinate position for all the ray bundles; this
                    parameter is used to control how the plot looks like mainly
            out_single -- boolean variable that controls whether plotting the
                          ray positions of one ray bundle on the output plane
            out_all -- boolean variable that controls whether plotting the
                       trajectories of all the ray bundles and all the ray
                       positions on the output plane
            test -- boolean varaible that controls which folder to save the
                    plot

        Return:
            The plot of aberration scale values against beam radius for both
            the RMS spot radius and the diffraction scale
        """

        # obtain related values from the input op_result
        cur_list, trial_dis_list, RMS_list, min_index = op_result

        # pass the optimised curvature and distance to reform the biconvex lens
        self.biconvex(cur_list[min_index], trial_dis_list[min_index])

        # obtain the focal length
        foc_len = np.around(self._focal_len, 3)

        # generate a list of testing ray bundles
        radius_list = []
        bundle_list = []
        for i in range(n):
            rad = (i+1) * rad_max/n
            bundle = ra.RayBundle(radius=rad, n=6, p_init=[0, 0, p_z0],
                                  k_init=[0, 0, 1])
            radius_list.append(rad)
            bundle_list.append(bundle)

            # plot the ray bundle when it reaches the output plane
            if out_single is True:
                self._bundle_list = [bundle]
                self.plot2D_output(title='No. '+str(i+1)+' ray bundle on the '
                                   + 'output plane for\nthe optimised biconvex'
                                   + f' lens with f={foc_len}mm')
                plt.show()

        self._bundle_list = bundle_list

        # calculate RMS and diffraction values by propagating the ray bundles
        RMS_list = self.RMS()
        diff_sca_list = self.diffraction_scale(focal_length=self._focal_len,
                                               lamb=lamb)

        # plot the graph
        self.plot2D_rcParams()

        # plot all ray bundles on output plane and along budnle diameter
        if out_all is True:
            self.plot2D_output(title=''+str(n)+' ray bundles on the output '
                               + 'plane for\nthe optimised biconvex '
                               + f'lens with f={foc_len}mm')
            plt.show()
            self.plot2D_xz_ray_cen(
                xlim=[p_z0-5, 105], ylim=[-25, 25],
                title=''+str(n)+' ray bundles along bundle diameter for\nthe '
                      + f'optimised biconvex lens with f={foc_len}mm')
            plt.show()

        plt.plot(radius_list, RMS_list, '-', label='RMS spot radius')
        plt.plot(radius_list, diff_sca_list, '-', label='Diffraction scale')

        plt.xlabel('Beam radius (mm)')
        plt.ylabel('Aberration scale (mm)')
        plt.title(f'Biconvex lens with f={foc_len}mm\noptimised with beam '
                  + f'radius of {self._radius_init} mm')

        plt.legend()
        plt.grid()

        if test is False:
            plt.savefig('Plots_from_tasks/Lens Opt. - comparison.png')
        else:
            plt.savefig('Plots_from_tests/Biconvex'
                        + f' lens with f={foc_len}mm optimised with beam '
                        + f'radius of {self._radius_init} mm.png')
        plt.show()

    def dis_vs_rms(self, op_result):
        """Plot the RMS spot radius against the trial distance values and the
           minimum RMS value as a horizontal reference line

           This plot graphically shows the optimization result for a set of
           distance values between the two surfaces of a biconvex lens. Lower
           the RMS value for a given distance means the better optimization got
           for the surface curvature values.

           This plot is only used as a check for the optimization result

        Arg:
            op_result -- the returned result from optimise_cur_n_dis function;
                         one can input the relevant curvature, distance, RMS,
                         and index of the minimum the last two to plot the
                         graph

        Return:
            The plot of RMS against trial distance with min values noted on the
            plot
        """

        # obtain related values from the input op_result
        cur_list, trial_dis_list, RMS_list, min_index = op_result

        plt.plot(trial_dis_list, RMS_list, '.', ls='-', color='green',
                 label='After optimization')

        rms_min = RMS_list[min_index]
        dis_min = np.around(trial_dis_list[min_index], 3)

        plt.plot(trial_dis_list, np.zeros(len(trial_dis_list)) + rms_min,
                 ls='-', label=(f'Min = {rms_min:8.3e}\nwhen dis = {dis_min}'))

        rms_avg = np.mean(RMS_list)
        rms_avg_err = np.std(RMS_list, ddof=1)

        plt.plot(trial_dis_list, np.zeros(len(trial_dis_list)) + rms_avg,
                 ls='-', label=(f'Avg = {rms_avg:8.3e}\n'
                                + f'       \u00b1 {rms_avg_err:8.3e}'))

        plt.xlabel('Distance between two surfaces (mm)')
        plt.ylabel('RMS spot radius (mm)')
        plt.title('Optimised biconvex lens with f='+str(self._focal_len)+'mm')

        plt.legend(fontsize=19)
        plt.grid()

        plt.savefig('Plots_from_tests/Optimised biconvex lens with'
                    + ' f='+str(self._focal_len)+'mm')
        plt.show()
