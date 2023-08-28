"""
Test file for all the modules of the ray-tracer

The running time of this file is around 45s

The module of unittest is used for all the testings

----------------------------------------------------------------
--- For the long RMS printouts in TestBiconvexOptimise class ---

If one does not want to print the RMS for each bundle, use out_all=False when
calling the diff_vs_rms method

Note that the plots for both output ray positions and ray trajectories will not
be produced if using out_all=False
----------------------------------------------------------------

-------------------------------------------------------------------------------
--- Special note for optimise_cur_n_dis method in bi.BiconvexOptimise class ---

The optimise_cur_n_dis method is tested in this file, however it is not used
for producing plots for assessment

The reason for that is I found out optimising both curvatures of the surfaces
and the distaces between the surfaces is not necessary, since the variations in
the RMS values for each optimization were small, and the trigger of these
variations would mainly from scipy.optimize.fmin rather than the lens
performance
-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import unittest

import ray as ra
import optical_element as ele
import propagate_and_plot as pnp
import biconvex_optimization as bi


class TestRay(unittest.TestCase):
    """Test the methods in the ra.Ray class """

    def test_input_length(self):
        """Check if exception is raised for worng length of the input list"""

        with self.assertRaises(Exception):
            ra.Ray(p=[0, 0])

        with self.assertRaises(Exception):
            ra.Ray(k=[0, 0, 0, 0])

    def test_append_N_return(self):
        """Check (1) if new p can be appended to the ray object, (2) if all
           the p values can be returned correctly, and (3) the last p value is
           returned correctly"""

        # from the function in the module
        ray = ra.Ray(p=[1, 1, 1])
        ray.append(p=[2, 2, 2], k=[0, 0, 0])
        ray_all = ray.vertices()
        ray_last = ray.p()

        self.assertTrue((ray_all == [[1, 1, 1], [2, 2, 2]]).all)
        self.assertTrue((ray_last == [2, 2, 2]).all)


class TestRayBundle(unittest.TestCase):
    """Test the methods in the ra.RayBundle class"""

    def test_generate_ray_bundle(self):
        """Check the ray bundle with correct coordinate is generated"""

        # from the function in the module
        init = ra.RayBundle(radius=1, n=3, p_init=[0, 0, -5], k_init=[0, 0, 1])
        bundle = init.generate_ray_bundle()
        ray_last = bundle[-1].p()

        self.assertTrue((ray_last == [1, 0, -5]).all)

    def test_input_list(self):
        """Check, for the propagate_via_elements() function, if exception is
           raised when the input element list is not in the form of a list"""

        with self.assertRaises(TypeError):
            ra.RayBundle().propagate_via_elements(element_list=1)


class TestEleDefs(unittest.TestCase):
    """Test the methods in the defined functions in the ele module"""

    def test_normalise(self):
        """Check the normalisation of the vectors works properly"""

        # expected
        vec = [1, 1, 2]
        exp_vec = np.array(vec) / np.sqrt(5)

        # from the function in the module
        nor_vec = ele.Normalise(vec)

        self.assertTrue((nor_vec == exp_vec).all)

    def test_snell(self):
        """Check the Snell() function for implementing Snell's law gives the
           correct result"""

        k1 = [1, 1, -1]
        surf_nor = [0, 0, 1]
        n1 = 1
        n2 = 1.5

        # expected
        theta1 = np.arctan(np.sqrt(2))
        theta2 = np.arcsin(n1 * np.sin(theta1) / n2)
        cos_theta2 = np.cos(theta2)

        # from the function in the module
        k2_hat = ele.Snell(k1, surf_nor, n1, n2)
        cos_angle2 = -np.dot(k2_hat, surf_nor)

        self.assertEqual(cos_theta2, cos_angle2)

    def test_paraxial_focus(self):
        """Check the paraxial focus is correctly obtained"""

        # input check with variables that would raise an error message
        surf = ele.SphericalRefraction()
        output = ele.OutputPlane()
        element_list = [surf, output]

        with self.assertRaises(TypeError):
            ele.paraxial_focus(bundle_list=[], element_list=1)

        with self.assertRaises(Exception):
            ele.paraxial_focus(bundle_list=[], element_list=element_list)

        # focal length check with a surface of expected focal length of 100
        print('')
        print('-'*70)
        print('Test for obtaining the correct focal length for convex surface')
        bun = ra.RayBundle(radius=1, n=2)
        elem = ele.SphericalRefraction(cur=0.03)

        foc = ele.paraxial_focus([bun], [elem])
        foc_round = np.around(foc, 1)
        foc_exp = 100.0
        self.assertEqual(foc_round, foc_exp)

        # focal length check with a surface of expected focal length of -100
        print('')
        print('-'*70)
        print('Test for obtaining the correct focal length for concave'
              + ' surface')
        bun = ra.RayBundle(radius=1, n=2, p_init=[0, 0, 2], k_init=[0, 0, -1])
        elem = ele.SphericalRefraction(cur=-0.03)

        foc = ele.paraxial_focus([bun], [elem])
        foc_round = np.around(foc, 1)
        foc_exp = -100.0
        self.assertEqual(foc_round, foc_exp)


class TestSphericalRefraction(unittest.TestCase):
    """Test the methods in the ele.SphericalRefraction class"""

    def test_intercept(self):
        """Check if None is returned for the rays that would not intercept with
           the surface, and check if the correct intercept is selected
           according to the nature of the surface"""

        # test for convex, concave, and plano surfaces
        cur_list = [1, -1, 0]

        for i in range(3):

            cur = cur_list[i]
            surf = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=cur, n1=1.0,
                                           n2=1.5, ap=1.0)

            # ray outside aperture radius
            ray1 = ra.Ray(p=[2, 0, -1], k=[0, 0, 1])
            result1 = surf.intercept(ray1)  # result for testing
            self.assertEqual(result1, None)

            # ray travelling away from surf
            ray2 = ra.Ray(p=[0, 0, 1], k=[0, 0, 1])
            result2 = surf.intercept(ray2)  # result for testing
            self.assertEqual(result2, None)

            # ray travelling along -z direction
            ray3 = ra.Ray(p=[0, 0, 1], k=[0, 0, -1])
            result3 = surf.intercept(ray3)  # result for testing
            self.assertTrue(result3 is not None)

            # correct intercept selected according to nature of the surface
            p_x = np.sqrt(2)/2  # for expected value calculation
            ray4 = ra.Ray(p=[p_x, 0, -1], k=[0, 0, 1])
            result4 = surf.intercept(ray4)  # result for testing

            # compare with expected
            if cur == 1:
                q_exp = [p_x, 0, (1-p_x)]
                self.assertTrue((result4 == q_exp).all)
            elif cur == -1:
                q_exp = [p_x, 0, (p_x-1)]
                self.assertTrue((result4 == q_exp).all)
            else:
                q_exp = [p_x, 0, 0]
                self.assertTrue((result4 == q_exp).all)

    def test_propagate_ray(self):
        """Check if the ray is propagated properly with correct position and
           and direction coordinates appended to the ray object"""

        # total internal reflection happens
        print('')
        print('-'*70)
        print('Test for propagate_ray method in ele.SphericalRefraction'
              + ' class:')
        ray1 = ra.Ray(p=[1/2, 1/2, -1])
        surf1 = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=1, n1=2, n2=1)
        result1 = surf1.propagate_ray(ray1)  # result for testing
        self.assertTrue(result1 is None)

        ray2 = ra.Ray(p=[1, 0, -1])
        surf2 = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=1, n1=1, n2=2)
        result2 = surf2.propagate_ray(ray2)  # result for testing

        # check position coordinate
        p_exp = [1., 0, 1.]
        self.assertTrue((result2.p() == p_exp).all)

        # check direction coordinate
        k_exp = ele.Snell([0, 0, 1], [1, 0, 0], 1, 2)
        self.assertTrue((result2.k() == k_exp).all)


class TestOutputPlane(unittest.TestCase):
    """Test the methods in the ele.OutputPlane class"""

    def test_inheritance(self):
        """Check of the OutputPlane class correctly inheritated the
           SphericalRefraction class"""

        out1 = ele.OutputPlane(z_output=5, lens_list=[])

        # check the surface centre posistion
        surf_exp = [0, 0, 5]
        self.assertTrue((out1.surface_centre() == surf_exp).all)

        # check the aperture radius
        ap_exp = np.inf
        self.assertEqual(out1.aperture(), ap_exp)

    def test_input_list(self):
        """Check if exception is raised when the input element list is not in
            the form of a list"""

        # check if error is raised for lens_list that is not an list
        with self.assertRaises(TypeError):
            ele.OutputPlane(z_output=5, lens_list=1)

    def test_propagate_ray(self):
        """Check if the ray is propagated correctly through the lens to the
           output plane"""

        test_lens = ele.SphericalRefraction()

        # if the ray does not intercept with the lens
        ray1 = ra.Ray(p=[2, 0, -2])
        test_lens.propagate_ray(ray1)
        result1 = ele.OutputPlane(z_output=2,
                                  lens_list=[test_lens]).propagate_ray(ray1)
        self.assertEqual(result1, None)

        # test if ray is propagated properly
        ray2 = ra.Ray(p=[1, 0, -2])
        test_lens.propagate_ray(ray2)

        k_exp = ray2.k()

        ele.OutputPlane(z_output=2.78889,
                        lens_list=[test_lens]).propagate_ray(ray2)

        self.assertTrue((ray2.k() == k_exp).all)

        p_test = ray2.p()
        p_test = [np.around(p_test[0], 4), p_test[1], p_test[2]]
        p_exp = [3., 0, 2.78889]
        self.assertEqual(p_test, p_exp)


class TestPropagateNPlot(unittest.TestCase):
    """Test the methods in the PropagateNPlot class"""

    def test_RMS(self):
        """Check the RMS spot radius calculation"""

        bun = ra.RayBundle(radius=1, n=2)
        elem = ele.OutputPlane()
        out = ele.OutputPlane(z_output=2)
        bun_prop = pnp.PropagateNPlot(bundle_list=[bun], element_list=[elem,
                                                                       out])

        # test the correct number of rays are generated in the ray bundle
        bun_prop.propagate_bundle()
        num = bun.n_tot()
        num_exp = 3
        self.assertEqual(num, num_exp)

        # test the rms calculation
        rms = bun_prop.RMS()[0]
        rms_exp = np.sqrt(2/bun.n_tot())
        self.assertEqual(rms, rms_exp)

    def test_diffraction_scale(self):
        """Check the diffraction scale calculation"""

        bun = ra.RayBundle(radius=5)

        diff = pnp.PropagateNPlot(
            bundle_list=[bun]).diffraction_scale(focal_length=100, lamb=588)[0]
        diff_exp = 0.00588
        self.assertEqual(diff, diff_exp)

    def test_plot1(self):
        """A ray bundle travels along -z direction is propagated through a
           single spherical refracting surface which is located at [3, 4, 5]
           with radius of 1 mm and focal length of 3 mm"""

        print('')
        print('-'*70)
        print('Plotting test 1')

        # define lens and ray bundles
        lens = ele.SphericalRefraction(surf_cen=[3, 4, 5], cur=1, n1=1, n2=1.5)
        r = ra.RayBundle(radius=0.8, n=4, p_init=[3, 4, 6], k_init=[0, 0, -1])
        r = [r]

        # put output plane at the focal point
        z_focus = ele.paraxial_focus(r, [lens])
        output = ele.OutputPlane(z_focus, [lens])

        # plot
        elem = [lens, output]
        pnp.PropagateNPlot(r, elem).plot3D(
            title='Plotting test 1 in 3D for all trajectories')
        plt.savefig('Plots_from_tests/'
                    + 'Plotting test 1 in 3D for all trajectories')
        plt.show()

        pnp.PropagateNPlot(r, elem).plot2D_xz_all_rays(
            xlim=[1.9, 6.1], ylim=[0.9, 5.1],
            title='Plotting test 1 in 2D for all trajectories')
        plt.savefig('Plots_from_tests/'
                    + 'Plotting test 1 in 2D for all trajectories')
        plt.show()

    def test_plot2(self):
        """Two ray bundles, one parallel to the optical axis and the other does
           not, are propagated through a convex-plano lens

        The radius of the convex surface is 1 mm, and the focal length for the
        convex-plano surface is around 2.33 mm. The refractive index within the
        lens is 1.5
        """

        print('')
        print('-'*70)
        print('Plotting test 2')

        # define lens and ray bundles
        sph = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=1, n1=1, n2=1.5)
        pla = ele.SphericalRefraction(surf_cen=[0, 0, 1], cur=0, n1=1.5,
                                      n2=1)
        r1 = ra.RayBundle(radius=0.6, n=4, p_init=[-0.2, 0, -1],
                          k_init=[0.1, 0.1, 1])
        r2 = ra.RayBundle(radius=0.6, n=4, p_init=[0, 0, -1],
                          k_init=[0, 0, 1])
        r = [r1, r2]

        # put output plane at the focal length
        z_focus = ele.paraxial_focus(r, [sph, pla])
        output = ele.OutputPlane(z_focus, [sph, pla])

        # plot
        lens2 = [sph, pla, output]
        pnp.PropagateNPlot(r, lens2).plot3D(
            title='Plotting test 2 in 3D for both ray bundles')
        plt.savefig('Plots_from_tests/Plotting test 2 '
                    + 'in 3D for both ray bundles')
        plt.show()

        pnp.PropagateNPlot(r, lens2).plot2D_incident(
            xlim=[-0.85, 0.85], ylim=[-0.85, 0.85],
            title='Plotting test 2 in 2D for\nincident ray positions')
        plt.savefig('Plots_from_tests/Plotting test 2 '
                    + 'in 2D for incident ray positions')
        plt.show()

        pnp.PropagateNPlot(r, lens2).plot2D_output(
            xlim=[-0.35, 0.35], ylim=[-0.35, 0.35],
            title='Plotting test 2 in 2D for\nray positions on output plane')
        plt.savefig('Plots_from_tests/Plotting test 2 '
                    + 'in 2D for ray positions on output plane')
        plt.show()

        pnp.PropagateNPlot(r, lens2).plot2D_xz_ray_cen(
            xlim=[-1.3, 2.5], ylim=[-1.9, 1.9],
            title='Plotting test 2 in 2D for\ntrajectories along ray bundle '
            + 'diameter')
        plt.savefig('Plots_from_tests/Plotting test 2 '
                    + 'in 2D for trajectories along ray bundle diameter')
        plt.show()

    def test_plot3(self):
        """One ray bundle parallel to the optical axis is propagated through a
           concave-plano lens

        The radius of the concave surface is 1 mm, and the focal length for the
        convex-plano surface is around 2.83 mm. The refractive index within the
        lens is 1.5
        """

        print('')
        print('-'*70)
        print('Plotting test 3')

        # define lens and ray bundles
        sph = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=-1, n1=1,
                                      n2=1.5)
        pla = ele.SphericalRefraction(surf_cen=[0, 0, 0.5], cur=0, n1=1.5,
                                      n2=1)
        r3 = ra.RayBundle(radius=0.6, n=4, p_init=[0, 0, -2.5],
                          k_init=[0, 0, 1])
        r3 = [r3]

        # put output plane at the focal length
        z_focus = ele.paraxial_focus(r3, [sph, pla])
        output = ele.OutputPlane(z_focus, [sph, pla])

        # plot
        lens3 = [sph, pla, output]
        pnp.PropagateNPlot(r3, lens3).plot3D(
            title='Plotting test 3 in 3D for all trajectories')
        plt.savefig('Plots_from_tests/Plotting test 3 '
                    + 'in 3D for all trajectories')
        plt.show()

        pnp.PropagateNPlot(r3, lens3).plot2D_incident(
            xlim=[-2, 2], ylim=[-2, 2],
            title='Plotting test 3 in 2D for\nincident ray positions')
        plt.savefig('Plots_from_tests/Plotting test 3 '
                    + 'in 2D for incident ray positions')
        plt.show()

        pnp.PropagateNPlot(r3, lens3).plot2D_output(
            xlim=[-2, 2], ylim=[-2, 2],
            title='Plotting test 3 in 2D for\nray positions on output plane')
        plt.savefig('Plots_from_tests/Plotting test 3 '
                    + 'in 2D for ray positions on output plane')
        plt.show()

        pnp.PropagateNPlot(r3, lens3).plot2D_xz_ray_cen(
            xlim=[-3, 3], ylim=[-3, 3],
            title='Plotting test 3 in 2D for\ntrajectories along ray bundle '
            + 'diameter')
        plt.savefig('Plots_from_tests/Plotting test 3 '
                    + 'in 2D for trajectories along ray bundle diameter')
        plt.show()


class TestBiconvexOptimise(unittest.TestCase):
    """Test the methods in the BiconvexOptimise class"""

    def test_optimise_cur_n_dis(self):
        """Check, for a biconvex lens, the optimization of surface curvature in
           relation with a series of distances between the two surfaces"""

        print('')
        print('-'*70)
        print('Plotting test for biconvex optimization')

        bun = ra.RayBundle(radius=1, n=6, p_init=[0, 0, -15], k_init=[0, 0, 1])
        op = bi.BiconvexOptimise(ray_bundle=[bun], focal_len=100)

        print('')
        print('--- Optimise curvature for a list of distances ---\n')
        op.plot2D_rcParams()
        result = op.optimise_cur_n_dis(trial_cur=[0.01, -0.005],
                                       trial_dis_max=10, n=15,
                                       print_each=False)
        # show the result of the optimization above
        op.dis_vs_rms(op_result=result)

        print('')
        print('--- Comparing diffraction scale with RMS for a series of bundle'
              + ' raidii ---')
        # if do not want to print the RMS for each bundle, use out_all=False
        op.diff_vs_rms(op_result=result, n=15, lamb=588, rad_max=10, p_z0=-20,
                       out_single=False, out_all=True, test=True)


if __name__ == '__main__':
    unittest.main()
