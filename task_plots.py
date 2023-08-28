"""
Plots required in the tasks of the script
"""
import numpy as np
import matplotlib.pyplot as plt

import ray as ra
import optical_element as ele
import propagate_and_plot as pnp
import biconvex_optimization as bi
# %%
###############################################################################
# T11 - Plot(s) for any additional simple test-cases you devised.
print('-'*70)
print('--- Simple test case ---')
print('\nRays parallel to optical axis and rays through sphere centre')
print('\nThey would intercept at z = 4 mm')
b1 = ra.RayBundle(radius=0.6, n=2, p_init=[0, 0, -5], k_init=[0, 0, 1])
b2 = ra.RayBundle(radius=0.2, n=1, p_init=[0.6, 0, -5], k_init=[-0.1, 0.1, 1])
b3 = ra.RayBundle(radius=0.2, n=1, p_init=[-0.6, 0, -5], k_init=[0.1, 0.1, 1])

surf2 = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=1, n1=1, n2=1.5)

output2 = ele.OutputPlane(4, [surf2])

bund2 = [b1, b2, b3]
all2 = [surf2, output2]

z_focus2 = ele.paraxial_focus(bund2, [surf2])
pnp.PropagateNPlot(bund2, all2).plot2D_xz_ray_cen(
    title='Simple test: rays through sphere centre\nof the surface with f=3mm')
plt.savefig('Plots_from_tasks/Task 11 - Simple test')
plt.show()

print('-'*70)
# %%
###############################################################################
# T9 - Tracing the trajectory of a few example rays through the specified
# spherical surface.
print('-'*70)
print('--- Trace a few rays through a single specified surface ---')
bund1 = ra.RayBundle(radius=30, n=2, p_init=[0, 0, 60], k_init=[0, 0, 1])

surf1 = ele.SphericalRefraction(surf_cen=[0, 0, 100], cur=0.03, n1=1, n2=1.5)

output1 = ele.OutputPlane(250, [surf1])

bund1 = [bund1]
all1 = [surf1, output1]

z_focus1 = ele.paraxial_focus(bund1, [surf1])
pnp.PropagateNPlot(bund1, all1).plot2D_xz_ray_cen(
    xlim=[55, 255], ylim=[-100, 100], title='Few rays parallel to the '
    + 'optical axis\nthrougth a single surface with f=100mm')
plt.savefig('Plots_from_tasks/Task 9 - Few rays through single surface')
plt.show()

print('-'*70)
# %%
###############################################################################
# T12 - Tracing a large-diameter uniform bundle of collimated rays through the
# specified spherical surface.
print('-'*70)
print('--- Single refracting surface ---')
bund3 = ra.RayBundle(radius=5, n=6, p_init=[0, 0, -20], k_init=[0, 0, 1])

surf3 = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=0.03, n1=1, n2=1.5168)

bund3 = [bund3]
surf_l3 = [surf3]

z_focus3 = ele.paraxial_focus(bund3, surf_l3)
output3 = ele.OutputPlane(z_focus3, surf_l3)

all3 = [surf3, output3]

foc_pri3 = np.around(z_focus3, 3)
pnp.PropagateNPlot(bund3, all3).plot3D(
    title='3D plot for a non-paraxial uniform ray bundle\nthrough a single '
    + f'surface with f={foc_pri3}mm')
plt.savefig('Plots_from_tasks/Task 12 - 3D plot for all rays')
plt.show()

pnp.PropagateNPlot(bund3, all3).plot2D_xz_ray_cen(
    xlim=[-25, 105], ylim=[-65, 65],
    title='2D plot for a non-paraxial uniform ray bundle\nthrough a single '
    + f'surface with f={foc_pri3}mm')
plt.savefig('Plots_from_tasks/Task 12 - 2D plot for all rays')
plt.show()
###############################################################################
# T13 - Corresponding spot diagram for the bundle of rays, at the paraxial
# focal plane.
pnp.PropagateNPlot(bund3, all3).plot2D_incident(
    title='Ray positions of the uniform\ncollimated beam with raidus of 5 mm')
plt.savefig('Plots_from_tasks/Task 13 - incident rays')
plt.show()

pnp.PropagateNPlot(bund3, all3).plot2D_output(title='Ray positions on output '
                                              + 'plane\nfor a single surface '
                                              + f'with f={foc_pri3}mm')
plt.savefig('Plots_from_tasks/Task 13 - output rays')
plt.show()

diff3 = pnp.PropagateNPlot(bundle_list=bund3).diffraction_scale(
    focal_length=z_focus3)
print(f'\nDiffraction scale with bundle radius of {bund3[0].radius()} mm is '
      + f'{diff3[0]} mm')

print('-'*70)
# %%
###############################################################################
# T15 - A plot illustrating ray trajectories for the plano-convex lens in both
# orientations.
print('-'*70)
bund4 = ra.RayBundle(radius=5, n=6, p_init=[0, 0, -20], k_init=[0, 0, 1])
bund4 = [bund4]

print('--- Convex-plano lens ---')
surf4 = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=0.02, ap=20,
                                n1=1, n2=1.5168)
plano4 = ele.SphericalRefraction(surf_cen=[0, 0, 5], cur=0, ap=20,
                                 n1=1.5168, n2=1)

conv_plan = [surf4, plano4]

z_focus_c_p = ele.paraxial_focus(bund4, conv_plan)
output4 = ele.OutputPlane(z_focus_c_p, conv_plan)

c_p_out = [surf4, plano4, output4]

foc_pri4 = np.around(z_focus_c_p, 3)
pnp.PropagateNPlot(bund4, c_p_out).plot2D_xz_ray_cen(
    xlim=[-25, 105], ylim=[-65, 65],
    title='Ray trajectories of the convex-plano\norientation with f='
          + f'{foc_pri4}mm')
plt.savefig('Plots_from_tasks/Task 15 - convex-plano')
plt.show()

RMS_c_p = pnp.PropagateNPlot(bund4, c_p_out).RMS()[0]
print(f'\nRMS spot radius for ray bundle No. 1 is {RMS_c_p} mm')

diff5 = pnp.PropagateNPlot(bundle_list=bund4).diffraction_scale(
    focal_length=z_focus_c_p)
print(f'\nDiffraction scale with bundle radius of {bund4[0].radius()} mm is '
      + f'{diff5[0]} mm')
print('-'*70)


print('-'*70)
print('--- Plano-convex lens ---')
plano5 = ele.SphericalRefraction(surf_cen=[0, 0, 0], cur=0, ap=20,
                                 n1=1, n2=1.5168)
surf5 = ele.SphericalRefraction(surf_cen=[0, 0, 5], cur=-0.02, ap=20,
                                n1=1.5168, n2=1)

plan_conv = [plano5, surf5]

z_focus_p_c = ele.paraxial_focus(bund4, plan_conv)
output5 = ele.OutputPlane(z_focus_p_c, plan_conv)

p_c_out = [plano5, surf5, output5]

foc_pri5 = np.around(z_focus_p_c, 3)
pnp.PropagateNPlot(bund4, p_c_out).plot2D_xz_ray_cen(
    xlim=[-25, 105], ylim=[-65, 65],
    title='Ray trajectories of the plano-convex\norientation with f='
          + f'{foc_pri5}mm')
plt.savefig('Plots_from_tasks/Task 15 - plano-convex')
plt.show()

RMS_p_c = pnp.PropagateNPlot(bund4, p_c_out).RMS()[0]
print(f'\nRMS spot raidus for ray bundle No. 1 is {RMS_p_c} mm')

diff5 = pnp.PropagateNPlot(bundle_list=bund4).diffraction_scale(
    focal_length=z_focus_p_c)
print(f'\nDiffraction scale with bundle radius of {bund4[0].radius()} mm is '
      + f'{diff5[0]} mm')
###############################################################################
# T15 - Plot(s) or table presenting the performance of the plano-convex lens.
pnp.PlanoConvex(c_p=conv_plan,
                p_c=plan_conv).plano_convex_orient(title='Performance '
                                                   + 'comparison for\na plano-'
                                                   + 'convex lens with '
                                                   + 'f \u2248 100mm',
                                                   print_foc=False)
plt.savefig('Plots_from_tasks/Task 15 - comparison')
plt.show()

print('-'*70)
# %%
###############################################################################
# Lens Opt. - Plot(s) or table(*) relevant to this task.
# NOTE: the previous block has to be run before running this block
print('-'*70)
print('--- Optimization for f = 500 mm at one selected distance ---\n')
bund6 = ra.RayBundle(radius=5, n=6, p_init=[0, 0, -20], k_init=[0, 0, 1])

op6 = bi.BiconvexOptimise(ray_bundle=[bund6], focal_len=z_focus_c_p)

dis = 5  # mm
cur6 = op6.optimise_cur(trial_cur=[0.02, -0.0015], trial_dis=dis)
rms6 = op6.biconvex(trial_cur=cur6, trial_dis=dis)

diff6 = pnp.PropagateNPlot(bundle_list=[bund6]).diffraction_scale(
    focal_length=z_focus_c_p)
print(f'\nDiffraction scale with bundle radius of {bund6.radius()} mm is '
      + f'{diff6[0]} mm')

result6 = ([cur6], [dis], [rms6], 0)
op6.diff_vs_rms(op_result=result6, n=120, rad_max=20, p_z0=-20, out_all=False,
                out_single=False)

print('-'*70)
