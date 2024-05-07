from astropy.io import fits
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
import tqdm
import astropy.coordinates 

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

import time
start_runtime = time.time() 


#constants
c = 299792.458 #km/s
H0 = 70 #km/s/Mpc

#dataset stuff
file_name = 'CF4gp_new_64-z008_delta.fits'
zmax = 0.08

#function to transform redshift to distance
def d_of_z(z):
	return (c/H0)*z

#function to transform distance to redshift
def z_of_d(d):
	return (H0/c)*d

#function to transform cartesian to spherical coordinates
def cartesian_to_spherical(r, th, phi):

	x = r*np.sin(th)*np.cos(phi)
	y = r*np.sin(th)*np.sin(phi)
	z = r*np.cos(th)

	return np.array([x, y, z])

#function to convert Galactic (l, b) coordinates to equatorial (ra, dec) coordinates
def Galacticlb_to_ICRSradec(l, b):

	#initialize galactic coordinates
	Galactic_coord = astropy.coordinates.SkyCoord(l=l, b=b, frame='galactic', unit='rad')

	#convert to ICRS
	ICRS_coord = Galactic_coord.icrs
	
	#return ra and dec
	return ICRS_coord.ra.radian, ICRS_coord.dec.radian

#function to convert equatorial (ra, dec) coordinates to Galactic (l, b) coordinates
def ICRSradec_to_Galacticlb(ra, dec):

	#initialize ICRS coordinates
	ICRS_coord = astropy.coordinates.SkyCoord(ra=ra, dec=dec, frame='icrs', unit='rad')

	#convert to galactic
	Galactic_coord = ICRS_coord.galactic
	
	#return l and b
	return Galactic_coord.l.radian, Galactic_coord.b.radian


#function to transform cartesian to spherical coordinates
def Galacticlb_to_Cartesian(r, l, b):

	x = r*np.cos(b)*np.cos(l)
	y = r*np.cos(b)*np.sin(l)
	z = r*np.sin(b)

	return np.transpose([x, y, z])


#######################################

#number of radiouses to plot
n_rad = 200 
n_mc = int(1e6)

#number of zs and declinations in 2d plot
N_z_2d = 100
N_d_2d = 50
N_mc_2d = int(1e4)

#redshift slices to plot
z_slices = [0.033, 0.049, 0.08]
Nth = 100
Nphi = 200
#######################################

#load data
delta_fits = fits.open(file_name)

print(delta_fits.info())

#extract data
delta = np.maximum(delta_fits[0].data, -1)
header = delta_fits[0].header

for head in header:
	print(head, '=', header[head])

print('Range of deltas: %.3f to %.3f'%(np.amin(delta), np.amax(delta)))	
print('Average delta:', np.mean(delta))
print('Fraction of pixels with delta<-1: %s/%s = %.3g'%(np.sum(delta<-1), len(delta.flatten()), np.sum(delta<-1)/len(delta.flatten())))

#number voxels
n_voxel = delta.shape[0]

#positions of points in grid
z_v = np.linspace(-zmax, zmax, n_voxel)

print('Number of voxels: %s^3'%(n_voxel))
print('Maximum distance: %.1f Mpc/h'%((c/H0)*zmax))
print('Length of voxel side: %.3f Mpc/h'%((c/H0)*(z_v[1]-z_v[0])))

#make 3D sicpy interpolator
delta_interp = RegularGridInterpolator((z_v, z_v, z_v), delta)

#compute radiouses at wich we want to compute the P(z) = integrate(delta dOmega)/(4*Pi)
zs = np.linspace(0, zmax, n_rad)
Ps = np.zeros(len(zs))
Ps_err = np.zeros(len(zs))

#compute also enhancement integrate((1 + delta) dOmega dz)*(3/(4*Pi*z**3))
enh = np.zeros(len(zs))
enh_err = np.zeros(len(zs))

#loop over radiouses
for iz, z in enumerate(tqdm.tqdm(zs)):
	
	#compute random distribution in theta and phi
	phi = np.random.uniform(0, 2*np.pi, size=n_mc)
	th = np.arccos(np.random.uniform(-1, 1, size=n_mc))
	
	#compute delta at x, y ,z
	d_i_r = delta_interp(np.transpose(cartesian_to_spherical(z, th, phi)))
	
	#compute P(r) and its error
	Ps[iz] = 1 + np.mean(d_i_r)
	Ps_err[iz] = np.std(d_i_r)/np.sqrt(len(d_i_r) - 1)
	
	#compute random distribution  uniform in z**3
	zrand = np.random.uniform(0, z**3, size=n_mc)**(1/3)
	
	#compute delta at x, y ,z
	d_i_r = delta_interp(np.transpose(cartesian_to_spherical(zrand, th, phi)))
	
	#compute enh and its error
	enh[iz] = 1 + np.mean(d_i_r)
	enh_err[iz] = np.std(d_i_r)/np.sqrt(len(d_i_r) - 1)

#compute integral d\alpha (1 + \delta) for different values of z and declination
zs_2d = np.linspace(0, zmax, N_z_2d)
ds_2d = np.linspace(-0.5*np.pi, 0.5*np.pi, N_d_2d)
ras = np.linspace(0, 2*np.pi, N_mc_2d)
dras = ras[1]-ras[0]

Pzds = np.zeros((N_d_2d, N_z_2d))

#loop over z
for iz, z in enumerate(tqdm.tqdm(zs_2d)):
	
	#loop over declination
	for idec, d in enumerate(ds_2d):
	
		#compute the Galactic coordinates
		l, b = ICRSradec_to_Galacticlb(ras, d*np.ones(N_mc_2d))
		
		#compute the delta
		d_i_r = delta_interp(Galacticlb_to_Cartesian(z, l, b))
		
		#compute the integral
		Pzds[idec, iz] = 1 + np.trapz(d_i_r, dx=dras)/(2*np.pi)
		
#plot delta in given redshift slices
b = np.linspace(-0.5*np.pi , 0.5*np.pi, Nth)
l = np.linspace(0 , 2*np.pi, Nphi)
L, B= np.meshgrid(l, b)
for z in z_slices:
	
	#compute delta at x, y ,z
	X, Y, Z = np.transpose(Galacticlb_to_Cartesian(z, L, B))
	d_i_r = delta_interp((X,Y,Z))
	
	plt.figure(figsize=(13,8))
	plt.pcolormesh((180/np.pi)*l, (180/np.pi)*b, d_i_r, vmin=-1, vmax=2.5)
	plt.xlabel(r'Galactic longitude $l$ [$^\circ$]')
	plt.ylabel(r'Galactic latitude $b$ [$^\circ$]')
	plt.title(r'Redshift  $z=%s$'%(z))
	plt.colorbar(label=r'$\delta$')
	plt.xlim(0, 360)
	plt.ylim(-90,90)
	
	plt.tight_layout()
	plt.savefig('delta_slice_z_%s_2211.16390.png'%(z))

'''
#plot delta in given redshift slices
dec = np.linspace(-0.5*np.pi , 0.5*np.pi, Nth)
ra = np.linspace(0 , 2*np.pi, Nphi)
RA, DEC = np.meshgrid(ra, dec)
L, B = ICRSradec_to_Galacticlb(RA, DEC)
for z in z_slices:
	
	#compute delta at x, y ,z
	X, Y, Z = np.transpose(Galacticlb_to_Cartesian(z, L, B))
	d_i_r = delta_interp((X,Y,Z))

	plt.figure(figsize=(13,8))
	plt.pcolormesh((24/360)*(180/np.pi)*ra, (180/np.pi)*dec, d_i_r, vmin=-1, vmax=2.5)
	plt.ylabel(r'Declination $\delta$ [$^\circ$]')
	plt.xlabel(r'Right Ascension $\alpha$ [h]')
	plt.title(r'Redshift  $z=%s$'%(z))
	plt.colorbar(label=r'$\delta$')
	plt.xlim(0, 24)
	plt.ylim(-90,90)
	
	plt.tight_layout()
	plt.savefig('delta_slice_z_%s_2211.16390.png'%(z))
'''

#make a histogram of delta
plt.figure(figsize=(13,8))
histo2, bins2, _ =plt.hist(delta.flatten(), bins=100)

#fit a gaussian
histo, bin_edges = np.histogram(delta.flatten(), bins=np.linspace(-0.1,0.1,100))
bins = 0.5*(bin_edges[1:]+bin_edges[:-1])
poly_coefs = np.polyfit(bins, np.log(np.amax(histo2)*histo/np.amax(histo)), 2)
x = np.linspace(-1,1,1000)
plt.plot(x, np.exp(np.poly1d(poly_coefs)(x)), label='Gaussian Approximation')
plt.xlabel(r'$\delta_M$')
plt.ylabel(r'Number of voxels')
plt.ylim(bottom=0.9)
plt.xlim(np.amin(delta), np.amax(delta))
plt.yscale('log')
plt.grid(visible=True, which='major')
plt.legend()
plt.tight_layout()
plt.savefig('delta_histo_2211.16390.png')
plt.savefig('delta_histo_2211.16390.pdf')

#make plot of Ps
fig, ax = plt.subplots(figsize=(13,8))
ax.plot(zs, Ps)
ax.axhline(y=1, color='k')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$P(z) = \frac{1}{4\pi} \int d\Omega (1 + \delta_M)$')
ax.set_xlim(0, zmax)
secax = ax.secondary_xaxis('top', functions=(d_of_z, z_of_d))
secax.set_xlabel(r'$d$ [Mpc/$h_{%s}$]'%(H0))
plt.grid(visible=True, which='major')
plt.tight_layout()
plt.savefig('Pz_2211.16390.png')
plt.savefig('Pz_2211.16390.pdf')

#make plot of int_Ps
fig, ax = plt.subplots(figsize=(13,8))
#ax.plot(zs, cumulative_trapezoid(Ps, x=zs**3, initial=0)/(zs**3))
ax.plot(zs, enh)
ax.axhline(y=1, color='k')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\frac{3}{4\pi z^3} \int_0^z z^2 dz \int d\Omega (1 + \delta_M)$')
ax.set_xlim(0, zmax)
secax = ax.secondary_xaxis('top', functions=(d_of_z, z_of_d))
secax.set_xlabel(r'$d$ [Mpc/$h_{%s}$]'%(H0))
plt.tight_layout()
plt.savefig('int_Pz_2211.16390.png')

#make plot of P(z,d)
fig, ax = plt.subplots(figsize=(16,10))
cmap = ax.contourf(zs_2d, ds_2d, Pzds)
ax.set_xlabel(r'Reshift $z$')
ax.set_ylabel(r'Declination $\delta$')
ax.set_xlim(0, zmax)
ax.set_ylim(-0.5*np.pi, 0.5*np.pi)
plt.colorbar(mappable=cmap, ax=ax, label=r'$P(z,\delta) = \frac{1}{2\pi} \int_{0}^{2\pi} d\alpha (1 + \delta_M(z,\delta,\alpha))$')
secax = ax.secondary_xaxis('top', functions=(d_of_z, z_of_d))
secax.set_xlabel(r'$d$ [Mpc/$h_{%s}$]'%(H0))
plt.tight_layout()
plt.savefig('Pzdelta_2211.16390.png')


#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))


plt.show()


