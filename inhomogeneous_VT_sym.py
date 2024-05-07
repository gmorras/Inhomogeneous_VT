from astropy.io import fits
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
import pycbc.psd
import pycbc.detector
import tqdm
import astropy.coordinates 

import time
start_runtime = time.time() 


#constants
c = 299792458 #m/s
GM_sun = 1.32712440018e20 #m^3 s^-2
t_sun = GM_sun/(c**3)
Mpc_m = 3.085677581491367e22 #value of 1 Megaparsec (Mpc) in meters
H0 = 70 #km/s/Mpc
rH0 = (1e-3*c)/H0
Omega_m = 0.315 #matter density parameter

#dataset stuff
file_name = 'CF4gp_new_64-z008_delta.fits'
zmax = 0.08

#function to convert Galactic (l, b) coordinates to equatorial (ra, dec) coordinates
def Galacticlb_to_ICRSradec(l, b):

	#initialize galactic coordinates
	Galactic_coord = astropy.coordinates.SkyCoord(l=l, b=b, frame='galactic', unit='rad')

	#convert to ICRS
	ICRS_coord = Galactic_coord.icrs
	
	#return ra and dec
	return ICRS_coord.ra.radian, ICRS_coord.dec.radian

#function to transform cartesian to spherical coordinates
def Galacticlb_to_Cartesian(r, l, b):

	x = r*np.cos(b)*np.cos(l)
	y = r*np.cos(b)*np.sin(l)
	z = r*np.sin(b)

	return np.transpose([x, y, z])

#function to compute a(t) from arXiv:gr-qc/9804014
def a_func(t, g, l, d, a, ph, O_r):
	return (1/16)*np.sin(2*g)*(3-np.cos(2*l))*(3-np.cos(2*d))*np.cos(2*(a-ph-O_r*t))-(1/4)*np.cos(2*g)*np.sin(l)*(3-np.cos(2*d))*np.sin(2*(a-ph-O_r*t))+(1/4)*np.sin(2*g)*np.sin(2*l)*np.sin(2*d)*np.cos(a-ph-O_r*t)-(1/2)*np.cos(2*g)*np.cos(l)*np.sin(2*d)*np.sin(a-ph-O_r*t)+(3/4)*np.sin(2*g)*(np.cos(l)**2)*(np.cos(d)**2)

#function to compute b(t) from arXiv:gr-qc/9804014	
def b_func(t, g, l, d, a, ph, O_r):
	return np.cos(2*g)*np.sin(l)*np.sin(d)*np.cos(2*(a-ph-O_r*t))+(1/4)*np.sin(2*g)*(3-np.cos(2*l))*np.sin(d)*np.sin(2*(a-ph-O_r*t))+np.cos(2*g)*np.cos(l)*np.cos(d)*np.cos(a-ph-O_r*t)+(1/2)*np.sin(2*g)*np.sin(2*l)*np.cos(d)*np.sin(a-ph-O_r*t)

#function to compute the antenna pattern
def FpFc_func(t, g, l, d, a, ph, O_r, psi):
	
	#compute a and b
	af = a_func(t, g, l, d, a, ph, O_r)
	bf = b_func(t, g, l, d, a, ph, O_r)
	
	#compute Fp and Fc
	Fp = af*np.cos(2*psi) + bf*np.sin(2*psi)
	Fc = bf*np.cos(2*psi) - af*np.sin(2*psi)

	return Fp, Fc

#function to compute |Q(\theta, \phi, \iota)| from Maggiore Eq.(7.178)
def CBC_angular_pattern(t, g, l, d, a, ph, O_r, psi, iota):
	
	#compute the antenna patterns
	Fp, Fc = FpFc_func(t, g, l, d, a, ph, O_r, psi)
	
	#return |Q(\theta, \phi, \iota)| from Maggiore Eq.(7.178)
	return np.sqrt((0.5*(1+np.cos(iota)**2)*Fp)**2 + (np.cos(iota)*Fc)**2)

#function to compute comoving distance from redshift
def z_to_dc(z):
	return (1 - 0.75*Omega_m*z)*z*rH0

#function to compute luminosity distance from redshift
def z_to_dL(z):
	return (1 +(1-0.75*Omega_m)*z)*z*rH0

#function to compute redshift from luminosity distance
def dL_to_z(dL):
	return (np.sqrt(1 + (dL/rH0)*(4 - 3*Omega_m)) - 1)/(2 - 1.5*Omega_m)


#information of the interferometers (arXiv:gr-qc/9804014)
info_ifos = {}
info_ifos['H1'] = {'gamma': 171.8*(np.pi/180),
              'lambda': 46.45*(np.pi/180),
              'psd_func': pycbc.psd.analytical.aLIGOZeroDetHighPower,
              'psd_name': 'Design Sensitivity'
              }
info_ifos['L1'] = {'gamma': 243.0*(np.pi/180),
              'lambda': 30.56*(np.pi/180),
              'psd_func': pycbc.psd.analytical.aLIGOZeroDetHighPower,
              'psd_name': 'Design Sensitivity'
              }
info_ifos['V1'] = {'gamma': 116.5*(np.pi/180),
              'lambda': 43.63*(np.pi/180),
              'psd_func': pycbc.psd.analytical.AdvVirgo,
              'psd_name': 'Design Sensitivity'              
              }


#function to compute efficiency
def compute_efficiency(ifo, Ninj, decs, normed_dLs):

	#simulate the efficiency as a funcion of luminosity distance and declination 
	eficiency = np.zeros((len(decs), len(normed_dLs)))
	for i_d, d in enumerate(tqdm.tqdm(decs)):

		#generate random alpha, iota, psi
		a = np.random.uniform(0, 2*np.pi, size=Ninj)
		iota = np.arccos(np.random.uniform(-1, 1, size=Ninj))
		psi = np.random.uniform(0, 2*np.pi, size=Ninj)

		if type(ifo) == type(''):
			#compute DH =|Q(th, ph, iota)| for single ifo
			DH = CBC_angular_pattern(0, info_ifos[ifo]['gamma'], info_ifos[ifo]['lambda'], d, a, 0, 0, psi, iota)
		else:
			#compute sum_i |A_i|^2 |Q(th, ph, iota)|^2 and sum_i |A_i|^2
			DH2sum = 0
			A2sum = 0
			for i in ifo:
				A2sum = A2sum + (A_ifos[i])**2 
				DH2sum = DH2sum + (A_ifos[i]*CBC_angular_pattern(0, info_ifos[i]['gamma'], info_ifos[i]['lambda'], d, a, 0, 0, psi, iota))**2
			
			#compute DH = (sum_i |A_i|^2 |Q(th, ph, iota)|^2)/(sum_i |A_i|^2)
			DH = np.sqrt(DH2sum/A2sum)
		
		#compute efficiency
		eficiency[i_d, :] = np.sum(DH[:,np.newaxis]>normed_dLs[np.newaxis,:], axis=0)/Ninj

	return eficiency

#funtion join list of strings
def ifo2label(ifo):
	if type(ifo) == type(''): return ifo
	else: return ''.join(ifo)

#######################################

#declinations to consider
decs = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)

#normed luminosity distances
normed_dLs = np.linspace(0, 1.05, 200)

#number of injections per declination
Ninj = int(1e7) 

#SNR threashold
SNR_thres = 8

#frequencies
fmin=20
fmax=2048
delta_f = 0.25

#sight distances to consider (Mpc)
dsight_min = 0.01
N_dsight = 100

#number of points for Monte-Carlo
N_mc = int(1e7)

#ifo configurations to run on
ifos = ['H1', 'L1', 'V1', ['H1', 'L1', 'V1']]

#######################################

#load data
delta_fits = fits.open(file_name)

#extract data
delta = np.maximum(delta_fits[0].data, -1)
header = delta_fits[0].header

print('Range of deltas: %.3f to %.3f'%(np.amin(delta), np.amax(delta)))	
print('Fraction of pixels with delta<-1: %s/%s = %.3g'%(np.sum(delta<-1), len(delta.flatten()), np.sum(delta<-1)/len(delta.flatten())))

#number voxels
n_voxel = delta.shape[0]

#positions of points in grid
z_v = np.linspace(-zmax, zmax, n_voxel)

print('Number of voxels: %s^3'%(n_voxel))
print('Maximum comoving distance: %.1f Mpc/h'%(z_to_dc(zmax)))
print('Maximum luminosity distance: %.1f Mpc/h'%(z_to_dL(zmax)))
print('Length of voxel side: %.3f Mpc/h'%(rH0*(z_v[1]-z_v[0])))

#make 3D sicpy interpolator
delta_interp = RegularGridInterpolator((z_v, z_v, z_v), delta)

#maximum dsight to consider: (5/2)*dsight_max
dsight_max = (2/5)*z_to_dL(zmax)
dsights = np.linspace(dsight_min, dsight_max, N_dsight)

#loop over interferometers and compute the psds and amplitudes
Mc_max_all = -np.inf
A_ifos = {}
#for single interferometers
for ifo in info_ifos:

	#find the ifo PSD
	ifo_psd = info_ifos[ifo]['psd_func'](int(2*fmax/delta_f), delta_f, fmin/2)
	freqs = ifo_psd.sample_frequencies
	idxs = (freqs>=fmin) & (freqs<=fmax)
	freqs = freqs[idxs]
	ifo_psd = ifo_psd[idxs]
		
	#compute the r*SNR/|Q|/SNR_thr from Eq.(7.179) of Maggiore
	A_ifos[ifo] = (np.pi**(-2/3))*np.sqrt(5/6)*c*(t_sun**(5/6))*np.sqrt(np.trapz(freqs**(-7/3)/ifo_psd, x = freqs))/(Mpc_m*SNR_thres)

#for multiple interferometers, add 'A's in quadrature	
for ifo in ifos:
	if type(ifo) != type(''):
		A2sum = 0
		for i in ifo:
			A2sum = A2sum + (A_ifos[i])**2 
		A_ifos[ifo2label(ifo)] = np.sqrt(A2sum)
		
#loop over interferometer configurations
enh_VTs = {}
for ifo in ifos:

	print('\nSimulating efficiency for', ifo)

	#simulate the efficiency as a funcion of luminosity distance and declination 
	eficiency = compute_efficiency(ifo, Ninj, decs, normed_dLs)

	#make an interpolator of the efficiency
	eficiency_interp = RegularGridInterpolator((decs, normed_dLs), eficiency)
	
	#compute distance at which \max_\dec (eff) = 0
	max_dec_eficiency = np.amax(eficiency, axis=0)
	i_normed_dL_max_ifo = np.arange(len(normed_dLs))[max_dec_eficiency==0][0]-1
	normed_dL_max_ifo = normed_dLs[i_normed_dL_max_ifo]
	dL_max_ifo = A_ifos[ifo2label(ifo)]*normed_dL_max_ifo
	print('Maximum normed distance with some efficiency: %.5g (unnormed %.5g Mpc)'%(normed_dL_max_ifo, dL_max_ifo))

	#find the largest mass we can access
	Mc_max_ifo = (z_to_dL(zmax)/dL_max_ifo)**(6/5)
	
	#record the largest of all masses
	Mc_max_all = max(Mc_max_all, Mc_max_ifo)
	
	#maximum mass we can access
	print('Maximum chirp mass we can consistently integrate: Mc = %.5g Msun'%(Mc_max_ifo))
	
	#loop over chirp masses
	VT_homo = np.zeros(len(dsights))
	eVT_homo = np.zeros(len(dsights))
	VT_inhomo = np.zeros(len(dsights))
	eVT_inhomo = np.zeros(len(dsights))
	print("Computing VTs")
	for idsight, dsight in enumerate(tqdm.tqdm(dsights)):
	
		#compute maximum luminosity distance
		dL_max_i = (5/2)*dsight*normed_dL_max_ifo

		#compute the corresponding redshift
		z_max_i = dL_to_z(dL_max_i)

		#generate random galactic coordinates
		l = np.random.uniform(0, 2*np.pi, size=N_mc)
		b = np.arcsin(np.random.uniform(-1, 1, size=N_mc))
		z = np.random.uniform(0, z_max_i, size=N_mc)

		#compute delta
		delta_i = delta_interp(Galacticlb_to_Cartesian(z, l, b))

		#compute normed luminosity distance
		normed_dL = z_to_dL(z)/((5/2)*dsight)
		
		#convert to ICRS	
		ra, dec = Galacticlb_to_ICRSradec(l, b)
		
		#compute efficiency
		eff_i = eficiency_interp(np.transpose([dec, normed_dL]))

		#compute homogeneous and inhomogeneous VT
		integrand = (z**2)*eff_i*(1 - (3*Omega_m + 1)*z)
		VT_homo[idsight] = np.mean(integrand)
		eVT_homo[idsight] = np.std(integrand)/np.sqrt(N_mc - 1)
		integrand = (1+delta_i)*integrand
		VT_inhomo[idsight] = np.mean(integrand)
		eVT_inhomo[idsight] = np.std(integrand)/np.sqrt(N_mc - 1)
		
	#compute the ratio betweeen inhomogeneus and homogeneous 
	enh_VT = VT_inhomo/VT_homo
	e_enh_VT = enh_VT*((eVT_homo/VT_homo) + (eVT_inhomo/VT_inhomo))
	
	print('Maximum relative error in ratio:', np.amax(e_enh_VT/enh_VT))
	print('Variance in ratio:', np.std(enh_VT))

	#save this enhancement factor and Mc
	enh_VTs[ifo2label(ifo)] = enh_VT

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

#make a plot of all the enh_VT
fig, ax = plt.subplots(figsize=(12,12))
for i_ifo, ifo in enumerate(ifos):
	ax.plot(dsights, enh_VTs[ifo2label(ifo)], label=ifo2label(ifo), color='C%s'%(i_ifo))
	
ax.set_xlabel(r'$d_\mathrm{sight}$ [Mpc]')
ax.set_ylabel(r'$\langle VT \rangle_\mathrm{inhomo}/\langle VT \rangle_\mathrm{homo}$')
ax.set_xlim(dsights[0], dsights[-1])

#make supplementary axes
twins = list()
for i_ifo, ifo in enumerate(ifos):
	twins.append(ax.twiny())
	twins[i_ifo].spines.top.set_position(("axes", 1+0.075*i_ifo))
	twins[i_ifo].spines.top.set_color('C%s'%(i_ifo))
	twins[i_ifo].tick_params(axis='x', colors='C%s'%(i_ifo), labelsize=20)
	Mc_ticks = np.arange(0, (dsights[-1]/((2/5)*A_ifos[ifo2label(ifo)]))**(6/5), 0.1)
	Mc_tick_labels = ["%.1f"%(Mc_tick) for Mc_tick in Mc_ticks]
	twins[i_ifo].set_xticks((Mc_ticks**(5/6))*((2/5)*A_ifos[ifo2label(ifo)]), labels=Mc_tick_labels)
	twins[i_ifo].set_xlim(dsights[0], dsights[-1])

twins[-1].set_xlabel(r'$\mathcal{M}_c \; [M_\odot]$', fontsize=20)

ax.legend()
plt.tight_layout()
plt.savefig('enh_VT_dsight_ifos.png')
plt.savefig('enh_VT_dsight_ifos.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))


plt.show()


