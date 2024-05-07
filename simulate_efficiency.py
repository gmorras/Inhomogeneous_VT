import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})
import tqdm
import pycbc.psd

import time
start_runtime = time.time() 

#constants
c = 299792458 #m/s
GM_sun = 1.32712440018e20 #m^3 s^-2
t_sun = GM_sun/(c**3)
Mpc_m = 3.085677581491367e22 #value of 1 Megaparsec (Mpc) in meters

#function to compute a(t) from arXiv:gr-qc/9804014
def a_func(t, g, l, d, a, ph, O_r):
	return (1/16)*np.sin(2*g)*(3-np.cos(2*l))*(3-np.cos(2*d))*np.cos(2*(a-ph-O_r*t))-(1/4)*np.cos(2*g)*np.sin(l)*(3-np.cos(2*d))*np.sin(2*(a-ph-O_r*t))+(1/4)*np.sin(2*g)*np.sin(2*l)*np.sin(2*d)*np.cos(a-ph-O_r*t)-(1/2)*np.cos(2*g)*np.cos(l)*np.sin(2*d)*np.sin(a-ph-O_r*t)+(3/4)*np.sin(2*g)*(np.cos(l)**2)*(np.cos(d)**2)

#function to compute b(t) from arXiv:gr-qc/9804014	
def b_func(t, g, l, d, a, ph, O_r):
	return np.cos(2*g)*np.sin(l)*np.sin(d)*np.cos(2*(a-ph-O_r*t))+(1/4)*np.sin(2*g)*(3-np.cos(2*l))*np.sin(d)*np.sin(2*(a-ph-O_r*t))+np.cos(2*g)*np.cos(l)*np.cos(d)*np.cos(a-ph-O_r*t)+(1/2)*np.sin(2*g)*np.sin(2*l)*np.cos(d)*np.sin(a-ph-O_r*t)

#function to compute <F^2(t)> = <a^2 + b^2>
def avg_squared_antenna_pattern(g, l, d):
	return ((1787-72*np.cos(4*g)*(np.cos(l)**4)-444*np.cos(2*l)+9*np.cos(4*l))/4096)+((-111+40*np.cos(4*g)*(np.cos(l)**4)+332*np.cos(2*l)-5*np.cos(4*l))/1024)*np.cos(2*d)+((9-280*np.cos(4*g)*(np.cos(l)**4)-20*np.cos(2*l)+35*np.cos(4*l))/4096)*np.cos(4*d)
	
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

#information of the interferometers (arXiv:gr-qc/9804014)
info_ifos = {}
info_ifos['H1'] = {'gamma': 171.8*(np.pi/180),
              'lambda': 46.45*(np.pi/180),
              'psd_func': pycbc.psd.aLIGO140MpcT1800545,
              'psd_name': 'aLIGO140MpcT1800545'
              }
info_ifos['L1'] = {'gamma': 243.0*(np.pi/180),
              'lambda': 30.56*(np.pi/180),
              'psd_func': pycbc.psd.aLIGO140MpcT1800545,
              'psd_name': 'aLIGO140MpcT1800545'
              }
info_ifos['V1'] = {'gamma': 116.5*(np.pi/180),
              'lambda': 43.63*(np.pi/180),
              'psd_func': pycbc.psd.AdVO3LowT1800545,
              'psd_name': 'AdVO3LowT1800545'              
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

############################################################################################

#number of 'injections'
N_rand = int(1e6)

#deltas to consider
ds = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)

#normalized luminosity distances
DLs = np.linspace(0, 1, 200)

#SNR threashold
SNR_thres = 8

#frequencies
fmin=20
fmax=2048
delta_f = 0.25

#ifo configurations to run on
ifos = ['H1', 'L1', 'V1', ['H1', 'L1', 'V1']]

###########################################################################################

#load PSDs
A_ifos = {}
for ifo in info_ifos:

	#find the ifo PSD
	info_ifos[ifo]['psd'] = info_ifos[ifo]['psd_func'](int(2*fmax/delta_f), delta_f, fmin/2)
	freqs = info_ifos[ifo]['psd'].sample_frequencies
	idxs = (freqs>=fmin) & (freqs<=fmax)
	freqs = freqs[idxs]
	info_ifos[ifo]['psd'] = info_ifos[ifo]['psd'][idxs]
	
	#compute the r*SNR/|Q|/SNR_thr from Eq.(7.179) of Maggiore
	A_ifos[ifo] = (np.pi**(-2/3))*np.sqrt(5/6)*c*(t_sun**(5/6))*np.sqrt(np.trapz(freqs**(-7/3)/info_ifos[ifo]['psd'], x = freqs))/(Mpc_m*SNR_thres)

#make a plot of the psds
plt.figure(figsize=(16,10))
for ifo in info_ifos:
	plt.loglog(freqs, np.sqrt(info_ifos[ifo]['psd']), label=ifo)

plt.xlim(freqs[0], freqs[-1])
plt.xlabel(r'$f$ [Hz]')
plt.ylabel(r'ASD $\sqrt{S(f)}$ [$\sqrt{\mathrm{Hz}^{-1/2}}$]')
plt.legend()
plt.grid(visible=None, which='both')
plt.tight_layout()
plt.savefig('PSDs.png')

#loop over ifos
eff_dec_avg_dict = {}
for ifo in ifos:

	#if there are multiple ifos, compute the effective A
	if type(ifo) != type(''):
		A2sum = 0
		for i in ifo: A2sum = A2sum + (A_ifos[i])**2 
		A_ifos[ifo2label(ifo)] = np.sqrt(A2sum)

	#compute scaled distance to plot
	DLs_scaled = A_ifos[ifo2label(ifo)]*DLs
	
	#compute efficiency by doing 'injections'	
	eficiency = compute_efficiency(ifo, N_rand, ds, DLs)

	#compute the declination averaged efficiency
	eff_dec_avg_dict[ifo2label(ifo)] = 0.5*np.trapz(eficiency*np.cos(ds)[:,np.newaxis], x=ds, axis=0)

	#compute the sight distance
	d_sight = A_ifos[ifo2label(ifo)]*(2/5)
		
	#2D plot of efficiency
	plt.figure(figsize=(16,10))
	plt.contourf(DLs_scaled, ds, eficiency, np.linspace(0, 1, 11))
	plt.xlabel(r'$(\mathcal{M}_c/M_\odot)^{-5/6} d_L$ [Mpc]')
	plt.ylabel(r'$\delta$ [rad]')
	plt.colorbar(label=r'Efficiency $\epsilon(d_L, \delta)$')
	plt.axvline(x=d_sight, color='r', label='$d_\mathrm{sight}$')
	plt.xlim(DLs_scaled[0], DLs_scaled[-1])
	plt.ylim(ds[0], ds[-1])
	if type(ifo) == type(''):
		plt.title('%s, %s, $\\rho_\\mathrm{thr}$=%s -> $\\left(\\frac{\\mathcal{M}_c}{M_\\odot}\\right)^{-5/6} d_L^\\mathrm{sight} =$%.1fMpc'%(ifo, info_ifos[ifo]['psd_name'], SNR_thres, d_sight))
	else:
		plt.title('%s, $\\rho_\\mathrm{thr}$=%s -> $\\left(\\frac{\\mathcal{M}_c}{M_\\odot}\\right)^{-5/6} d_L^\\mathrm{sight} =$%.1fMpc'%(ifo, SNR_thres, d_sight))

	plt.legend()
	plt.tight_layout()
	plt.savefig(ifo2label(ifo)+'_efficiency.png')
	plt.savefig(ifo2label(ifo)+'_efficiency.pdf')
	
#plot declination averaged efficiencies
plt.figure(figsize=(16,10))
for ifo in ifos:
	plt.plot((5/2)*DLs, eff_dec_avg_dict[ifo2label(ifo)], label=ifo)

plt.axvline(x=1, color='k', label='$d_\mathrm{sight}$')

plt.ylabel(r'Declination averaged efficiency $\overline{\epsilon}(d_L)$')
plt.xlabel(r'$d_L/d_{L}^\mathrm{sight}$')
plt.xlim((5/2)*DLs[0], (5/2)*DLs[-1])
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig('efficiencies_dec_avg.png')
plt.savefig('efficiencies_dec_avg.pdf')


#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
