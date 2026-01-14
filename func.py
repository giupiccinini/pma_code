import numpy as np
import radvel

#------------Proper Motion Anomaly-------------

def PMA_func(params):

	A = ax*(cos(w)*cos(W)-sin(w)*sin(W)*cos_i)
	B = ax*(cos(w)*sin(W)+sin(w)*cos(W)*cos_i)
	F = ax*(-sin(w)*cos(W)-cos(w)*sin(W)*cos_i)
	G = ax*(-sin(w)*sin(W)+cos(w)*cos(W)*cos_i)

	M_an = 2*pi*(((T-T0)/P) - np.floor((T-T0)/P))	
	E_an = radvel.kepler.kepler(M_an, Ec)

	x = A*(cos(E_an)-e) + F*sqrt(1.-e**2)*sin(E_an) 
	y = B*(cos(E_an)-e) + G*sqrt(1.-e**2)*sin(E_an)

	E_der = (2*pi/P)*(1./(1.-e*cos(E_an)))
        
	x1 = E_der*(-A*sin(E_an)+F*cos(E_an)*sqrt(1.-e**2))   
	y1 = E_der*(-B*sin(E_an)+G*cos(E_an)*sqrt(1.-e**2))

	err_dmu_dec_H = sqrt(x1_H**2+(x_G/t_HG)**2+(x_H/t_HG)**2)
	err_dmu_dec_G = sqrt(x1_G**2+(x_G/t_HG)**2+(x_H/t_HG)**2)
	err_dmu_ra_H = sqrt(y1_H**2+(y_G/t_HG)**2+(y_H/t_HG)**2)
	err_dmu_ra_G = sqrt(y1_G**2+(y_G/t_HG)**2+(y_H/t_HG)**2)

	return np.asarray([x1_H-(x_G-x_H)/t_HG, x1_G-(x_G-x_H)/t_HG]), np.asarray([y1_H-(y_G-y_H)/t_HG, y1_G-(y_G-y_H)/t_HG])


#---------------Radial Velocity------------ 

def RV_mod(pars, t_V):
	RV_model = radvel.kepler.rv_drive(t_V,pars) + RV_off
		return RV_model, RV_off
	
#--------Likelihood-------------

def log_likelihd(params, t_V):	
	chi2 = sum(((PMA_u_dec - delta_u_dec)/err_dec)**2 + ((PMA_u_ra - delta_u_ra)/err_ra)**2) 
	chi2 += sum((((RV_model - data)**2)/err2) + log(2*pi*err2))
	return -0.5*(chi2)
	
