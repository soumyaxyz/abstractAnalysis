arxiv 	= [0.2946236559,	0.5935483871,	0.5956989247,	0.6344086022,	0.6387096774,	0.651214128,	0.6521739130434783]
ax 		= [0,				10,				20,				40,				70,				90, 			110]
TLT 	= [0.5686900958,	0.6549520767,	0.7156549521,	0.7444089457,	0.7763578275,	0.7827476038,	0.7944785276073619 ]
Tx 		= [0,				10,				20,				40,				70,				90,				110]
TPAMI 	= [0.5015873016,	0.726984127,	0.7523809524,	0.7682539683,	0.7968253968,	0.8222222222,	0.8343558282208589]
Tpx 	= [0,				10,				20,				40,				70,				90,				110]
merged 	= [0.408719346, 	0.5449591281, 	0.5903723887, 	0.6475930972, 	0.659400545, 	0.6930063579, 	0.7093551317, 	0.7111716621, 	0.7356948229, 	0.736603088101725, 	0.7438692098, 	0.7517985612]
mx 		= [0, 				10, 			20, 			40, 			70, 			90, 			110, 			150, 			190, 			230, 				280, 			330	]


from scipy.ndimage.filters import gaussian_filter1d

def f(y, sigma=0):
	if sigma == 0:
		return y
	else:
		return gaussian_filter1d(y, sigma)


# ysmoothed = gaussian_filter1d(y, sigma=2)
# plt.plot(x, ysmoothed)
# plt.show()

import matplotlib.pyplot as plt  



plt.figure()  
plt.axes()   
plt.ylim(0.25,0.85)
# 
plt.plot(ax, f(arxiv), label='cs.NI')
plt.plot(Tx, f(TLT), label='cs.TLT')
plt.plot(Tpx, f(TPAMI), label='cs.TPAMI')
plt.plot(mx, f(merged), label='cs.combined')
plt.legend(loc=4,  ncol=2)
# plt.title('Accuracy vs Traninig size')
plt.xlabel('Traninig size')
plt.ylabel('Accuracy')
plt.savefig('acc_vs_trn_size.png')
plt.ylim(0,1)
plt.savefig('acc_vs_trn_size_alt.png')  
plt.show()                                                                                                                                          