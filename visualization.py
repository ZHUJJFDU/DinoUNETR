import numpy as np
import matplotlib.pyplot as plt
from toolkit import *
import os


data_path = r'C:\Users\960\Desktop\Top2\data\dataset_update_Lung\LUNG1-418+2Ac+MOS_57603.npz'
data_npz = np.load(data_path, allow_pickle=True)
data_dict = dict(data_npz)['arr_0'].item()

print(data_dict.keys())
print(data_dict['PTV'].shape)
# print(data_dict['comb_optptv'].shape,data_dict['electron_density'].shape,data_dict['mass_density'].shape,data_dict['comb_oar_distance'].shape)
# print (data_dict['isocenter'], data_dict['img'].shape, data_dict['beam_plate'].shape, data_dict['dose'].shape, data_dict['isVMAT'], data_dict['PTV'].shape , data_dict['spacing'])


# fig = plt.figure(figsize=(24, 8))

# ct = data_dict['comb_optptv']
# ax1 = fig.add_subplot(431)
# ax1.imshow(ct[int(data_dict['isocenter'][0])])
# ax1.axis('off')
# ax1.set_title('View from Axial plane')

# ax2 = fig.add_subplot(432)
# ax2.imshow(ct[::-1, int(data_dict['isocenter'][1]), :])
# ax2.axis('off')
# ax2.set_title('View from Sagittal plane')

# ax3 = fig.add_subplot(433)
# ax3.imshow(ct[::-1, ::-1, int(data_dict['isocenter'][2])])
# ax3.axis('off')
# ax3.set_title('View from Coronal plane')

# dose = data_dict['electron_density'] 

# ax4 = fig.add_subplot(434)
# ax4.imshow(dose[int(data_dict['isocenter'][0])], 'jet')
# ax4.axis('off')
# ax4.set_title('View from Axial plane')

# ax5 = fig.add_subplot(435)
# ax5.imshow(dose[::-1, int(data_dict['isocenter'][1]), :], 'jet')
# ax5.axis('off')
# ax5.set_title('View from Sagittal plane')

# ax6 = fig.add_subplot(436)
# ax6.imshow(dose[::-1, ::-1, int(data_dict['isocenter'][2])], 'jet')
# ax6.axis('off')
# ax6.set_title('View from Coronal plane')

# beam_plate = data_dict['mass_density']

# ax7 = fig.add_subplot(437)
# ax7.imshow(beam_plate[int(data_dict['isocenter'][0])])
# ax7.axis('off')
# ax7.set_title('View from Axial plane')

# ax8 = fig.add_subplot(438)
# ax8.imshow(beam_plate[::-1, int(data_dict['isocenter'][1]), :])
# ax8.axis('off')
# ax8.set_title('View from Sagittal plane')

# ax9 = fig.add_subplot(439)
# ax9.imshow(beam_plate[::-1, ::-1, int(data_dict['isocenter'][2])])
# ax9.axis('off')
# ax9.set_title('View from Coronal plane')

# ptv = data_dict['comb_oar_distance']
# ax10 = fig.add_subplot(4,3,10)
# ax10.imshow(ptv[int(data_dict['isocenter'][0])])
# ax10.axis('off')
# ax10.set_title('View from Axial plane')

# ax11 = fig.add_subplot(4,3,11)
# ax11.imshow(ptv[::-1, int(data_dict['isocenter'][1]), :])
# ax11.axis('off')
# ax11.set_title('View from Sagittal plane')

# ax12 = fig.add_subplot(4,3,12)
# ax12.imshow(ptv[::-1, ::-1, int(data_dict['isocenter'][2])])
# ax12.axis('off')
# ax12.set_title('View from Coronal plane')

# plt.show()