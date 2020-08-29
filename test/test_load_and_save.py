#%%

from logic.simple_itk_utils import load_itk, save_itk

volume00Array, volume00Origin, volume00Spacing = load_itk("dataset00/image00.mhd")

# print(volume00Array)
print(volume00Origin)
print(volume00Spacing)

#%%
# save_itk(volume00Array, volume00Origin, volume00Spacing, "dataset00/image00alex.mhd")

#%%
