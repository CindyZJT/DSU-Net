


import os
import SimpleITK as sitk
import numpy as np



def save_fnd(union_fnd,number):
    root_path = '/home/ext/chenzhihao/Datasets/StructSeg_2019/Lung_GTV/test'
    check_mkdir(root_path)
    union_path = os.path.join(root_path, str(number))
    check_mkdir(union_path)
    print(f'Saving predictions to: {union_path}')
    fnd_out = sitk.GetImageFromArray(union_fnd.astype(np.float32))

    sitk.WriteImage(fnd_out, '{}/fnd.nii.gz'.format(union_path))


def load_itk(fnd_path):

    for grid_fnd_file in os.listdir(fnd_path):
        itk_CT = sitk.ReadImage(os.path.join(fnd_path,grid_fnd_file))
        fnd_arr = sitk.GetArrayFromImage(itk_CT).astype(np.float32)

        # number = [grid_fnd_file[0:2] if int(grid_fnd_file[0:2]) else grid_fnd_file[0:1]][0]
        try:
            grid_number = int(grid_fnd_file[0:2])
        except ValueError:
            grid_number = int(grid_fnd_file[0:1])

        # union = union_Set(grid_fnd_arr, aspp_fnd_arr)
        save_fnd(fnd_arr, grid_number)


def main():
    grid_fnd_path = '/home/chenzhihao/MIP/code/3d_unet_v2/residual/0.002GDL/DS_test/fpd_nii/'
    grid_fnd_All = load_itk(grid_fnd_path)
    # aspp_fnd_All = load_itk(aspp_fnd_path)
    # union = union_Set(grid_fnd_All,aspp_fnd_All)
    # save_fpd(union)

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()
