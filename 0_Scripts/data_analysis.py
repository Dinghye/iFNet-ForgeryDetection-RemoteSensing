import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fftpack
from mpl_toolkits.axes_grid1 import make_axes_locatable


def dct2(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def fft2d(array):
    array = fftpack.fft2(array)
    array = fftpack.fftshift(array)
    return array


def log_scale(array, epsilon=1e-12):
    """Log scale the input array.
    """
    array = np.abs(array)
    array += epsilon  # no zero in log
    array = np.log(array)
    return array


def _dct2_wrapper(image, log=False):
    image = np.asarray(image)
    image = dct2(image)
    if log:
        image = log_scale(image)

    return image


def load_images_from_folder(folder, size, grayscale=True, num=50):
    size = (size, size)
    images = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        img = Image.open(os.path.join(folder, filename))
        if grayscale:
            img = img.convert('L')
        img = img.resize(size)
        images.append(np.array(img))
        if i == num-1:
            break
    return images


# def dct2(array):
#     return fftpack.dct(fftpack.dct(array.T, norm='ortho').T, norm='ortho')


def calculate_average_dct(images):
    sum_dct = None
    for image in images:
        dct_image = dct2(np.float32(image))
        if sum_dct is None:
            sum_dct = dct_image
        else:
            sum_dct += dct_image

    sum_dct = sum_dct / len(images)

    # 应用对数缩放
    dct_log = np.log(np.abs(sum_dct) + 1e-5)

    return dct_log


def plot_dcts(real_dct, fake_dct, dst_pth, colormap='jet', title_prefix='Average DCT of'):

    # Set up the figure and the subplots grid
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Plot the real DCT image
    im1 = ax1.imshow(real_dct, cmap=colormap, aspect='auto')
    ax1.set_title(f"{title_prefix} Real Images")
    ax1.axis('off')  # Hide the axis

    # Plot the fake DCT image
    im2 = ax2.imshow(fake_dct, cmap=colormap, aspect='auto')
    ax2.set_title(f"{title_prefix} Fake Images")
    ax2.axis('off')  # Hide the axis

    # Create an axis for the colorbar to the right of ax2
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    # Save the combined figure
    plt.tight_layout()
    fig.savefig(f"{dst_pth}.svg", format='svg', bbox_inches='tight')
    plt.close(fig)


def plot_dcts_2(real_dct, fake_dct, dst_pth, colormap='jet', title_prefix='Average DCT of'):

    # Define the figure and axes
    fig_real, ax_real = plt.subplots()
    fig_fake, ax_fake = plt.subplots()
    fig_colorbar, ax_colorbar = plt.subplots(figsize=(1, 6))

    # Plot the real DCT
    im_real = ax_real.imshow(real_dct, cmap=colormap, aspect='auto')
    ax_real.axis('off')
    fig_real.savefig(f"{dst_pth}_Real.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig_real)

    # Plot the fake DCT
    im_fake = ax_fake.imshow(fake_dct, cmap=colormap, aspect='auto')
    ax_fake.axis('off')
    fig_fake.savefig(f"{dst_pth}_Fake.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig_fake)

    # Create colorbar for the second plot (fake DCT)
    cb = fig_colorbar.colorbar(im_fake, cax=ax_colorbar)
    ax_colorbar.remove()
    fig_colorbar.savefig(f"{dst_pth}_ColorBar.svg", format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig_colorbar)


def main():
    sample_num = 10000
    # EXP_ID = {
    #     1: 'resnet50', 2: 'freqnet50',
    # }[2]
    # state = ['success', 'fail'][1]
    im_size = 256

    plot_title_prefix = f'Average DCT of {sample_num}'
    # dst_pth = f'work_dirs/0_RSI_Authentication_v2/{EXP_ID}/analysis_top2000/dct_analys_of_{sample_num}_{state}_samples_imsize{im_size}'
    # real_folder = f'work_dirs/0_RSI_Authentication_v2/{EXP_ID}/analysis_top2000/{state}_0_real'  # 真实图像文件夹路径
    # fake_folder = f'work_dirs/0_RSI_Authentication_v2/{EXP_ID}/analysis_top2000/{state}_1_fake'  # 生成图像文件夹路径

    real_folder = r'D:\Classification\SDGen-Detection\SD_Potsdam\0_real'  # 真实图像文件夹路径
    fake_folder = r'D:\Classification\SDGen-Detection\SD_Potsdam\1_fake'  # 生成图像文件夹路径
    dst_pth = r'D:\Classification\SDGen-Detection\SD_Potsdam_samples_imsize' + str(im_size)

    real_images = load_images_from_folder(real_folder, im_size, num=sample_num)
    fake_images = load_images_from_folder(fake_folder, im_size, num=sample_num)

    real_dct = calculate_average_dct(real_images)
    fake_dct = calculate_average_dct(fake_images)
    plot_dcts(real_dct, fake_dct, dst_pth, title_prefix=plot_title_prefix)
    # plot_dcts_2(real_dct, fake_dct, dst_pth, title_prefix=plot_title_prefix)
    print('Finished...')


if __name__ == "__main__":
    print('...')
    main()
