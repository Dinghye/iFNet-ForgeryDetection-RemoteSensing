''' Example
conda activate open-mmlab

python 0_Scripts/batch_RSIAU_Exps_analysis.py --gpu_id 1

'''
import os
import subprocess


# Basic Settings

version = '0_RSI_Authentication_v2'
# version = '0_RSI_Authentication_ForenSynths'

CFG_DIR = f'configs/{version}'
WORKDIR = f'work_dirs/{version}'
LOG_DIR = f'2_Log/{version}'
EXP_ID = ''

MODEL_AND_LAYERS = {
    # 'resnet18',

    # Compared methods of image space
    # 'vgg16',
    # 'inception-v3_8xb32',
    'resnet50': 'backbone.layer2.3',

    # Compared methods of freq space
    'freqnet50': 'backbone.layer2.3',
    # 'fcanet50',  #  TODO

    # Our method

    'sfnet_full': 'backbone.layer2.3',
}

SAMPLE_LIST = {
    # fail_0_real
    # r'freqnet50\analysis_top2000\fail_0_real\0000_image-659.png',
    # r'freqnet50\analysis_top2000\fail_0_real\0001_image-1059.png',
    # r'freqnet50\analysis_top2000\fail_0_real\0002_image-794.png',
    # r'freqnet50\analysis_top2000\fail_0_real\0006_image-1320.png',
    # r'freqnet50\analysis_top2000\fail_0_real\0007_image-883.png',
    # r'freqnet50\analysis_top2000\fail_0_real\0008_image-1018.png',
    # r'freqnet50\analysis_top2000\fail_0_real\0009_image-2.png',


    # fail_1_fake
    # r'freqnet50\analysis_top2000\fail_1_fake\0022_image-3664.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0023_image-1446.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0025_image-2760.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0051_image-1417.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0059_image-1728.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0099_image-4198.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0102_image-2228.png',

    # r'freqnet50\analysis_top2000\fail_1_fake\0024_image-4288.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0447_image-4631.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0553_image-4354.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0602_image-2867.png',
    # r'freqnet50\analysis_top2000\fail_1_fake\0554_image-4099.png',

    # success_1_fake
    r'freqnet50\analysis_top2000\success_1_fake\0772_image-4704.png',
    r'freqnet50\analysis_top2000\success_1_fake\1609_image-2614.png',
}


def get_best_ckpt_path(ckpt_dir):
    best_ckpt_name = None
    for fname in os.listdir(ckpt_dir):
        if fname.startswith('best_') and fname.endswith('.pth'):
            best_ckpt_name = fname
            break
    return best_ckpt_name


def main_run_exps(args):
    # Run exps
    for model, layers in MODEL_AND_LAYERS.items():
        exp_id = f'{model}'
        if EXP_ID:
            exp_id += f'_{EXP_ID}'

        topk = 2000
        # * analyze results *
        # cmd_analysis = [
        #     'python', 'tools/analysis_tools/analyze_results.py', '--config', f'{CFG_DIR}/{model}.py',
        #     '--result', f'{LOG_DIR}/{model}.pkl',
        #     # analye_settings
        #     '--topk', str(topk),
        #     '--out-dir', f'{WORKDIR}/{exp_id}/analysis_top{topk}'
        # ]
        # subprocess.run(cmd_analysis)

        # * analyze by Grad CAM *
        best_ckpt_path = get_best_ckpt_path(f'{WORKDIR}/{exp_id}')
        if best_ckpt_path is None:
            print('Warning!!! Not best ckpt for ', exp_id)
            continue

        for img_pth in SAMPLE_LIST:
            img_full_pth = f'{WORKDIR}/{img_pth}'
            save_dir = os.path.dirname(img_full_pth)

            cmd_gradcam = [
                'python', 'tools/visualization/vis_cam.py',
                '--img', img_full_pth,
                '--config', f'{CFG_DIR}/{model}.py',
                '--checkpoint', f'{WORKDIR}/{exp_id}/{best_ckpt_path}',
                '--target-layers', layers,  # TODO
                '--save-dir', f'{save_dir}_VisCAM_{layers}',
            ]
            subprocess.run(cmd_gradcam)

    print('Finished all.')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument('--ds_list', type=str, nargs='+',
    #                     default=None
    #                     # default=['nr', 'rsicb128']
    #                     )
    parser.add_argument('--gpu_id', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    main_run_exps(args)

    print('...')
