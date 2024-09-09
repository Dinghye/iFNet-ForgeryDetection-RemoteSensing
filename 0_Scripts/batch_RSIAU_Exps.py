''' Example
conda activate open-mmlab

python 0_Scripts/batch_RSIAU_Exps.py --gpu_id 1

'''
import os
import subprocess


# Basic Settings

version = '0_RSI_Authentication_v2'
# version = '0_RSI_Authentication_v2_SD_ISPRS'
# version = '0_RSI_Authentication_AntiDeepfake'
# version = '0_RSI_Authentication_ForenSynths'

CFG_DIR = f'configs/{version}'
WORKDIR = f'work_dirs/{version}'
LOG_DIR = f'2_Log/{version}'
EXP_ID = ''

MODEL_LIST = [
    # 'resnet18',

    # Compared methods of image space
    # 'vgg16',
    # 'densenet121_4xb256_in1k',
    # 'inception-v3_8xb32',
    # 'vit-base-p16_64xb64_in1k',
    # 'conformer-base-p16_8xb128_in1k',
    # 'resnet50',

    # Compared methods of freq space
    # 'freqnet50',
    # 'fcanet50',
    # 'fslnet',

    # Our method

    # 'resnet50_freq',
    # 'resnet50_freq_dual_v1',
    # 'resnet50_freq_dual_CBAM',
    # 'resnet50_freq_dual_CBAM_simple',
    # 'resnet50_freq_dual_CBAM_simple_v2',
    # 'resnet50_freq_dual_CBAM_v2',

    # 'sfnet_full_v3',
    # 'sfnet_freq_only',
    # 'sfnet_image_only',
    # 'sfnet_freq_only_wo_att',
    # 'sfnet_image_only_wo_att',
    # 'sfnet_two_branch_wo_att',
    'SFNet_wo_proj',
    # 'sfnet_full',
]


def get_best_ckpt_path(ckpt_dir):
    best_ckpt_name = None
    for fname in os.listdir(ckpt_dir):
        if fname.startswith('best_') and fname.endswith('.pth'):
            best_ckpt_name = fname
            break
    return best_ckpt_name


def main_run_exps(args):
    # Run exps
    for model in MODEL_LIST:
        exp_id = f'{model}'
        if EXP_ID:
            exp_id += f'_{EXP_ID}'

        # * Training *
        cmd_train = [
            'python', 'tools/train.py', '--config', f'{CFG_DIR}/{model}.py',
            '--work-dir', f'{WORKDIR}/{exp_id}',
        ]
        subprocess.run(cmd_train)

        # * Evaluation *
        best_ckpt_path = get_best_ckpt_path(f'{WORKDIR}/{exp_id}')
        if best_ckpt_path is None:
            print('Warning!!! Not best ckpt for ', exp_id)
            continue
        cmd_eval = [
            'python', 'tools/test.py', '--config', f'{CFG_DIR}/{model}.py',
            '--cfg-options', f'experiment_name={exp_id}',
            '--checkpoint', f'{WORKDIR}/{exp_id}/{best_ckpt_path}',
            '--work-dir', f'{LOG_DIR}',
            '--out', f'{LOG_DIR}/{exp_id}.pkl', '--out-item', 'pred',
            '--metrics_log_file', f'{LOG_DIR}/{exp_id}.log',
            '--metrics_log_file_all', f'{LOG_DIR}/0_All_Exps.log',
            # '--cfg-options', 'data.batch_size=64',
            # '--eval', 'bbox',
        ]
        subprocess.run(cmd_eval)

        # * Analysis *
        # Confusion matrix
        cmd_analysis = [
            'python', 'tools/analysis_tools/confusion_matrix.py',
            f'{CFG_DIR}/{model}.py',  # config path
            f'{LOG_DIR}/{exp_id}.pkl',  # ckpt_or_result
            '--show-path', f'{LOG_DIR}/{exp_id}_confusion_matrix.png'
        ]
        subprocess.run(cmd_analysis)
    print('Finished all.')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument('--ds_list', type=str, nargs='+',
    #                     default=None
    #                     # default=['nr', 'rsicb128']
    #                     )
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    main_run_exps(args)

    print('...')
