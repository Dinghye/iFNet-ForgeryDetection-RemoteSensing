''' Example
conda activate open-mmlab

python 0_Scripts/batch_RSIAU_Exps_data_augment.py --gpu_id 1

'''
import os
import subprocess


# Basic Settings

version = '0_RSI_Authentication_v2_data_augment'

CFG_DIR = f'configs/{version}'
WORKDIR = f'work_dirs/{version}'
LOG_DIR = f'2_Log/{version}'
EXP_ID = ''

MODEL_LIST = [
    # 'sfnet_full',
    # '10_Randomcrop_weak',
    # '11_Randomcrop',
    # '12_Randomcrop_Filp',
    # '13_Randomcrop_Blur',
    # '14_Randomcrop_Erase',
    # '15_Randomcrop_Color',
    # '22_Filp',
    # '23_Filp_Blur',
    # '24_Filp_Erase',
    # '25_Filp_Color',
    # '33_Blur',
    # '34_Blur_Erase',
    # '35_Blur_Color',
    # '44_Erase',
    # '45_Erase_Color',
    '55_Color',
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
    parser.add_argument('--gpu_id', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    main_run_exps(args)

    print('...')
