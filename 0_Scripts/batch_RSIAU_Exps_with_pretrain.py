''' Example
conda activate open-mmlab

python 0_Scripts/batch_RSIAU_Exps_with_pretrain.py --gpu_id 1

'''
import os
import subprocess


# Basic Settings

CFG_DIR = 'configs/0_RSI_Authentication_v2'
WORKDIR = 'work_dirs/0_RSI_Authentication_v2_w_pt'
LOG_DIR = '2_Log/0_RSI_Authentication_v2_w_pt'
EXP_ID = ''
CKPT_DIR = 'N:/Model/RSI_Authentication/pretrained'

MODEL_LIST = [
    # 'resnet50',
    # 'resnet50_freq',
    # 'resnet50_freq_dual_v1',

    # 'resnet50_freq_dual_CBAM',
    # 'resnet50_freq_dual_CBAM_simple',
    # 'resnet50_freq_dual_CBAM_v2',
    # 'resnet50_freq_dual_CBAM_full',

    'sfnet_full'
]
CKPTS = {
    # 'ImageNet1K(SL)': ('resnet50_8xb32_in1k_mmpretrain-ea4938fc.pth', 'backbone'),
    # 'Million-AID(SL)': ('resnet50_8xb32_maid_rsp-last.pth.tar', ''),
    'Million-AID(SSL)': ('resnet50_8xb32_maid_simclr.pth.tar', ''),
    # 'ImageNet1K(SSL)': ('resnet50_8xb32_in1k_simclr.pth.tar', ''),
    # 'TOV-RS(SSL)': ('resnet50_8xb32_tov_22014162253-last.pth.tar', ''),
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
    for model in MODEL_LIST:
        for ckpt_name, (ckpt_pth, prefix) in CKPTS.items():
            exp_id = f'{model}_{ckpt_name}'
            if EXP_ID:
                exp_id += f'_{EXP_ID}'

            # * Training *
            cmd_train = [
                'python', 'tools/train.py', '--config', f'{CFG_DIR}/{model}.py',
                '--cfg-options', 'model.backbone.init_cfg.type=Pretrained',
                f'model.backbone.init_cfg.checkpoint={CKPT_DIR}/{ckpt_pth}',  # index=6
                # 'model.backbone.frozen_stages=1',
                '--work-dir', f'{WORKDIR}/{exp_id}',
            ]
            if prefix:
                cmd_train.insert(7, f'model.backbone.init_cfg.prefix={prefix}')
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
