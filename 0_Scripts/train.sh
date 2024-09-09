
# python tools/train.py --config configs/0_rsi_authentication/resnet50.py
# python tools/train.py --config configs/0_rsi_authentication/densenet121_4xb256_in1k.py
# python tools/train.py --config configs/0_rsi_authentication/vgg19.py
# python tools/train.py --config configs/0_rsi_authentication/vit-base-p16_4xb544-ipu_in1k.py
# python tools/train.py --config configs/0_rsi_authentication/swin-base_16xb64_in1k.py
# python tools/train.py --config configs/0_rsi_authentication/convnext-base_32xb128_in1k.py
# python tools/train.py --config configs/0_rsi_authentication/conformer-base-p16_8xb128_in1k.py


#第二次跑：
# python tools/train.py --config configs/0_rsi_authentication/resnet18.py
# python tools/train.py --config configs/0_rsi_authentication/resnet50.py
# python tools/train.py --config configs/0_rsi_authentication/resnet50_freq.py
python tools/train.py --config configs/0_rsi_authentication/resnet50_freq_dual_v1.py