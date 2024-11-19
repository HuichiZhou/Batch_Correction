python main.py \
    --batch_size 8 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 10 \
    --data_path "/home/gyang/MAE-GAN/test_image" \
    --lr 1e-3 \
    --cuda 'cuda'