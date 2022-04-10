#CUDA_VISIBLE_DEVICES=1 .
python main_avvp.py --mode train --audio_dir ../feats/vggish/ --video_dir ../feats/res152/ --st_dir ../feats/r2plus1d_18 --checkpoint DHHN_Origin --epoch 10

python main_avvp.py --mode test --audio_dir ../feats/vggish/ --video_dir ../feats/res152/ --st_dir ../feats/r2plus1d_18 --checkpoint DHHN_Origin --no-log
