#CUDA_VISIBLE_DEVICES=1 .
python main_avvp.py --mode retrain --audio_dir feats/vggish/ --video_dir feats/res152/ --st_dir feats/r2plus1d_18 --checkpoint DHHN_Refinement --epoch 5
python main_avvp.py --mode test --audio_dir feats/vggish/ --video_dir feats/res152/ --st_dir feats/r2plus1d_18 --checkpoint DHHN_Refinement --no-log
