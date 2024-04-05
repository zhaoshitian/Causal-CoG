for ds in vqav2 gqa vizwiz winoground mme mmbench_dev vsr okvqa
do
python debias.py \
    --logits_recorded_base_path_1 /home/ubuntu/stzhao/Causal-CoG/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1/${ds} \
    --logits_recorded_base_path_2 /home/ubuntu/stzhao/Causal-CoG/logits_recorded/llava_v1.5_7B_noisestep500/${ds} \
    --method contrast
done