path='/home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1'

for ds in pope_adversarial pope_random pope_popular
do
for modelpath in ${path}
do
python /home/ubuntu/stzhao/Causal-CoG/inference.py \
    --model-path ${modelpath} \
    --ds ${ds} \
    --cuda "0,1,2,3" \
    --batchsize 1 \
    --use_cog False
done
done

# pope_adversarial pope_random pope_popular mme vsr okvqa vqav2 vizwiz gqa winoground seedbench mmbench_dev
# /home/ubuntu/stzhao/LLaVA_1.1.1/LLaVA/llava_1.5_weight/llava_v1.5_7B
# /home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1
# /home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Vucina-13B
