import os, glob, json, statistics
ROOT='/hdd1/an34232/entrgi_sft_results'
#MODEL='dream-chosen-trunc128-sft-lora-r32-alllinear-500steps_checkpoint-500'
#MODEL='dream-entrgi-sft-lora-r32-alllinear-500steps_checkpoint-500'
MODEL='dream-grpo-wildchat-skywork-r32-alllinear-500steps_checkpoint-500'
#MODEL='dream-entrgi-wildchat-skywork-r32-alllinear-500steps_checkpoint-500'

datasets=['judgebench','reward-bench-2','rm-bench', 'wildchat-heldout']

print('dataset,base_mean,wc200_mean,delta_wc200_minus_base,base_std,wc200_std,n_base,n_wc200')
b_over=[]; w_over=[]
for ds in datasets:
    b=[]; w=[]
    for p in sorted(glob.glob(os.path.join(ROOT,'base',f'{ds}_base_k1_temp0.1_T128_infer_seed*.json'))):
        #print(f"Processing base file: {p}")
        with open(p) as f: b.append(json.load(f)['metrics']['mean_top1_reward'])
    for p in sorted(glob.glob(os.path.join(ROOT,MODEL,f'{ds}_{MODEL}_k1_temp0.1_T128_infer_seed*.json'))):
        #print(f"Processing adapter file: {p}")
        with open(p) as f: w.append(json.load(f)['metrics']['mean_top1_reward'])
    try:
        bm=sum(b)/len(b); wm=sum(w)/len(w)
    except Exception as e:
        #print(f"Error occurred while processing {ds}: {e}")
        bm=0.0; wm=0.0
    bs=statistics.stdev(b) if len(b)>1 else 0.0
    ws=statistics.stdev(w) if len(w)>1 else 0.0
    print(f'{ds},{bm:.4f},{wm:.4f},{wm-bm:+.4f},{bs:.4f},{ws:.4f},{len(b)},{len(w)}')
    b_over.append(bm); w_over.append(wm)

print(f'benchmark_overall,{sum(b_over)/3:.4f},{sum(w_over)/3:.4f},{(sum(w_over)-sum(b_over))/3:+.4f},-,-,-,-')