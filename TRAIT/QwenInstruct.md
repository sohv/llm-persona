
## Install reqs:

```
pip install -r requirements.txt
```

## Inference: 
model_name_short could be anything, just decides the name of the output file.

```
cd src/
​python run.py --model_name Qwen/Qwen2.5-0.5B-Instruct --model_name_short qwen2.5-0.5B --inference_type chat --prompt_type 1
```

Include the tags --fine_tuned and --adapter_model followed by the adapter to use eg. Qwen2.5-0.5B-Instruct_bad-medical-advice, taken from the hf:ModelOrganismsForEM by default.

```
​python run.py --model_name Qwen/Qwen2.5-0.5B-Instruct --fine_tuned --adapter_model Qwen2.5-0.5B-Instruct_risky-financial-advice --model_name_short qwen2.5-0.5B_med --inference_type chat --prompt_type 1
```

## Analysis:
make sure --model_name is same as --model_short_name you used for inference.

```
python analysis.py --model_name qwen2.5-0.5B
```

## Plot Stat graph:
Pass the output of analysis here.

```
cd results/
python plot.py --num n <file1.txt> ... <filen.txt>
```
