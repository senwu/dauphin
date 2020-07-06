task=${1:-imdb}
model=${2:-bert-large-uncased}
augment_policy=${3:-uncertainty_sampling}
batch_size=${4:-8}
num_comp=${5:-1}
augment_k=${6:-4}
augment_enlarge=${7:-1}
epoch=${8:-5}
lr=${9:-0.00003}
l2=${10:-0}
lr_scheduler=${12:-linear}
warmup_percentage=${13:-0.1}
gradient_accumulation_steps=${14:-4}

text --task ${task} \
     --log_path ${task}_logs/${model}_${augment_policy}_c_${augment_k}_s_${augment_enlarge} \
     --model ${model} \
     --n_epochs ${epoch} \
     --batch_size ${batch_size} \
     --gradient_accumulation_steps ${gradient_accumulation_steps} \
     --optimizer bert_adam \
     --bert_adam_eps 1e-6 \
     --lr ${lr} \
     --l2 ${l2} \
     --lr_scheduler ${lr_scheduler} \
     --warmup_percentage ${warmup_percentage} \
     --valid_split test \
     --checkpointing 1 \
     --checkpoint_metric ${task}/${task}/test/accuracy:max \
     --augment_policy ${augment_policy} \
     --augment_k ${augment_k} \
     --augment_enlarge ${augment_enlarge} \
     --num_comp ${num_comp} \
     --counter_unit batch \
     --evaluation_freq 600
