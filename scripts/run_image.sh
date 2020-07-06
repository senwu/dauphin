task=${1:-cifar10}
model=${2:-wide_resnet}
augment_policy=${3:-uncertainty_sampling}
batch_size=${4:-128}
num_comp=${5:-2}
augment_k=${6:-4}
augment_enlarge=${7:-1}
epoch=${8:-200}
lr=${9:-0.1}
l2=${10:-0.0005}
grad_clip=${11:-None}
lr_scheduler=${12:-cosine_annealing}
lr_scheduler_step_unit=${13:-batch}

image --task ${task} \
      --data data \
      --log_path ${task}_logs/${model}_${augment_policy}_c_${augment_k}_s_${augment_enlarge} \
      --model ${model} \
      --wide_resnet_depth 28 \
      --wide_resnet_width 10 \
      --wide_resnet_dropout 0.0 \
      --n_epochs ${epoch} \
      --batch_size ${batch_size} \
      --valid_batch_size 1000 \
      --optimizer sgd \
      --lr ${lr} \
      --l2 ${l2} \
      --grad_clip ${grad_clip} \
      --sgd_momentum 0.9 \
      --sgd_dampening 0 \
      --sgd_nesterov 1 \
      --lr_scheduler ${lr_scheduler} \
      --lr_scheduler_step_unit ${lr_scheduler_step_unit} \
      --valid_split test \
      --checkpointing 1 \
      --checkpoint_metric ${task}/${task}/test/accuracy:max \
      --augment_policy ${augment_policy} \
      --augment_k ${augment_k} \
      --augment_enlarge ${augment_enlarge} \
      --num_comp ${num_comp}
