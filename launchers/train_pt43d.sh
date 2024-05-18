
RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

gpu=0
lr=1e-4

nepochs=10
nepochs_decay=0

# model stuff
model='pt43d'
tf_cfg='configs/pt43d.yaml'

cat='all'
vq_model='pvqvae'
vq_cfg='configs/pvqvae_snet.yaml'
vq_ckpt='/mnt/raid/xiong/logs/2024-01-17T09-32-16-pvqvae-snet-all-LR1e-4-T0.2-clean_code/ckpt/vqvae_epoch-latest.pth'
vq_dset='snet'
vq_cat='all'

# dataset stuff
batch_size=10
max_dataset_size=10000000
dataset_mode='snet_img'
trunc_thres=0.2

display_freq=300
print_freq=25

today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

note="pt43d"
name="${DATE_WITH_TIME}-${model}-${dataset_mode}-${cat}-LR${lr}-${note}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=10
	max_dataset_size=10
    nepochs=400
    nepochs_decay=0
	update_html_freq=1
	display_freq=100
	print_freq=1
    name="DEBUG-${name}"
fi

# CUDA_VISIBLE_DEVICES=${gpuid} python train.py --name ${name} --gpu_ids ${gpuid} --lr ${lr} --batch_size ${batch_size} \
CUDA_LAUNCH_BLOCKING=${gpu} python train.py --name ${name} --gpu ${gpu} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --tf_cfg ${tf_cfg} \
                --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
                --nepochs ${nepochs} --nepochs_decay ${nepochs_decay} \
                --dataset_mode ${dataset_mode} --cat ${cat} --max_dataset_size ${max_dataset_size} --trunc_thres ${trunc_thres} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --debug ${debug}
