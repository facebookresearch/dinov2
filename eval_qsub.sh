#$ -l gpu=2
#$ -l cuda_memory=45G
#$ -pe smp 8
#$ -cwd
#$ -V
#$ -e /home/jluesch/output_dir/log_$JOB_ID.err
#$ -o /home/jluesch/output_dir/log_$JOB_ID.out
#$ -l h_rt=06:00:00
#$ -A kainmueller

PYTHONPATH=/fast/AG_Kainmueller/jluesch/plankton-dinov2 torchrun --standalone --nnodes=1 --nproc_per_node=2 dinov2/run/eval/knn.py --config-file dinov2/configs/train/whoi_eval.yaml --pretrained-weights /fast/AG_Kainmueller/nkoreub/plankton-dinov2_copy/model_0029999.rank_0.pth --output-dir /home/jluesch/output_dir/knn --train-dataset='HDF5Dataset:split=TRAIN:root=/fast/AG_Kainmueller/plankton/data/WHOI/preprocessed_hdf5:extra=*' --val-dataset='HDF5Dataset:split=VAL:root=/fast/AG_Kainmueller/plankton/data/WHOI/preprocessed_hdf5:extra=*'

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi
exit 0
