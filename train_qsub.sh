#$ -l gpu=10
#$ -l cuda_memory=45G
#$ -pe smp 40
#$ -cwd
#$ -V
#$ -e /home/jluesch/output_dir/log_$JOB_ID.err
#$ -o /home/jluesch/output_dir/log_$JOB_ID.out
#$ -l h_rt=120:00:00
#$ -A kainmueller

PYTHONPATH=/fast/AG_Kainmueller/jluesch/plankton-dinov2 torchrun --standalone --nnodes=1 --nproc_per_node=10 dinov2/run/train/train.py --config-file dinov2/configs/train/whoi.yaml --output-dir /fast/AG_Kainmueller/jluesch/output_dir

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi
exit 0
