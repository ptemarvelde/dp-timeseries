#!/usr/bin/env bash
### modify the 'meta_start' (and optionally 'njobs') and run this script multiple times to pretrain the discriminators in parallel

meta_start=0 # the discriminator start index for the current process (need to be modified for each process)
ndis=100   # total number of discriminators
dis_per_job=100 # number of discriminators to be trained for each process
njobs=1     # ndis // dis_per_job

(echo "import sys" ; echo "print(sys.executable)") | python
for i in $(seq 0 $njobs); do
  start=$((i * dis_per_job + meta_start))
  end=$((start + dis_per_job - 1))
  vals=$(seq $start $end)
  echo $vals
  python pretrain.py -data 'ptb' -ids $vals --pretrain -gen 'ResNet' -name 'ptb_pretrain' -noise 0. -ndis $ndis
done
deactivate