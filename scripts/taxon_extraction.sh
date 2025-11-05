#!/bin/bash

fname="taxon_extraction"
out_dir="./slurm_logs/${fname}_out"
err_dir="./slurm_logs/${fname}_err"
sbatch -J $fname -e $err_dir -o $out_dir -p gpu --ntasks=12 --mem=90G -t 3-00:00:00 --gres=gpu:1 taxon_extraction.job
