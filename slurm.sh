#!/bin/bash
#SBATCH --job-name=ct_streamlit
#SBATCH --output=ct_streamlit_%j.out
#SBATCH --error=ct_streamlit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dept_gpu
#SBATCH --constraint="L40|A4500"

# Run the Streamlit app
streamlit run app.py --server.port=9000 --server.address=0.0.0.0 --server.maxUploadSize=1024
