# multisnpnet-Cox_gpu
Multiresponse time-to-event Cox proportional hazards model - GPU

Running multisnpnet-Cox_gpu on Nero is very easy.
- First connect to Nero. Detailed instructions can be found here https://github.com/rivas-lab/wiki/wiki/nero-cluster-system. Make sure you connect to Stanford VPN.
- Request compute resources. `srun --pty -p mrivas --mem=770000 --cpus=56 --gres=gpu:8 bash`. It's okay if you only request 1 GPU.
- Run `ml snpnet_gpu/0.1.0`. This will load all dependencies of this package.
- Take a look at the file `/local-scratch/mrivas/ruilin/test/test.R`. You can simply modify the responseid in this file to analyze different set of phenotypes. You might want to copy that file to a different directory and modify the result and save directories.
- Run `Rscript test.R`.
