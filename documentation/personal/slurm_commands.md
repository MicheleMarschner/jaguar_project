

## Entry
ssh hpi

## zippen
tar -czf results.tar.gz results
tar -xzf experiments.tar.gz

tar -xzf ~/Downloads/final_round_1.tar.gz

## Monitoring
- squeue --me
- tail -n 100 -f /sc/home/michele.marschner/project/jaguar_project/logs/xai_1768566.err

## Logs
less /sc/home/michele.marschner/project/jaguar_project/logs/array_<JOBID>_<TASKID>.out

## Launching

find configs/_generated/kaggle_ensemble -name "*.toml" | sort > generated_runs.txt
wc -l generated_runs.txt--experiment_name
sbatch slurm/your_array_script.slurm

## Prüfen
wc -l generated_runs.txt
cat generated_runs.txt
head -n 5 generated_runs.txt


## Cancel 
scancel [job_id]


## Git

git fetch --all --prune
git pull --no-rebase
git reset --hard origin/$(git branch --show-current)
git clean -fd
git branch --show-current
git branch -vv
git worktree list
git ls-tree -r origin/cluster --name-only | grep "experiments/round_2/splits"
git checkout origin/cluster -- experiments/round_2/splits


## Transfer
rsync -avz --progress michele.marschner@lx01.hpc.sci.hpi.de:/sc/home/michele.marschner/project/jaguar_project/experiments/round_2/kaggle_ensemble ~/Downloads/

tar -czf .tar.gz eda_xai_class_attribution

rsync -avz --progress michele.marschner@lx01.hpc.sci.hpi.de:/sc/home/michele.marschner/project/jaguar_project/checkpoints/round_1/baseline.tar.gz ~/Downloads/


python src/jaguar/experiments/experiment_runner.py \
  --base_config base/scientific_base \
  --experiment_config experiments/scientific_deduplication


rsync -avz --progress michele.marschner@lx01.hpc.sci.hpi.de:/sc/home/michele.marschner/project/jaguar_project/experiments/round_1/eda_xai_similarity/eva02_triplet_poseasy_hard_neghard/ ~/Downloads/


rsync -avz --progress ~/Downloads/kaggle_ensemble michele.marschner@lx01.hpc.sci.hpi.de:/sc/home/michele.marschner/project/jaguar_project/configs/_generated/

rsync -avz --progress experiments.tar.gz michele.marschner@hpc.sci.hpi.de:/sc/projects/sci-demelo/computer-vision/vanessaemanuela.guarino/michele.marschner/


rsync -avz michele.marschner@hpc.sci.hpi.de:/sc/projects/sci-demelo/computer-vision/vanessaemanuela.guarino/michele.marschner/round_1.tar.gz .


# 1) On your Mac: connect to HPI login node
ssh michele.marschner@hpc.sci.hpi.de

# 2) Request the specific interactive GPU node
salloc --account=sci-demelo-computer-vision -p gpu-interactive -w gx27 --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=04:00:00

# 3) Leave that terminal open
#    (this keeps your allocation alive)

# 4) On your Mac: make sure ~/.ssh/config contains this
Host hpi
    HostName hpc.sci.hpi.de
    User michele.marschner
    IdentityFile ~/.ssh/id_ed25519

Host hpi-gx27
    HostName gx27.hpc.sci.hpi.de
    User michele.marschner
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump hpi

# 5) Test from your Mac
ssh hpi
ssh hpi-gx27

# 6) In VS Code
# Cmd+Shift+P
# Remote-SSH: Connect to Host
# choose: hpi-gx27

# 7) In the VS Code terminal: verify you are really on gx27
hostname
whoami
nvidia-smi

# 8) Open your project folder in VS Code
# File -> Open Folder
# choose your repo, e.g.
cd ~/project
ls
cd ~/project/your_repo

# 9) Set git identity once on the cluster
git config --global user.name "Michele Marschner"
git config --global user.email "your-email@example.com"

# 10) Check git config
git config --show-origin --list | grep -E "user.name|user.email"

# 11) Pull the branch cleanly
git pull --rebase origin cluster

# 12) If rebase stops
git status
# solve conflicts if any, then:
git add <fixed-files>
git rebase --continue

# 13) Activate environment
micromamba activate jaguar
# or:
conda activate jaguar

# 14) Check Python / CUDA
python --version
python -c "import torch; print(torch.cuda.is_available())"

# 15) Work / debug interactively
# example:
python src/jaguar/main.py --help

# 16) When finished, end the allocation
exit
# or from login node:
scancel <JOBID>



PYTHONPATH=src python -m jaguar.experiments.experiment_runner   --base_config base/kaggle_base   --experiment_config experiments/kaggle_baseline