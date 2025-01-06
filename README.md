# DACE
The Source code for paper *Domain-Agnostic Co-Evolution of Generalizable Parallel Algorithm Portfolios*

This page will tell you how to config the environment for the source code and run it.

## Quick Start

### Setup Environment

#### Python Environment
```shell
conda create -n test_env -q -y python=3.8
conda activate test_env
pip install -r requirements.txt
```

#### Binary Lib compile
```shell
# You need to set the $DACE as the root path of this project
# export DACE="The Path of this project on your server"
# gcc we used is 11.2/11.4
# Install some package by apt
sudo apt install make gcc g++ libeigen3-dev libssl-dev swig git libboost-dev libasio-dev
# Download and install cmake>=3.14
cd $DACE #The root path of the project
wget https://cmake.org/files/v3.22/cmake-3.22.4.tar.gz
tar xf cmake-3.22.4.tar.gz
cd cmake-3.22.4
./bootstrap --parallel=48
make -j 255
sudo make install 

#Download and install pybind11
cd $DACE #The root path of the project
git clone https://github.com/pybind/pybind11.git
cd  pybind11
mkdir build
cd build
cmake ..
make check -j 255
sudo make install

#Download and install Boost
cd $DACE #The root path of the project
wget https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
tar xf boost_1_84_0.tar.gz
cd boost_1_84_0
./bootstrap.sh
sudo ./b2 install --prefix=/usr toolset=gcc threading=multi

# Update Dynamic Lib list 
sudo ldconfig
# Build the binary lib
# Modify the Python Environments !!! Important!!!
conda activate test_env

cd $DACE/src/cpp/com_imp/
cmake -DCMAKE_BUILD_TYPE=Release && make
```
#### Runtime Prepare
You need to run the following instructions to download the dataset for compiler arguments optimization.
```shell
ck pull repo:ck-env
ck pull repo:ck-autotuning
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
```

Create a new dictionary named **/tmp_ck**
```shell
sudo mkdir /tmp_ck
sudo chown -R ${USER} /tmp_ck
```

To avoid the disk I/O block, we recommend that create a memory disk and link it to **/tmp_ck**.
```shell
sudo mount -t tmpfs -o size=32G tmpfs /tmp_ck
```
Then copy the runtime to the **/tmp_ck** dictionary. 
```shell
cp -r ~/CK /tmp_ck
cp -r ~/CK-TOOLS /tmp_ck
cp -r ~/CK/ctuning-programs/program/* /tmp_ck
```

### Dataset
There are 3 problem classes in this repo and 3 of them are generated according to the existing datasets. The datasets of 
complementary influence maximization problem and compiler arguments optimization problem have upload into the folder `data/dataset`.

#### Complementary Influence Maximization Problem
The dataset of Facebook/Wiki/Epinions for Complementary Influence Maximization Problem is located in the folder **data/dataset/com_imp**.

#### Compiler Arguments Optimization problem
The dataset for Compiler Arguments Optimization problem is located in the folder **data/dataset/compiler_args**.

### Set the PYTHONPATH
Set the env_variable PYTHONPATH as: 
```shell
# You need to set the $DACE as the root path of this project
# export DACE="The Path of this project on your server"

export PYTHONPATH=$DACE:$DACE/src
```
While `$DACE` is the root path of this project.

### Generate Problem Instance
Run the `experiment_problem.py` in the `src` path
```shell
cd $DACE/src/experiments
python experiment_problem.py
```


### Construct PAP

#### Build Experts Models for Training Instances

```shell
cd $DACE/src/experiments
python experiment_build_surrogate.py
```

#### Generate Initial Config
```shell
cd $DACE/src/experiments
python generate_initail_cofing_set.py
```

#### Run DACE to train a PAP
```shell
cd $DACE/src/pap
python dace.py --problem_domain contamination_problem --problem_dim 30
python dace.py --problem_domain compiler_args_selection_problem --problem_dim 80 --add_recommend_config_set True
python dace.py --problem_domain com_influence_max_problem --problem_dim 80
```
#### Run CEPS to train a PAP
```shell
cd $DACE/src/pap
python ceps.py --problem_domain contamination_problem --problem_dim 30 --max_parallel_num 300
python ceps.py --problem_domain compiler_args_selection_problem --problem_dim 80 --add_recommend_config_set True --max_parallel_num 300
python ceps.py --problem_domain com_influence_max_problem --problem_dim 80 --max_parallel_num 20
```
#### Run GLOBAL to train a PAP
```shell
cd $DACE/src/pap
python global.py --problem_domain contamination_problem --problem_dim 30 --max_parallel_num 300 --distributed False --time_limit 12 --config_batch_size 200 --smac_max_try 51200
python global.py --problem_domain compiler_args_selection_problem --problem_dim 80 --max_parallel_num 300 --distributed True --time_limit 12 --config_batch_size 100 --smac_max_try 6400
python global.py --problem_domain com_influence_max_problem --problem_dim 80 --max_parallel_num 50 --distributed True --time_limit 12 --config_batch_size 75 --smac_max_try 6400
```

#### Run PARHYDRA to train a PAP
```shell
cd $DACE/src/pap
python parhydra.py --problem_domain contamination_problem --problem_dim 30 --max_parallel_num 300 --distributed False --time_limit 12 --config_batch_size 200 --smac_max_try 25600
python parhydra.py --problem_domain compiler_args_selection_problem --problem_dim 80 --max_parallel_num 300 --distributed True --time_limit 12 --config_batch_size 100 --smac_max_try 4800
python parhydra.py --problem_domain com_influence_max_problem --problem_dim 80 --max_parallel_num 50 --distributed True --time_limit 12 --config_batch_size 20 --smac_max_try 12800
```

### Test Performance for the PAPs


#### Test Performance for DACE and CEPS
Test the performance of the PAP constructed by DACE and CEPS
```shell
cd $DACE/src/experiments
python -u experiment_run_pap.py --method DACE --problem_domain contamination_problem --problem_dim 30 --max_parallel_num 200 --repeat_time 20
python -u experiment_run_pap.py --method CEPS --problem_domain contamination_problem --problem_dim 30 --max_parallel_num 200 --repeat_time 20
```

#### Test Performance for Other Baselines
```shell
cd $DACE/src/experiments
python -u experiment_run_default_pap.py --repeat_time 20
```
#### Statistic Performance Result
```shell
cd $DACE/src/experiments
python experiment_test_result.py
```
### Instance Visualization

#### Run $\text{DACE}_{NoReg}$ and Get Generated Instances
```shell
cd $DACE/src/pap
python dace_no_reg.py --problem_domain contamination_problem --problem_dim 30
python dace_no_reg.py --problem_domain compiler_args_selection_problem --problem_dim 80 --add_recommend_config_set True
python dace_no_reg.py --problem_domain com_influence_max_problem --problem_dim 80
```

#### Run Visualization
```shell
cd $DACE/src/experiments
python experiment_vis.py
```

## Distributed Computation

You can run some experiments in distributed mode, such as training PAP, evaluating performance on test set for DACE and CEPS.

In `src/pap/ceps.py`, `src/pap/global.py`, `src/pap/parhydra.py`, `src/experiments/experiment_run_pap.py`, you can run the code in distributed by add a script parameter `"--distributed true"`. And then, you need to start the evaluator script on the machines in cluster:
```shell
cd $DACE/src/distribution
python -u start_evaluator.py --pap $PAP_TYPE --max_parallel_num $PARALLEL_NUM --server_host $IP_ADDRESS
```
`$PAP_TYPE` has two candidate value `ceps` and `base`, `ceps` used in the PAP construction process of CEPS, and `base` used in the performance evaluation process. 

`$PARALLEL_NUM` is the num of the evaluation process in parallel.

`$IP_ADDRESS` is the ip address of the master machine that run the script with parameter `"--distributed true"`. 

For example, I want to Run CEPS to train a PAP on compiler args optimization problem in distributed mode.

Firstly, run the `ceps.py` on a machine with ip address `172.18.18.18`
```shell
cd $DACE/src/pap
python ceps.py --problem_domain compiler_args_selection_problem --problem_dim 80 --add_recommend_config_set True --max_parallel_num 300 --distributed true
```
Then, on each machines in your cluster:
```shell
cd $DACE/src/distribution
python -u start_evaluator.py --pap ceps --max_parallel_num 300 --server_host "172.18.18.18"
```

