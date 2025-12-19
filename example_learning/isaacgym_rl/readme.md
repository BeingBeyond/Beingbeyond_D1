# IsaacGym仿真环境强化学习训练BeingbeyondD1抓取

## 配环境
```
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# 创建另一个conda环境，用python3.8
conda create -n isaac python=3.8.19
conda activate isaac

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 # pytorch版本>=2.0.0即可

下载IsaacGym preview 4 (https://developer.nvidia.com/isaac-gym/download)并解压
cd IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .

下载IsaacGymEnvs (https://github.com/isaac-sim/IsaacGymEnvs/)
cd IsaacGymEnvs
pip install -e .

pip install Cython==0.29.37 gym==0.26.2 h5py==3.11.0 gymnasium==0.29.1 hydra-core==1.3.2  imageio==2.34.1 joblib==1.4.2 loguru==0.7.2 matplotlib==3.5.1 networkx==3.1 numpy==1.23.5 nlopt==2.7.1 omegaconf==2.3.0 open3d==0.18.0 opencv-contrib-python==4.10.0.84  opencv-python==4.10.0.84 pandas==2.0.3 patchelf==0.17.2.1 psutil==5.9.8 ray==2.10.0 pytransform3d==3.5.0 scipy==1.10.0 transform3d==0.0.4 transforms3d==0.4.1 trimesh==3.23.5 tyro==0.8.4 wandb==0.12.21 tqdm==4.66.4 yapf==0.40.2 sorcery==0.2.2 pynvml==11.5.0 ipdb==0.13.13
```


## 训练样例

- 跑少量环境，进行可视化：
```
cd isaacgym_rl
python run_rl_grasp.py num_envs=10 +debug=check_joint task.env.enableDebugVis=True
```

- 训练抓取
```
python run_rl_grasp.py num_envs=8000 # 并行8000个环境，用PPO训练抓取
```

- 测试训好的模型
```
找到保存的模型路径runs_ppo/grasp_xxx/model_xxx.pt
python run_rl_grasp.py num_envs=1000 test=True checkpoint=runs_ppo/grasp_xxx/model_xxx.pt
```