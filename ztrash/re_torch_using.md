# vessl ai 
## 登录
ssh -i C:\Users\DELL\.ssh\0302vessl.pem -p 31010 root@betelgeuse.cloud.vessl.ai
source ~/anaconda3/bin/activate
conda activate py310
cd QGymGPU

## 运行代码
export PYTHONPATH=/root/QGymGPU
git pull origin master
nohup python RL/PPO/train.py a2c.yaml reentrant_4 > /dev/null 2>&1 &
python RL/PPO/train.py a2c.yaml n_model_mm_10

## 下载到本地当前目录
scp -i C:\Users\DELL\.ssh\0302vessl.pem -P 31010 -r root@betelgeuse.cloud.vessl.ai:/root/QGymGPU/results .

## 查看当前进程
ps -ef


# 新版torch hpc的登录流程

https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/

## 最初配置（win）

1. 打开powershell
2. 输入New-Item -Path $HOME\.ssh\config -ItemType File -Force
3. 输入notepad $HOME\.ssh\config  （选择用记事本打开）
4. 输入
    ```yaml
    Host torch-login
        HostName login.torch.hpc.nyu.edu
        User jf5524
        StrictHostKeyChecking no
        ServerAliveInterval 60
        ForwardAgent yes
    ```
5. 保存，关闭

## 常规使用
1. 终端输入 ssh torch-login
2. 点击microsoft的链接输入PIN
3. 两步验证
4. 在microsoft页面显示成功后，再回到终端按enter
5. 连接成功

## 登录成功后
使用目录
- /home/jf5524  一直保存，内存较小，放配置，conda
- /scratch/jf5524  用于gpu计算的文件，代码，长期不用会被清

## 初始装conda


## 遇到denied问题
1. 删除本地指纹
   1. ssh-keygen -R login.torch.hpc.nyu.edu
   2. ssh-keygen -R "[login.torch.hpc.nyu.edu]:22"
2. 重新获取 ssh-keyscan -t ed25519 login.torch.hpc.nyu.edu >> $env:USERPROFILE\.ssh\known_hosts
3. 终端输入 ssh torch-login

srun --pty -t 02:00:00 -c 6 --mem=24G /bin/bash





