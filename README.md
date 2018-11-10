
# ppo_pytorch :

A simple implementation actor critic style [Clipped Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) in pytorch that runs in [gym](https://gym.openai.com/) envs. This library also contains some weird additions, shortcuts and experimental stuff like  truncated distributions, fixed std on the policy network (suprisingly works quite well) and full episode rollouts so it may not always marry up precisely with openai baselines. 


![gif](gifs/hardcore_runner2.gif "gif")
*This guy has been training for 50409 16 episode rollouts in the episode shown he scored 284.6*


## Installation:
   
   - ideally make yourself a virtualenv so i don't fuck up your torch install or whatever and then do:

   ```bash
    git clone https://github.com/leaprovenzano/ppo_pytorch.git
    pip install -e ppo_pytorch
   ```


## Super basic example : 

COMING SOON ... a notebook or something, soz!


