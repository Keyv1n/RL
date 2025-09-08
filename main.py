# from Algorithm_SAC import SAC
# from Algorithm_DDPG import DDPG
from Algorithm_TD3 import TD3
from env1 import CarlaEnv
import time
import cv2

models_dir = f"DDPG_seed42_lr0005_si03-r1"
logdir =     f"DDPG_seed42_lr0005_si03-r1"
enviro = CarlaEnv()
print('connected')
# model = SAC(lr_rate=.0005, enviro=enviro,gamma=0.99,alpha=1,
#             adaptive_alpha=True,buffer=10000,
#             tau=0.005, batch_size=128, reward_scale=0.02)
# model = DDPG(lr_actor=.0005, lr_critic=.0002,enviro=enviro,sigma=.3,#theta=.15,
#              gamma=0.99,buffer=10000, tau=0.005, batch_size=128)
model = TD3(lr_actor=.0005, lr_critic=.0005,enviro=enviro,
            warmup=128, noise=0.2,update_actor_step=2,
            gamma=0.99,buffer=10000, tau=0.005, batch_size=128)

time_step =20000
iters = 0
while iters < 1:
    iters += 1
    print('Iteration ', iters, ' is to commence...')
    model.learn(step=time_step,n_step_update=1, tensorboard=f"tensorboard/{logdir}_train{time_step * iters}")
    print('Iteration ', iters, ' has been trained')
    model.save_models(f"{models_dir}_train{time_step * iters}")
    model.save_buffer(f"{models_dir}_train{time_step * iters}")
enviro.close()
