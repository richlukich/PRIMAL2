# PRIMAL_2: Pathfinding via Reinforcement and Imitation Multi_agent Learning - Lifelong

## Setting up Code
- `conda create --name primal python=3.7`
- `conda activate primal`
- `pip install -r requirements.txt`

## Running Code
- `python pogema_wrapper.py`

## Key Files

- `DRLMAPF_A3C_RNN.py` - The main file which contains the entire training code, worker classes and training parameters. 
- `ACNet.py` - Defines network architecture.
- `Env_Builder.py` - Defines the lower level structure of the Lifelong MAPF environment for PRIMAL2, including the world and agents class.
- `PRIMAL2Env.py` - Defines the high level environment class. 
- `Map_Generator.py` - Algorithm used to generate worlds, parameterized by world size, obstacle density and wall components.
- `PRIMAL2Observer.py` - Defines the decentralized observation of each PRIMAL2 agent.
- `PRIMALObserver.py` - Defines the decentralized observation of our previous work, PRIMAL. 
- `Obsever_Builder.py` - The high level observation class
- `pogema_wrapper.py` - Интегрирование PRIMAL2 в режиме LMAPF и среды POGEMA
-  `midel_primal2_continuous` - Веса модели
-  `render.svg` -  Сохранение результата работы программы
