# This is the implementation of FLI


***Description*** 

FLI (Fast, Learned and Interpretable) is a deep learning algorithm that employs the concept of algorithm unrolling to solve the beampattern design problem and the constant modulus constraint.


***Architecture Details***

Although the code is pretty flexible and can be configured the way you may like, we use the following setup in our expirements.


1. The network structure is visualized as follows.

![Architecture](imgs/fli.png)

where the projection and retraction operations are defined as follows.

![project_and_retract](imgs/project_retract_visualize.png)

and the exansion operation concatenates to the its inputs as follows.

![expand](imgs/expand_visualization.png)


2. The implementation details of each unrolled step (block in FLI) is defined as follows.

![details](imgs/unrolled_step.png)


3. The initialization block is defined as follows.

![initialize](imgs/initialize.png)


4. The direction evaluation module is defined as follows.

![direction](imgs/direction.png)


5. The architecture details of the expansion layer is as follows.

![expand](imgs/expand.png)
