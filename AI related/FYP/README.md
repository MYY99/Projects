# Differentiable Neural Architecture Search through a Hypernetwork For Image Recognition
Neural Architecture Search (NAS) automates the search for optimal neural network architectures, a process traditionally requiring human expertise and trial-and-error. NAS employs machine learning algorithms to explore a predefined search space, evaluating and comparing architectures based on task performance. It utilizes strategies like reinforcement learning, evolutionary algorithms, or gradient-based optimization to efficiently identify architectures with superior performance. By automating architecture design, NAS accelerates the development of advanced neural networks, propelling AI research and applications.

DARTS (Differentiable Architecture Search) is a gradient-based approach to (NAS). It uses a continuous relaxation of the architecture space, allowing for efficient optimization through backpropagation. By introducing differentiable variables and leveraging the softmax relaxation, DARTS enables joint optimization of architecture and model weights. This approach has proven effective in discovering competitive neural architectures while reducing computational costs. 

DARTS uses unrolling method to approach the bilevel optimization search problem to approximate hypergradient. This project proposes developing a DARTS variant by replacing the unrolling method with a hypernetwork to eliminate the approximation loss and further improve its performance in searching architectures.

![image](https://github.com/MYY99/Projects/assets/133868293/acc277d1-dca8-4ede-a386-bbad35c4c581)


