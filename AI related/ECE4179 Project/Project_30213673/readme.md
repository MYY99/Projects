# Investigating the Performance of a Capsule Network in Digit Classification Task
Hinton et al. [1] and Sabour et al. [2] presented capsule network with dynamic routing, an approach closer to replicating the human vision, to resolve the fundamental limitations of CNNs: translational equivariance [3], no build-in understanding of 3D space, and Picasso problem [4]. 

This project investigates the performance of a Capsule Network (CapsNet) in digit classification task.

![image](https://github.com/MYY99/Projects/assets/133868293/5ef92f12-249c-43b5-9a5f-e9d82d91cda8)

## Results
![image](https://github.com/MYY99/Projects/assets/133868293/ca851a27-0da2-47ac-8ab5-aa3cb9e13fbc)
All networks have been trained solely on MNIST or padded MNIST. However, CapsNet is capable of performing multi-label image classification despite being trained exclusively on single-label images.

## References
[1] G. E. Hinton, A. Krizhevsky, and S. D. Wang, “Transforming Auto-Encoders,” in  International conference on artificial neural network, 2011, pp. 44-51, doi:  10.1007/978-3-642-21735-7_6. 
[2] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in Advances in neural information processing systems 30, 2017. 
[3] L. Alzubaidi et al., “Review of deep learning: concepts, CNN architectures, challenges, applications, future directions,” Journal of Big 
Data, vol. 8, no. 53, Mar. 2021. 
[4] J. D. Kelleher, “The Future of Deep Learning,” in Deep Learning. Cambridge, U.S: MIT Press, 2019.

