# Feature learning
## 1. Self-attention (A. Vaswani, 2017)
link for explanation: https://www.linkedin.com/pulse/explanation-self-attention-single-head-nhut-hai-huynh/
\
The below image is an example in which attention is the preprocessinng stage of CNN layer.\
![alt text](https://github.com/nhuthai/Feature-learning/blob/master/SAN/imgs/architecture.PNG)\
The example is run by model of CNN with input: Sun Light intensity as signal. The below images illustrate the change of the similarity of the first wavelength compared to the others. At the first 10 passes, SAN shows that the first is dependent on the higher wavelength. After 100 training epoches, the most important wave channel to the first wavelength is the 40th.\
**The first 10 training epoches.**\
![alt text](https://github.com/nhuthai/Feature-learning/blob/master/SAN/imgs/head_loop10.PNG)\
**The first 50 training epoches.**\
![alt text](https://github.com/nhuthai/Feature-learning/blob/master/SAN/imgs/head_loop50.PNG)\
**After 100 training epoches.**\
![alt text](https://github.com/nhuthai/Feature-learning/blob/master/SAN/imgs/head_loop100.PNG)\
The below image illustrates the dependencies between the first and the others among **different heads**.\
![alt text](https://github.com/nhuthai/Feature-learning/blob/master/SAN/imgs/multihead.PNG)
## 2. Deep Infomax (D. Hjelm, 2019)

# References
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, **Attention is all you need.** in *NIPS 2017*, 2017.\
D. Hjelm, A. Fedorov, S. Lavoie-Marchildon, K. Grewal, P. Bachman, A. Trischler, Y. Bengio, **Learning deep representations by mutual information estimation and maximization** in *ICLR 2019*, April 2019.
