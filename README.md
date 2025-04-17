# 2048-Torch

This is an attempt to teach a Reinforcement Learning model using a Deep Q Network to play 2048.  I've written all of the code from scratch, with significant help from a couple of sites:
1. [Torch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
2. [Towards Data Science/Medium post on Deep Q Networks](https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b/).

As of now, things seem to be 'working' in the sense that I am getting the right objects in the right places and all that, but it won't really learn anything and I'm not sure why yet.  It has typically been learning enough to be almost, but not quite, as good as picking moves entirely at random.  Which is not good. 🤷

If anyone is trying to work with this and needs the point of comparison, a 4x4 2048 board with all random moves should average around 1080 points.  A 3x3 board should average around 132 points.  But, on a 4x4 board, I can get 20k+ points and on a 3x3 baord I can consistently get 500+ points, and occasionally over 1000.  And I'm not that good at it.  My goal is to get the AI to be able to beat me.

More later as things improve...
