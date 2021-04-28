# Code

* You can fetch the latest version of <a href="https://github.com/juho-lee/set_transformer" target="_blank">Set Transformers</a> in the same folder.
* We use ESC-10, a subset of <a href="https://github.com/karolpiczak/ESC-50" target="_blank">ESC-50</a> as our data.

We have the following files for training the models and running the experiments:

1. *settransformer.py* - Training Framewise Set Transformer (FST)
2. *baseline.py* - Training Framewise Feed forward baseline (FB)
3. *settransformertemp.py* - Training Temporal Set Transformer (3ST)
4. *baselinetemp.py* - Training Temporal CNN baseline (CNN_temp)
5. *pceval.py* - Experiments on FST
6. *baseline_eval.py* - Experiments on FB
7. *pc_temp3d_eval.py* - Experiments on 3ST
8. *baseline_temp_eval.py* - Experiments on CNN_temp

*model_saves* contains our trained models, and *paper_plots* contains code to generate the plots and results we report in our paper.


