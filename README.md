# c-GNF : Causal-Graphical Normalizing Flows for Causal Effect Identification and Personalized Treatment/Public-Policy Analysis using Counterfactual Inference, i.e., First Law of Causal Inference.

Original PyTorch implementation of Causal-Graphical Normalizing Flows demonstrating the use of c-GNF on simulated datasets.

``Balgi, S., Peña, J. M., & Daoud, A. (2022). Personalized Public Policy Analysis in Social Sciences Using Causal-Graphical Normalizing Flows. Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 11810-11818.`` [[paper]](https://doi.org/10.1609/aaai.v36i11.21437)

The implementation of c-GNF is done by extending the offical codes for the paper: Graphical Normalizing Flows,  Antoine Wehenkel and Gilles Louppe.  (May 2020). [[arxiv]](https://arxiv.org/abs/2006.02548) [[github]](https://github.com/AWehenkel/Graphical-Normalizing-Flows)


# Dependencies
The list of dependencies can be found in requirements.txt text file and installed with the following command:
```bash
pip install -r requirements.txt
```
# Code architecture
This repository provides some code to build diverse types normalizing flow models in PyTorch. The core components are located in the **models** folder. The different flow models are described in the file **NormalizingFlow.py** and they all follow the structure of the parent **class NormalizingFlow**.
A flow step is usually designed as a combination of a **normalizer** (such as the ones described in Normalizers sub-folder) with a **conditioner** (such as the ones described in Conditioners sub-folder). Following the code hierarchy provided makes the implementation of new conditioners, normalizers or even complete flow architecture very easy.

## Refer cGNF_Wodtke_simulated_experiments_s1.ipynb jupyter notebook for further details on simulated experiments with no model misspecification and no data heterogeneity setting !!

## Refer cGNF_Wodtke_simulated_experiments_s2.ipynb jupyter notebook for further details on simulated experiments with only outcome model misspecification and data heterogeneity setting !!

## Refer cGNF_Wodtke_simulated_experiments_s3.ipynb jupyter notebook for further details on simulated experiments with both treatment and outcome model misspecification and data heterogeneity setting !!


# $\rho$-GNF : $\rho$-Graphical Normalizing Flows for sensitivity analysis using Gaussian Copula to model the degree of the non-causal association due to the unobserved confounding. [[arxiv]](https://arxiv.org/abs/2209.07111)

The extended version of c-GNF for the purpose of sensitivity analysis under unobserved confounders using a novel copula-based idea can be found as $\rho$-GNF, where $\rho$ represents the sensitivity parameter of the Gaussian copula that represents the non-causal associaition/dependence between the Gaussian noise of the $\rho$-GNF, $Z_A$ and $Z_Y$. Since the transformations of $Z_A \rightarrow A$ and $Z_Y \rightarrow Y$ are monotonic by design, the non-causal association due to unobserved confounding modeled by the copula represents the non-causal association between $A$ and $Y$ thanks to scale-invariance property of $\rho$ to monotonically increasing transformations.

The implementation of rho-GNF is done by extending the offical codes for the paper: ``Graphical Normalizing Flows,  Antoine Wehenkel and Gilles Louppe.  (May 2020)`` [[arxiv]](https://arxiv.org/abs/2006.02548) [[github]](https://github.com/AWehenkel/Graphical-Normalizing-Flows) and ``Balgi, S., Peña, J. M., & Daoud, A. (2022). Personalized Public Policy Analysis in Social Sciences Using Causal-Graphical Normalizing Flows. Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 11810-11818.`` [[paper]](https://doi.org/10.1609/aaai.v36i11.21437) [[github]](https://github.com/sobalgi/cGNF)

This implemnetation is an adaptation of the c-GNF (Causal-Graphical Normalizing Flows) for causal effect identification and estimation for sensitivity analysis to relax the unconfoundedness assumptions. 

