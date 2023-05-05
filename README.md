# FIBERS
**Feature Inclusion Bin Evolver for Risk Stratification (FIBERS)** is an evolutionary algorithm for automatically binning features to stratify risk in right-censored survival data. In particular it was designed for features that correspond to mismatches between donor and recipients for transplantation.

![alttext](https://github.com/UrbsLab/FIBERS/blob/main/Pictures/FIBERS_Schematic.png?raw=true)

FIBERS utilizes an evolutionary algorithm approach to optimizing bins of features based on their stratification of event risk through the following steps:

1) Random bin initialization or expert knowledge input; the bin value at an instance is the sum of the instance's values for the features included in the bin
2) Repeated evolutionary cycles consisting of:
   - Candidate bin evaluation with logrank test to evaluate for significant difference in survival curves of the low risk group (instances for which bin value = 0) and high risk group (instances for which bin value > 0).
   - Genetic operations (elitism, parent selection, crossover, and mutation) for new bin discovery and generation of the next generation of candidate bins
3) Final bin evaluation and summary of risk stratification provided by top bins

## Read More About FIBERS
The first publication detailing FIBERS (release 0.1.0) and applying it to amino acid missmatch data in predicting graft failure in kidney transplantion can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S1532046423000953?casa_token=txUcZIBcNgMAAAAA:RMGojJf4fp6fMwu38OZRNwtA-1cv8p7eSl0AW9i2gHxvfjbVij-W_Z6qkdQC6YDIj1aU5d31pA).

## Using FIBERS
Please note, we will not be further developing this FIBERS repository. Instead we point interested users to the [scikit-FIBERS repository](https://github.com/UrbsLab/scikit-FIBERS). Release 0.9.3 of scikit-FIBERS is a faithful re-implementation of this original FIBERS code other than adding a random_seed user parameter to ensure code reproducibility). Further development of the FIBERS algorithm will occur within the scikit-FIBERS repository.

### Using this implementation of FIBERS
Please see the FIBERS_Algorithm_Code.py file for full code of the FIBERS algorithm and its subfunctions. The Example.py file gives an example of how FIBERS would be run. In addition, the CoxModel_Implementation.py file gives an example of running a covariate-adjusted analysis of a top bin constructed by FIBERS in order to calculate a hazard ratio.

## Citing FIBERS
If you use FIBERS in a scientific publication, please consider citing the following paper:

Dasariraju S, Gragert L, Wager GL, McCullough K, Brown NK, Kamoun M, Urbanowicz RJ. HLA Amino Acid Mismatch-Based Risk Stratification of Kidney Allograft Failure Using a Novel Machine Learning Algorithm. Journal of Biomedical Informatics. 2023 Apr 27:104374.

BibTeX entry:
```
@article{dasariraju2023hla,
  title={HLA Amino Acid Mismatch-Based Risk Stratification of Kidney Allograft Failure Using a Novel Machine Learning Algorithm},
  author={Dasariraju, Satvik and Gragert, Loren and Wager, Grace L and McCullough, Keith and Brown, Nicholas K and Kamoun, Malek and Urbanowicz, Ryan J},
  journal={Journal of Biomedical Informatics},
  pages={104374},
  year={2023},
  publisher={Elsevier}
}
```
## Contact Us
Please email Ryan.Urbanowicz@cshs.org for any inquiries related to FIBERS.
