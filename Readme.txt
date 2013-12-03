Readme for the CM_1 Filter for the WEKA Suite.

The CM_1 filter computes the CM1 Scores for an input dataset. All values have to be nominal and no repetition of attributes allowed.

The user defines a number of folds k ( in code called m_folds, default 10) for the k-folding.
As well top and bottomranges are defined which define how many top and bottom attributes should be remain in the dataset.

For visual feedback an graph is computed using the NVD3 libary. As input a generated json file  is used. If the Graph Computed option is selected the browser is automatically open after the computation is finished. Because of the most big amount of middleattribute only each 2 is shown.

The comutation for the CM1 score is done for each fold seperately and in the end the average score for each attribute computed.


The file itself has to be included into the WEKA filestructure. In this case into weka/WEKA_src/src/main/java/weka/filters/supervised/attribute



