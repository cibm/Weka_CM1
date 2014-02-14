Readme for the CM_1 Filter for the WEKA Suite.

The CM_1 filter computes the CM1 Scores for an input dataset. All values have to be nominal and no repetition of
attributes allowed.

The user defines a number of folds k ( in code called m_folds, default 10) for the k-folding.
As well top and bottomranges are defined which define how many top and bottom attributes should be remain in the
dataset.

For visual feedback an graph is computed using the NVD3 libary. As input a generated json file is used. If the Graph
Computed option is selected the browser is automatically open after the computation is finished. Because of the most big
amount of middleattribute only each 2 is shown.

The comutation for the CM1 score is done for each fold seperately and in the end the average score for each attribute computed.


===============================
COMPILATION
===============================

To compile the CM1 filter plug-in, you need to follow these steps:

javac -Xlint:unchecked -classpath <path_to_weka_jarfile>/weka.jar -d weka/filters/weka/filters/supervised/attribute/ CM_1.java

Ignore the warnings produced by the compiler.

Once the filter has been compiled, there must be generated a jar file with the plug-in
to make it visible to Weka. The jar file is generated using the following command:

jar -cvf cm1.jar weka/filters/supervised/attribute/*


==============================
RUNNING CM1 WITH WEKA
==============================

After the CM1 jar file has been generated, it will be automatically detected by Weka's GenericObjectEditor
and placed together with other filters under filters/supervised/attribute on all graphical and command line interfaces.
The dynamic detection of classes is by default enabled in version 3.7+. Check if it is enabled in your version on Weka's
website.
A requirement of the GenericObjectEditor system is that CM1 jar file must be added to the Java Virtual Machine
classpath either by the -classpath flag or setting the environment variable CLASSPATH.
To run Weka setting the classpath with the flag, use the following commands:

java -classpath <path_to_cm1.jar>/cm1.jar:<path_to_weka.jar>/weka.jar weka.gui.GUIChooser
