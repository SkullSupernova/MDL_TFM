# CheXpert Dataset
CheXpert is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radiographs of 65,240 patients. We retrospectively collected the chest radiographic examinations from Stanford Hospital, performed between October 2002 and July 2017 in both inpatient and outpatient centers, along with their associated radiology reports. More information regarding the CheXpert dataset can be found on the Stanford Machine Learning Group's website (https://stanfordmlgroup.github.io/competitions/chexpert/) as well as the CheXpert paper (https://arxiv.org/abs/1901.07031).

## Images
The train and valid directories contain chest radiographs organized by patient.

## Validation Labels
The valid.csv file contains radiologist labels for each file in the valid directory. Three board-certified radiologists individually annotated each of the studies in the validation set, classifying each observation into one of present, uncertain likely, uncertain unlikely, and absent. Their annotations were binarized such that all present and uncertain likely cases are treated as positive and all absent and uncertain unlikely cases are treated as negative. The majority vote of these binarized annotations is used to define a strong ground truth.

## Training Labels
Training labels are automatically extracted from radiology reports for each file in the train directory. We offer 3 different sets of training labels, each using a different automated report labeler.

The train.csv file contains labels obtained by running the CheXpert labeler on reports in the CheXpert dataset. These labels include positive (1), negative (0), uncertain (-1), and unmentioned (blank) classes. More information regarding the CheXpert labeler can be found here: https://arxiv.org/pdf/1901.07031.pdf.

The train_cheXbert.csv file contains labels obtained by running the CheXbert labeler on reports in the CheXpert dataset. These labels include positive (1), negative (0), uncertain (-1), and unmentioned (blank) classes. More information regarding the CheXbert labeler can be found here: https://arxiv.org/abs/2004.09167. We use the publicly released CheXbert labeler, which can be found here: https://github.com/stanfordmlgroup/CheXbert. We recommend use of the CheXbert labels over the CheXpert labels. On certain conditions, the CheXbert labeler outperforms the CheXpert labeler on a test set of reports with radiologist report labels. Radiologist report labels are obtained by radiologists labeling the report without access to the associated image.

The train_visualCheXbert file contains labels obtained by running the VisualCheXbert labeler on reports in the CheXpert dataset. Unlike the previous two labelers, these labels are binary, with positive (1) and negative (0) labels. More information regarding the VisualCheXbert labeler can be found here: https://arxiv.org/abs/2102.11467. We use the publicly released VisualCheXbert labeler, which can be found here: https://github.com/stanfordmlgroup/VisualCheXbert. On certain conditions, the VisualCheXbert labeler outperforms the CheXpert labeler on a test set of reports with radiologist image labels. Radiologist image labels are obtained by radiologists labeling the chest X-ray image associated with a particular report.
