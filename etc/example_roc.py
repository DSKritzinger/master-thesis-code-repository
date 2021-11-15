from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# set golden ratio values
gr = (np.sqrt(5)-1)/2
fig_width = 4
# set set_style


def set_style():
    sns.set(context="paper", font='serif', style="white", rc={"xtick.bottom": True,
                                                              "xtick.labelsize": "x-small",
                                                              "ytick.left": True,
                                                              "ytick.labelsize": "x-small",
                                                              "legend.fontsize": "x-small",
                                                              "ytick.major.size": 2,
                                                              "xtick.major.size": 2})


actual1 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
           0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
predictions1 = [0, 0, 1, 1, 0, 0.3, 1, 0, 0, .9, 1, 1, 0, 0.1, 0, 0, 1, 0, 0, 0.3, 1, 0, 0, .9, 1, 1,
                1, 0.1, 1, 0, 0, 1, 0, 0.3, 1, 0, 0, .9, 1, 1, 0, 0.1, 0, 0, 1, 0, 0, 0.3, 1, 0, 0, .9, 1, 1, 1, 0.1]
actual2 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
           0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
predictions2 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
                0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]

false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(actual1, predictions1)
roc_auc1 = auc(false_positive_rate1, true_positive_rate1)

false_positive_rate2, true_positive_rate2, thresholds2 = roc_curve(actual2, predictions2)
roc_auc2 = auc(false_positive_rate2, true_positive_rate2)
set_style()
plt.figure(figsize=(fig_width, fig_width*gr))
plt.plot(false_positive_rate1, true_positive_rate1, 'blue',
         label=' Skillful (AUC = %0.2f)' % roc_auc1)
plt.plot(false_positive_rate2, true_positive_rate2, 'red',
         label=' Perfect Skill (AUC = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--', label=' No skill (AUC = %0.2f)' % 0.5)
plt.legend(loc='lower right')
plt.xticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
plt.xlim([0, 1])
plt.ylim([0, 1.1])
sns.despine()
plt.ylabel("Sensitivity")
plt.xlabel("1-Specificity")
plt.savefig("C:/Users/Daniel/Google Drive/Postgraduate/Thesis/Thesis Figures/example_roc.png",
            bbox_inches="tight", dpi=1000)

# adjust tick label size
