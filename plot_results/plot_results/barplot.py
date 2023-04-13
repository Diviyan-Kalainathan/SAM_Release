#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import pandas as pd


nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
}
mpl.rcParams.update(nice_fonts)
Algos = ["PC-Gauss",
        "PC-HSIC",
        "PC-RCOT",
        "PC-RCIT",
        "GES",
        "GIES",
        "MMHC",
        "LiNGAM",
        "CAM",
        "CCDr",
        "GENIE3",
        "SAM-lin-mse",
        "SAM-mse",
        "SAM-lin",
        "SAM"]
data = pd.read_csv('../allscores.csv')
dream = pd.read_csv('DREAM4.csv')
cyto = pd.read_csv('cyto.csv')
print(data.head())
mean = data[(data["Type"] == 'Mean') & ~(data["Algorithm"]).isin(['SAM-lin','SAM-mse'])]
data20 = mean[(mean["Size"] == 20) & (mean['Graph']!='Syntren')]
data100 = mean[(mean["Size"] == 100) & (mean['Graph']!='Syntren')]
syntren = mean[mean['Graph'] == 'Syntren']
cytoshd = cyto[cyto['Metric']=='SHD']
cytoaupr = cyto[cyto['Metric']=='AUPR']
fig, ax = plt.subplots(1, 2, squeeze=False)
sns.barplot(x='Metric', y='Score', hue='Algorithm', data=cytoaupr, palette='RdYlBu_r', ax=ax[0][0])
#plt.legend().set_draggable()
ax[0][0].legend([])
ax[0][0].set_xlabel("")
sns.barplot(x='Metric', y='Score', hue='Algorithm', data=cytoshd, palette='RdYlBu_d', ax=ax[0][1])
# sns.barplot(x='Graph', y='AUPR', hue='Algorithm', data=data100, palette='RdYlBu_r')
#plt.legend().draggable()
ax[0][1].set_ylabel("")
ax[0][1].set_xlabel("")
plt.subplots_adjust(left=0.09, bottom=0.13, right=0.81, top=0.88, wspace=None, hspace=None)
plt.savefig("barplot.png")
