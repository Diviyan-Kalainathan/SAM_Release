import pandas as pd
import numpy as np

results = np.loadtxt("agsam_syntren.csv", delimiter = ",");

df_data = pd.read_csv("syntrenHop1_100_5_data.csv")

df_target = pd.read_csv("syntrenHop1_100_5_target.csv")


select_links = pd.DataFrame(columns=["Source", "Target", "Color"])

nodes = df_data.columns

for i in range(results.shape[0]):
    for j in range(results.shape[1]):

        if(results[i,j] > 0.5):

            cause = nodes[i]
            effect = nodes[j]

            color = "blue"

            for k in range(df_target.shape[0]):

                if(df_target["Cause"].loc[k] == cause and df_target["Effect"].loc[k] == effect):
                    color = "green"
                    break;

                elif(df_target["Effect"].loc[k] == cause and df_target["Cause"].loc[k] == effect):
                    color = "red"
                    break;



            select_links.loc[select_links.shape[0]+1] =[cause, effect, color]



for k in range(df_target.shape[0]):

    cause = df_target["Cause"].loc[k]
    effect = df_target["Effect"].loc[k]

    found = False;

    for i in range(select_links.shape[0]):



        if((select_links["Source"].iloc[i] == cause and select_links["Target"].iloc[i] == effect)or (select_links["Source"].iloc[i] == effect and select_links["Target"].iloc[i] == cause)):
            found = True;

            break;

    if(found == False):

        select_links.loc[select_links.shape[0] + 1] = [cause, effect, "black"]






select_links.to_csv("graph_100.csv", index_label=0)

print(select_links)