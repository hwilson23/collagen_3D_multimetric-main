import os
import numpy as np
import pandas as pd
import tifffile as tiff


def load_csv(file, columns=None):
    df = pd.read_csv(file)
    if columns:
        df = df[columns]
    return df 

def reshape_CA(df):
    pivotdf = df.pivot_table(
        index=["image_name","slice"],
        columns = "hist_type",
        values = ["mean","n","std","median","z_depth"]
    )
    pivotdf.columns = [f"{stat}_{hist}" for stat,hist in pivotdf.columns]
    pivotdf = pivotdf.reset_index()
    return pivotdf

def reshape_texture(df):
    pivotdf = df.pivot(
        index=["image_name","slice"],
        columns = "texture_type",
        values = ["concentration","type","roi","texture_mean","texture_median","texture_std"]
    )
    pivotdf.columns = [f"{stat}_{tex}" for stat,tex in pivotdf.columns]
    pivotdf = pivotdf.reset_index()

    return pivotdf

def image_stats(imagepath, stackstats):
    img = tiff.imread(imagepath)
    img = np.moveaxis(img,0,-1)
    #print(f"im shape {img.shape}")
       

    #index of end filename
    idx = os.path.basename(imagepath).find("8bit")
    print(os.path.basename(imagepath)[:idx+8])
    
    for z in range(img.shape[2]):
        currentim = img[:,:,z]
        imgstats = {
            "slice" : z+1,
            "image_name": os.path.basename(imagepath)[:idx+len("8bit.ome")],
            "texture_type": os.path.basename(imagepath).split('_')[5][:-3],
            "concentration": os.path.basename(imagepath).split('_')[2],
            "type": os.path.basename(imagepath).split('_')[1],
            "roi": os.path.basename(imagepath).split('_')[3],
            "texture_mean": np.mean(currentim[currentim>0]),
            "texture_median": np.median(currentim[currentim>0]),
            "texture_std": np.std(currentim[currentim>0])
        }
        stackstats.append(imgstats)
    return stackstats

def process_img_folder(folder):
    stackstats = []

    for file in os.listdir(folder):
        if file.endswith((".tif",".tiff")):
            full = os.path.join(folder,file)
            stats = image_stats(full,stackstats)
            
            #print(results)
    return pd.DataFrame(stackstats)

def extract_stack_key(filename):
    idx = os.path.basename(filename).find("ome")
    #name of stack here will be used to collapse slices
    if idx != -1:
        return os.path.basename(filename)[:idx + len("ome")]
    return None

def twombli_slice_data(df):
    df['slice'] = df['slice'] +1 ###corrected bc spreadsheet counts from 0
    df['image_name'] = df['image_name'].apply(extract_stack_key)
    print('TWOMBLI spreadsheet processed')
    return df
##main

dfCA = load_csv("G:/FluorescentCollagen/20260427_flucol_ows3/flucol_crops/ctFIREout/results_test_masked_min30/ctfire_stats_per_slice.csv")
dfCAreorg = reshape_CA(dfCA)
dfTWOMBLI = load_csv("C:/Users/hwilson23/Desktop/TWOMBLI-master/TWOMBLI_v1/Twombli_Results_concentration_shgandflu.csv")
dfTWOMBLI = twombli_slice_data(dfTWOMBLI)
print(dfTWOMBLI)

dftexture = process_img_folder("G:/FluorescentCollagen/20260427_flucol_ows3/20260427_texturemapdata/texturemap")
dftexture= reshape_texture(dftexture)


print(dfCAreorg["image_name"].nunique(), len(dfCAreorg))
print(dfTWOMBLI["image_name"].nunique(), len(dfTWOMBLI))
print(dftexture["image_name"].nunique(), len(dftexture))
csvdf = pd.merge(dfCAreorg, dfTWOMBLI, on=["image_name","slice"], how="left")

fulldf = pd.merge(csvdf, dftexture, on=["image_name","slice"], how="left")


print(fulldf)

fulldf.to_csv("final_dataframe_byslice.csv", index=False)