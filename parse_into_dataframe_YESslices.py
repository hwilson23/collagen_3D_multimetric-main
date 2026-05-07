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

def image_stats_glcm2D(imagepath, stackstats):
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

def image_stats_glcm3D(imagepath, stackstats):
    img = tiff.imread(imagepath)
    img = np.moveaxis(img,0,-1)
    #print(f"im shape {img.shape}")
       

    #index of end filename
    idx = os.path.basename(imagepath).find(".tif")
    print(os.path.basename(imagepath)[:idx+8])
    
    for z in range(img.shape[2]):
        currentim = img[:,:,z]
        imgstats = {
            "slice" : z+1,
            "image_name": os.path.basename(imagepath)[:idx+len(".tif")],
            "texture_type": os.path.basename(imagepath).split('_')[-4], #check this
            "concentration": os.path.basename(imagepath).split('_')[2], 
            "type": os.path.basename(imagepath).split('_')[1],
            "roi": os.path.basename(imagepath).split('_')[3],
            "texture_mean": np.mean(currentim[currentim>0]),
            "texture_median": np.median(currentim[currentim>0]),
            "texture_std": np.std(currentim[currentim>0]),
            "distance": os.path.basename(imagepath).split('_')[-3],
            "neighbor": os.path.basename(imagepath).split('_')[-2],
            "bin_num": os.path.basename(imagepath).split('_')[-1][-3],
        }
        stackstats.append(imgstats)
    return stackstats

def process_img_folder(folder, is_3d):
    stackstats = []
    if is_3d ==0:
        for file in os.listdir(folder):
            if file.endswith((".tif",".tiff")):
                full = os.path.join(folder,file)
                stats = image_stats_glcm2D(full,stackstats)
    elif is_3d ==1:
        for file in os.listdir(folder):
            if file.endswith((".tif",".tiff")):
                full = os.path.joing(folder,file)
                stats = image_stats_glcm3D(full,stackstats)
            
            #print(results)
    return pd.DataFrame(stats)

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

dftexture = process_img_folder("G:/FluorescentCollagen/20260427_flucol_ows3/20260427_texturemapdata/texturemap",is_3d = 0)
dftexture= reshape_texture(dftexture)

dftexture3D = process_img_folder("G:\FluorescentCollagen\20260427_flucol_ows3\20260427_texturemapdata\texture_3d_matlab")
dftexture3D = reshape_texture(dftexture3D)


print(dfCAreorg["image_name"].nunique(), len(dfCAreorg))
print(dfTWOMBLI["image_name"].nunique(), len(dfTWOMBLI))
print(dftexture["image_name"].nunique(), len(dftexture))
csvdf = pd.merge(dfCAreorg, dfTWOMBLI, on=["image_name","slice"], how="left")

mostdf = pd.merge(csvdf, dftexture, on=["image_name","slice"], how="left")
fulldf = pd.merge(mostdf,dftexture3D, on=["image_name","slice"], how = "left")


print(fulldf)

fulldf.to_csv("final_dataframe_byslice.csv", index=False)