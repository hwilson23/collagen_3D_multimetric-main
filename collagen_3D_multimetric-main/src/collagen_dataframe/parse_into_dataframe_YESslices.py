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
    # List of all possible value columns
    possible_values = ["concentration",
                       "type",
                       "roi",
                       "texture_mean",
                       "texture_median",
                       "texture_std", 
                       "texture3d_mean", 
                       "texture3d_median",
                       "texture3d_std", 
                       "distance3d", 
                       "neighbor3d", 
                       "bin_num3d"]
    
    # Filter to only include columns that exist in the dataframe
    available_values = [col for col in possible_values if col in df.columns]
    
    if not available_values:
        # If no values are available, just return the dataframe as-is
        return df
    
    pivotdf = df.pivot(
        index=["image_name","slice"],
        columns = "texture_type",
        values = available_values
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
    #print(os.path.basename(imagepath)[:idx+8])
    
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
    #print(os.path.basename(imagepath)[:idx+8])
    
    for z in range(img.shape[2]):
        currentim = img[:,:,z]
        imgstats = {
            "slice" : z+1,
            "image_name": os.path.basename(imagepath)[:idx+len(".tif")],
            "texture_type": os.path.basename(imagepath).split('_')[-4], 
            "concentration": os.path.basename(imagepath).split('_')[2], 
            "type": os.path.basename(imagepath).split('_')[1],
            "roi": os.path.basename(imagepath).split('_')[3],
            "texture3d_mean": np.mean(currentim[currentim>0]),
            "texture3d_median": np.median(currentim[currentim>0]),
            "texture3d_std": np.std(currentim[currentim>0]),
            "distance3d": os.path.basename(imagepath).split('_')[-3],
            "neighbor3d": os.path.basename(imagepath).split('_')[-2],
            "bin_num3d": os.path.basename(imagepath).split('_')[-1][-3],
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
                full = os.path.join(folder,file)
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
#print(dfTWOMBLI)

dftexture = process_img_folder("G:/FluorescentCollagen/20260427_flucol_ows3/20260427_texturemapdata/texturemap",is_3d = 0)
dftexture= reshape_texture(dftexture)

dftexture3D = process_img_folder("G:/FluorescentCollagen/20260427_flucol_ows3/20260427_texturemapdata/texture_3d_matlab_endedbeforedone",is_3d = 1)
dftexture3D = reshape_texture(dftexture3D)


print(dfCAreorg["image_name"].nunique(), len(dfCAreorg))
print(dfTWOMBLI["image_name"].nunique(), len(dfTWOMBLI))
print(dftexture["image_name"].nunique(), len(dftexture))
print(dftexture3D["image_name"].nunique(), len(dftexture3D))

csvdf = pd.merge(dfCAreorg, dfTWOMBLI, on=["image_name","slice"], how="left")

mostdf = pd.merge(csvdf, dftexture, on=["image_name","slice"], how="left")
fulldf = pd.merge(mostdf,dftexture3D, on=["image_name","slice"], how = "left")


print(fulldf.head())
print(fulldf.columns.values)

# Split into FLU and SHG dataframes if type column exists
if 'type' in fulldf.columns:
    fulldf_flu = fulldf[fulldf['type'].str.lower() == 'flu'].copy()
    fulldf_shg = fulldf[fulldf['type'].str.lower() == 'shg'].copy()
    
    #remove the _flu_ _shg_ part of the filename
    fulldf_flu['short_image_name'] = fulldf_flu['image_name'].str.replace('_flu_', '_', regex=False)
    fulldf_shg['short_image_name'] = fulldf_shg['image_name'].str.replace('_shg_', '_', regex=False)
    print(f"FLU dataframe shape: {fulldf_flu.shape}")
    print(f"SHG dataframe shape: {fulldf_shg.shape}")
    
    # Save separate files
    fulldf_flu.to_csv("final_dataframe_byslice_FLU.csv", index=False)
    fulldf_shg.to_csv("final_dataframe_byslice_SHG.csv", index=False)
    
    print("Saved FLU and SHG dataframes separately")

# Also save the combined dataframe
fulldf.to_csv("final_dataframe_byslice.csv", index=False)