import pandas as pd
import pdb
import numpy as np
import argparse
import json 

import os
import fnmatch
import shutil

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=("Preprocessing give scalar dense columns into lists"))
  args = parser.parse_args()
  #TODO read from args?
  train_folder_in =  "/home/junzhang/workspace/dataset/vec_dense_criteo/criteo_data/train"
  train_folder_out = "/home/junzhang/workspace/dataset/vec_dense_criteo/workspace/I1I2/train"
  train_file_names = []
  train_metadata = os.path.join( train_folder_in, "_metadata.json" )
  train_metadata_out = os.path.join( train_folder_out, "_metadata.json" )
  combined_list = [["I1","I2","I3"],["I4","I5"],["I9","I10","I11","I12"]]
  slot_size_arr = np.zeros(26,dtype=np.int64)

  for file in os.listdir(train_folder_in):
      if fnmatch.fnmatch(file, '*.parquet'):
          train_file_names.append(  os.path.join( train_folder_in, file ))

  train_dst_file = pd.DataFrame({"file_name":[],"num_rows":[]})
  for file_id , src_path in enumerate(train_file_names):
    dst_path = os.path.join(train_folder_out,str(file_id)+".parquet")

    df_src = pd.read_parquet(src_path)
    # TODO Truncated/?
    df_src100k = df_src.loc[0:99999]
    for combined in combined_list:
      # 1.to list
      for cc in combined:
        df_src100k[cc] = df_src100k[cc].apply(lambda x: [x])
      # 2.concat
      for cc in range(1,len(combined)):
        df_src100k[combined[0]] = df_src100k[combined[0]]+df_src100k[combined[cc]]
      # 3.to float32
      df_src100k[combined[0]]=df_src100k[combined[0]].apply(lambda x: np.array(x,dtype=np.float32))
      # 4.renaming
      df_src100k=df_src100k.rename(columns = {combined[0]: "_".join(combined)})
      # TODO Drop the original or not?
      df_src100k = df_src100k.drop(combined,axis = 1,errors='ignore')
    df_src100k.to_parquet(dst_path)
    train_dst_file = train_dst_file.append({'file_name':str(file_id)+".parquet",'num_rows':100000},ignore_index=True)
    
    cat_col_name = df_src100k.columns[df_src100k.columns.str.startswith("C")]
    slot_size_cur_df = df_src100k[cat_col_name].max()
    slot_size_arr = np.maximum(slot_size_cur_df.to_numpy().astype('int64'),slot_size_arr)
  # use last dataframe
  new_column_id = pd.DataFrame({'col_name':df_src100k.columns,'index':np.arange(len(df_src100k.columns))})
  conts_id_df = new_column_id[df_src100k.columns.str.startswith("I")]
  cats_id_df = new_column_id[df_src100k.columns.str.startswith("C")]
  # insert slot_size_array
  cats_id_df.insert(cats_id_df.shape[1],"slot_size",slot_size_arr)

  label_id_df = new_column_id[df_src100k.columns.str.startswith("label")]

  col_idx = conts_id_df.to_dict('records')
  cat_idx = cats_id_df.to_dict('records')
  label_idx = label_id_df.to_dict('records')
  file_stats = train_dst_file.to_dict('records')

  with open(train_metadata,'r+') as f:
      meta_ = json.load(f)
      conts_ = meta_['conts']
      meta_['conts'] = col_idx
      meta_['cats'] = cat_idx
      meta_['labels'] = label_idx
      meta_['file_stats'] = file_stats
      
  print(meta_)
  # dump json to output folder
  with open(train_metadata_out,'w') as f:
      json.dump(meta_,f)


