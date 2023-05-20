import pandas as pd
import os
import time
import pickle

FILE_NUM = 100

def num_total_s2orc():
    start_time = time.time()
    total_size = 0
    for i in range(FILE_NUM):
        print("="*20)
        current_time = int((time.time() - start_time) // 60)
        print("start reading data {} at {} m".format(i, current_time))
        metadata_df = pd.read_json('{}/metadata/metadata_{}.jsonl.gz'.format(os.getcwd(),i), lines=True, compression='infer')
        total_size += len(metadata_df)
        print(total_size)
        fields_of_study = metadata_df['mag_field_of_study']
        print(fields_of_study[:100])

    print("TOTAL S2ORC PAPER SIZE: ", total_size)

def create_acl_dataset():
  COLUMNS = ['title', 'year', 'outbound_citations', 'acl_id', 'abstract']
  start_time = time.time()
  for i in range(100): # 100
      print("="*20)
      current_time = int((time.time() - start_time) // 60)
      print("start reading data {} at {} m".format(i, current_time))
      metadata_df = pd.read_json('{}/metadata/metadata_{}.jsonl.gz'.format(os.getcwd(),i), lines=True, compression='infer')
      metadata_cs_df = metadata_df[
                          (metadata_df.mag_field_of_study.apply(lambda field: 'Computer Science' in field if field is not None else False))
                        ]
      metadata_cs_df = metadata_cs_df.set_index('paper_id')
      metadata_cs_df.index = metadata_cs_df.index.astype('str')
      df = metadata_cs_df[metadata_cs_df['acl_id'].notna()]
      data = df[COLUMNS].dropna(how='any')
      data.info()
      data.to_csv('{}/acl/acl{}.csv'.format(os.getcwd(), i))

def concat_acl_dataset():
  COLUMNS = ['title', 'year', 'outbound_citations', 'acl_id', 'abstract', 'paper_id']
  df_total = pd.DataFrame(columns=COLUMNS)
  for i in range(100):
      df = pd.read_csv('{}/acl/acl{}.csv'.format(os.getcwd(), i))
      df_total = pd.concat([df_total, df], ignore_index=True)
      
  df_total.info()

def create_cs_dataset():
  IN_COLUMNS = ['paper_id', 'title', 'year', 'abstract', 'outbound_citations', 'inbound_citations', 'has_inbound_citations']
  OUT_COLUMNS = ['paper_id', 'title', 'year', 'abstract', 'outbound_citations', 'inbound_citations']
  start_time = time.time()
  in_citations = []
  sorted_citations = []
  max_num = 100000
  OUT_DIR = os.path.join(os.getcwd(), 's2orc_cs')
  for i in range(100): # 100
      print("="*20)
      current_time = int((time.time() - start_time) // 60)
      print("start reading data {} at {} m".format(i, current_time))
      metadata_df = pd.read_json('{}/metadata/metadata_{}.jsonl.gz'.format(os.getcwd(),i), lines=True, compression='infer')
      metadata_cs_df = metadata_df[
                          (metadata_df.mag_field_of_study.apply(lambda field: 'Computer Science' in field if field is not None else False))
                        ]
      df_data = metadata_cs_df[IN_COLUMNS]
      df_data = df_data[df_data['has_inbound_citations']]
      df_data = df_data[OUT_COLUMNS].query('2006 <= year < 2021').dropna(how='any')
      sz_in_citations = [len(ic) for ic in df_data['inbound_citations']]
      in_citations = sorted_citations + [(o, s) for (o, s) in zip(df_data['paper_id'], sz_in_citations)]
      sorted_citations = sorted(in_citations, reverse=True, key=lambda x: x[1])
      sorted_citations = sorted_citations[:max_num]
      df_extracted = df_data[df_data["paper_id"].isin([elms[0] for elms in sorted_citations])]
      df_extracted.to_csv('{}/data{}.csv'.format(OUT_DIR, i), index=False)
  
  with open(os.path.join(OUT_DIR, 'top_inbound.pickle'), 'wb') as f:
      pickle.dump(sorted_citations, f)

concat_acl_dataset()
create_cs_dataset()
num_total_s2orc()