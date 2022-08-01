# Fetal-RL-updated_priv


# Data preparation

order of running scripts to prepare data:

1. data_preparation.py
2. best_frame_selection.py
3. ssmi.py
4. distances_and_cosine_similarity.py
5. train_test_split.py

Output of train_test_split.py is 'fetal_extracted_map_train_scr.csv', and 'fetal_extracted_map_train_scr.csv'.

To run swin-transformer first refer to (get_stared.md), create folder 'fetal'. Inside of folder 'fetal' should be 'fetal_extracted' that contains images (no subfolders). It should also containt 'fetal_extracted_map_train_scr.csv', and 'fetal_extracted_map_train_scr.csv'.

To run the train process type in terminal: 

```bash
srun python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py --cfg configs/swin/fetal_base.yaml --data-path /data/kpusteln/fetal/fetal_extracted/ --batch-size 128
```
