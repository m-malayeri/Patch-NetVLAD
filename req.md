### Before Run
Rename the Tokyo247 folder
original: database_gsv_vga
expected: 247_Tokyo_GSV_Perspective

Replace ./Mapillary_Street_Level_Sequences with Mapillary in:
    Patch-NetVLAD/patchnetvlad/dataset_imagenames/mapillarycph_imageNames_index.txt
    and 
    Patch-NetVLAD/patchnetvlad/dataset_imagenames/mapillarycph_imageNames_query.txt

### DON'T FORGET: ##############################################################################
- Pretrained models are hardcoded in (.ini) files.
#### #########################################################################
### Training Tokyo247
python train.py \
    --config_path patchnetvlad/configs/train.ini \
    --cache_path=/tmp \
    --save_path=/home/jovyan/data/VPR/PatchNetVLAD_out/results/tokyo247 \
    --dataset_root_dir=/home/jovyan/data/VPR/Tokyo247
    --nEpochs=1
    --threads=32

### Training Pitts30k
python train.py \
    --config_path patchnetvlad/configs/train.ini \
    --cache_path=/tmp \
    --save_path=/home/jovyan/data/VPR/PatchNetVLAD_out/results/pitts30k \
    --dataset_root_dir=/home/jovyan/data/VPR/Pittsburgh250k

### Training Mapillary
python train.py \
    --config_path patchnetvlad/configs/train.ini \
    --cache_path=/tmp \
    --save_path=/home/jovyan/data/VPR/PatchNetVLAD_out/results/mapillary \
    --dataset_root_dir=/home/jovyan/data/VPR/Mapillary \
    --nEpochs=1 \
    --threads=24
    
### Training LW
python train.py \
    --config_path patchnetvlad/configs/train.ini \
    --cache_path=/tmp \
    --save_path=/home/jovyan/data/VPR/PatchNetVLAD_out/results/LW \
    --dataset_root_dir=/home/jovyan/data/VPR/LW \
    --nEpochs=1 \
    --threads=16

#### #########################################################################
### Feature Extraction (INDEX) for Tokyo247 
python feature_extract.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_file_path=tokyo247_imageNames_index.txt \
    --dataset_root_dir=/home/jovyan/data/VPR/Tokyo247 \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/tk247_index

### Feature Extraction (QUERY) for Tokyo247 
python feature_extract.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_file_path=tokyo247_imageNames_query.txt \ 
    --dataset_root_dir=/home/jovyan/data/VPR/Tokyo247 \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/tk247_query
    
#### #########################################################################
### Feature Extraction (INDEX) for Pitts30K
python feature_extract.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_file_path=pitts30k_imageNames_index.txt \
    --dataset_root_dir=/home/jovyan/data/VPR \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/pitts30k_index

### Feature Extraction (QUERY) for Pitts30K
python feature_extract.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_file_path=pitts30k_imageNames_query.txt \
    --dataset_root_dir=/home/jovyan/data/VPR \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/pitts30k_query

#### #########################################################################
### Feature Extraction (INDEX) for Mapillary
python feature_extract.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_file_path=mapillarycph_imageNames_index.txt \
    --dataset_root_dir=/home/jovyan/data/VPR \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/mapillary_index

### Feature Extraction (QUERY) for Mapillary
python feature_extract.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_file_path=mapillarycph_imageNames_query.txt \
    --dataset_root_dir=/home/jovyan/data/VPR \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/mapillary_query

#### #########################################################################
### Feature Extraction (INDEX) for LW
python feature_extract.py \
    --config_path patchnetvlad/configs/speed.ini \
    --dataset_file_path=lw_imageNames_index.txt \
    --dataset_root_dir=/home/jovyan/data/VPR/LW \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/lw_index

### Feature Extraction (QUERY) for LW
python feature_extract.py \
    --config_path patchnetvlad/configs/speed.ini \
    --dataset_file_path=lw_imageNames_query.txt \
    --dataset_root_dir=/home/jovyan/data/VPR/LW \
    --output_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/lw_query
    
#### #########################################################################
### Feature Matching for Tokyo247
python feature_match.py \
    --config_path patchnetvlad/configs/performance.ini \
    --dataset_root_dir=/home/jovyan/data/VPR/Tokyo247 \
    --query_file_path=tokyo247_imageNames_query.txt \
    --index_file_path=tokyo247_imageNames_index.txt \
    --query_input_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/tk247_query \
    --index_input_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/tk247_index \
    --ground_truth_path patchnetvlad/dataset_gt_files/tokyo247.npz \
    --result_save_folder /home/jovyan/data/VPR/PatchNetVLAD_out/results/tk247

### Feature Matching for Pitts30K
python feature_match.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_root_dir=/home/jovyan/data/VPR \
    --query_file_path=pitts30k_imageNames_query.txt \
    --index_file_path=pitts30k_imageNames_index.txt \
    --query_input_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/pitts30k_query \
    --index_input_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/pitts30k_index \
    --ground_truth_path patchnetvlad/dataset_gt_files/pitts30k_test.npz \
    --result_save_folder /home/jovyan/data/VPR/PatchNetVLAD_out/results/pitts30k

### Feature Matching for Mapillary
### Needs extra work
python feature_match.py \
    --config_path patchnetvlad/configs/storage.ini \
    --dataset_root_dir=/home/jovyan/data/VPR \
    --query_file_path=mapillarycph_imageNames_index.txt \
    --index_file_path=mapillarycph_imageNames_query.txt \
    --query_input_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/mapillary_query \
    --index_input_features_dir /home/jovyan/data/VPR/PatchNetVLAD_out/output_features/mapillary_index \
    --ground_truth_path patchnetvlad/dataset_gt_files/**.npz \
    --result_save_folder /home/jovyan/data/VPR/PatchNetVLAD_out/results/pitts30k
    
### Add PCA
python add_pca.py \
    --config_path patchnetvlad/configs/train.ini \
    --resume_path=full/path/with/extension/to/your/saved/checkpoint \
    --dataset_root_dir=/path/to/your/mapillary/dataset
