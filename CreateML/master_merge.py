import pandas as pd

def load_and_merge_features(cnn_path, vllm_path, clip_path, incong_path, output_path):
    # 1. LOAD DATA SOURCES
    print("Loading feature files...")
    # CNN Visual Features (Parquet - contains video_id, v_cnn)
    cnn_df = pd.read_parquet(cnn_path)
    
    # VLLM Text Embeddings (Parquet - contains video_id, T_vllm)
    vllm_emb_df = pd.read_parquet(vllm_path)
    
    # CLIP Similarity Scores (CSV - contains video_id, clip_max_similarity, clip_mean_similarity, clickbait_label)
    clip_df = pd.read_csv(clip_path)
    
    # Incongruence Scores (CSV - contains video_id, incongruence_score)
    incong_df = pd.read_csv(incong_path)

    # 2. PERFORM MULTI-JOIN
    print("Aligning videos across all modalities...")
    # We use inner joins to ensure we only train on videos that have all 4 feature types
    master_df = clip_df.merge(cnn_df, on='video_id') \
                       .merge(incong_df, on='video_id') \
                       .merge(vllm_emb_df, on='video_id')

    # 3. VECTOR EXPANSION (Flattening)
    # Most ML models need individual columns rather than a list/array in a cell.
    print("Expanding embedding vectors into feature columns...")
    
    # Expand V_cnn (e.g., ResNet 2048-dim)
    cnn_vecs = pd.DataFrame(master_df['v_cnn'].tolist(), index=master_df.index)
    cnn_vecs.columns = [f'cnn_{i}' for i in range(cnn_vecs.shape[1])]
    
    # Expand T_vllm (e.g., BERT 768-dim)
    vllm_vecs = pd.DataFrame(master_df['T_vllm'].tolist(), index=master_df.index)
    vllm_vecs.columns = [f'vllm_{i}' for i in range(vllm_vecs.shape[1])]

    # 4. FINAL CONCATENATION
    # Drop the original list columns and combine with the expanded ones
    final_df = pd.concat([
        master_df[['video_id', 'clip_max_similarity', 'clip_mean_similarity', 'incongruence_score', 'clickbait_label']],
        cnn_vecs,
        vllm_vecs
    ], axis=1)

    # 5. SAVE FOR TRAINING
    final_df.to_parquet(output_path, index=False)
    
    print(f"Success! Master file created with shape: {final_df.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    # Define file paths
    input_dir = 'features/'
    output_dir = 'datasets/'

    cnn_path = input_dir +'visual_features.parquet'
    vllm_path = input_dir +'vllm_embeddings.parquet'
    clip_path = input_dir +'clip_features.csv'
    incong_path = input_dir +'incongruence_scores.csv'
    output_path = output_dir +'master_vector_dataset.parquet'
    
    # Run the merging process
    load_and_merge_features(cnn_path, vllm_path, clip_path, incong_path, output_path)