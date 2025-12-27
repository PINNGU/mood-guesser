"""
Dataset loader for Spotify mood classification.
Handles loading, preprocessing, and non-IID distribution across platforms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os


class MoodDataset(Dataset):
    """PyTorch Dataset class for mood classification."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize the MoodDataset.
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,) with encoded labels
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def extract_mood_labels(df: pd.DataFrame) -> pd.Series:
    """
    Extract mood labels from audio features.
    
    Rules (applied in order):
    - If valence > 0.6 AND energy > 0.5: mood = 'happy'
    - If valence < 0.4 AND energy < 0.5: mood = 'sad'
    - If energy > 0.7 AND danceability > 0.6: mood = 'energetic'
    - If energy < 0.4 AND acousticness > 0.5: mood = 'relaxed'
    - Otherwise: mood = 'neutral'
    
    Args:
        df: DataFrame with audio features
        
    Returns:
        Series with mood labels
    """
    conditions = [
        (df['valence'] > 0.6) & (df['energy'] > 0.5),
        (df['valence'] < 0.4) & (df['energy'] < 0.5),
        (df['energy'] > 0.7) & (df['danceability'] > 0.6),
        (df['energy'] < 0.4) & (df['acousticness'] > 0.5),
    ]
    
    choices = ['happy', 'sad', 'energetic', 'relaxed']
    
    mood = np.select(conditions, choices, default='neutral')
    return pd.Series(mood, index=df.index)


def normalize_tempo(df: pd.DataFrame) -> pd.Series:
    """
    Normalize tempo to 0-1 range using min-max normalization.
    
    Args:
        df: DataFrame with tempo column
        
    Returns:
        Series with normalized tempo values
    """
    tempo_min = df['tempo'].min()
    tempo_max = df['tempo'].max()
    return (df['tempo'] - tempo_min) / (tempo_max - tempo_min)


def load_and_preprocess_data(data_path: str = 'data/dataset.csv') -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess the Spotify dataset.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (preprocessed DataFrame, list of feature columns)
    """
    # Get absolute path relative to project root
    if not os.path.isabs(data_path):
        # Try to find the data file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_path = os.path.join(project_root, data_path)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Define feature columns
    feature_cols = ['valence', 'energy', 'danceability', 'acousticness', 
                    'tempo', 'loudness', 'speechiness']
    
    # Check for required columns
    required_cols = feature_cols.copy()
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values by dropping rows
    df_clean = df[feature_cols].dropna()
    print(f"Dataset size after dropping missing values: {len(df_clean)} rows")
    print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
    
    # Normalize tempo to 0-1 range
    df_clean = df_clean.copy()
    df_clean['tempo'] = normalize_tempo(df_clean)
    
    # Extract mood labels
    df_clean['mood'] = extract_mood_labels(df_clean)
    
    return df_clean, feature_cols


def create_balanced_test_set(df: pd.DataFrame, test_size: float = 0.2, 
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a balanced global test set.
    
    Args:
        df: Full DataFrame with mood labels
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train DataFrame, test DataFrame)
    """
    # Get the minimum count per mood for balanced test set
    mood_counts = df['mood'].value_counts()
    print("\nOverall mood distribution:")
    for mood, count in mood_counts.items():
        print(f"  {mood}: {count} ({100*count/len(df):.1f}%)")
    
    # Calculate samples per mood for test set (balanced)
    total_test_samples = int(len(df) * test_size)
    n_moods = df['mood'].nunique()
    samples_per_mood = total_test_samples // n_moods
    
    # Ensure we don't exceed available samples for any mood
    min_mood_count = mood_counts.min()
    samples_per_mood = min(samples_per_mood, int(min_mood_count * 0.5))
    
    print(f"\nCreating balanced test set with {samples_per_mood} samples per mood")
    
    # Sample balanced test set
    test_dfs = []
    train_dfs = []
    
    for mood in df['mood'].unique():
        mood_data = df[df['mood'] == mood]
        
        if len(mood_data) >= samples_per_mood:
            test_samples = mood_data.sample(n=samples_per_mood, random_state=random_state)
            train_samples = mood_data.drop(test_samples.index)
        else:
            # If not enough samples, take half for test
            test_samples = mood_data.sample(frac=0.5, random_state=random_state)
            train_samples = mood_data.drop(test_samples.index)
        
        test_dfs.append(test_samples)
        train_dfs.append(train_samples)
    
    test_df = pd.concat(test_dfs, ignore_index=True)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    print(f"Test set size: {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")
    print(f"Train set size: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    
    return train_df, test_df


def distribute_non_iid(train_df: pd.DataFrame, n_platforms: int = 5, 
                       random_state: int = 42) -> List[pd.DataFrame]:
    """
    Distribute training data across platforms with non-IID distribution.
    
    Platform distributions:
    - Platform 0: 40% happy, 30% energetic, 10% each for others
    - Platform 1: 40% sad, 30% relaxed, 10% each for others
    - Platform 2: 50% energetic, 20% happy, 10% each for others
    - Platform 3: Balanced 20% each
    - Platform 4: 40% relaxed, 30% neutral, 10% each for others
    
    Args:
        train_df: Training DataFrame
        n_platforms: Number of platforms
        random_state: Random seed
        
    Returns:
        List of DataFrames, one per platform
    """
    np.random.seed(random_state)
    
    # Define target distributions for each platform
    # Format: {mood: target_percentage}
    platform_distributions = [
        {'happy': 0.40, 'energetic': 0.30, 'sad': 0.10, 'relaxed': 0.10, 'neutral': 0.10},
        {'sad': 0.40, 'relaxed': 0.30, 'happy': 0.10, 'energetic': 0.10, 'neutral': 0.10},
        {'energetic': 0.50, 'happy': 0.20, 'sad': 0.10, 'relaxed': 0.10, 'neutral': 0.10},
        {'happy': 0.20, 'sad': 0.20, 'energetic': 0.20, 'relaxed': 0.20, 'neutral': 0.20},
        {'relaxed': 0.40, 'neutral': 0.30, 'happy': 0.10, 'sad': 0.10, 'energetic': 0.10},
    ]
    
    # Group data by mood
    mood_groups = {mood: group for mood, group in train_df.groupby('mood')}
    
    # Track remaining samples per mood
    remaining_indices = {mood: set(group.index) for mood, group in mood_groups.items()}
    
    # Calculate samples per platform (roughly equal total sizes)
    samples_per_platform = len(train_df) // n_platforms
    
    platform_dfs = []
    
    for platform_id, dist in enumerate(platform_distributions):
        platform_samples = []
        
        for mood, target_pct in dist.items():
            if mood not in remaining_indices:
                continue
                
            n_samples = int(samples_per_platform * target_pct)
            available = list(remaining_indices[mood])
            
            if len(available) == 0:
                continue
            
            # Sample without replacement
            n_to_sample = min(n_samples, len(available))
            sampled_indices = np.random.choice(available, size=n_to_sample, replace=False)
            
            # Remove sampled indices from remaining
            remaining_indices[mood] -= set(sampled_indices)
            
            platform_samples.extend(sampled_indices)
        
        platform_df = train_df.loc[platform_samples].copy()
        platform_dfs.append(platform_df)
    
    # Distribute any remaining samples (round-robin)
    for mood, indices in remaining_indices.items():
        indices_list = list(indices)
        for i, idx in enumerate(indices_list):
            platform_idx = i % n_platforms
            platform_dfs[platform_idx] = pd.concat([
                platform_dfs[platform_idx], 
                train_df.loc[[idx]]
            ], ignore_index=True)
    
    return platform_dfs


def prepare_platform_data(df: pd.DataFrame, feature_cols: List[str], 
                          test_df: pd.DataFrame, scaler: StandardScaler,
                          label_encoder: LabelEncoder) -> Dict:
    """
    Prepare data for a single platform.
    
    Args:
        df: Platform's training DataFrame
        feature_cols: List of feature column names
        test_df: Global test DataFrame (shared across platforms)
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        
    Returns:
        Dictionary with X_train, y_train, X_test, y_test
    """
    # Extract features and labels
    X_train = df[feature_cols].values
    y_train = label_encoder.transform(df['mood'].values)
    
    X_test = test_df[feature_cols].values
    y_test = label_encoder.transform(test_df['mood'].values)
    
    # Scale features (scaler is already fitted)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test
    }


def load_data(data_path: str = 'data/dataset.csv', 
              n_platforms: int = 5,
              test_size: float = 0.2,
              random_state: int = 42) -> Tuple[List[Dict], LabelEncoder, List[str], StandardScaler]:
    """
    Main function to load data and prepare for federated learning.
    
    Args:
        data_path: Path to the CSV file
        n_platforms: Number of platforms to distribute data to
        test_size: Fraction of data for global test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - platform_data: List of dicts with X_train, y_train, X_test, y_test
        - label_encoder: Fitted LabelEncoder
        - feature_cols: List of feature column names
        - scaler: Fitted StandardScaler
    """
    print("=" * 60)
    print("LOADING AND PREPARING DATASET")
    print("=" * 60)
    
    # Load and preprocess data
    df, feature_cols = load_and_preprocess_data(data_path)
    
    # Create balanced test set
    train_df, test_df = create_balanced_test_set(df, test_size, random_state)
    
    # Distribute training data across platforms (non-IID)
    platform_train_dfs = distribute_non_iid(train_df, n_platforms, random_state)
    
    # Fit label encoder on all moods
    label_encoder = LabelEncoder()
    all_moods = ['happy', 'sad', 'energetic', 'relaxed', 'neutral']
    label_encoder.fit(all_moods)
    
    print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Fit scaler on all training data
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    
    # Prepare data for each platform
    platform_data = []
    
    print("\n" + "=" * 60)
    print("PLATFORM DATA DISTRIBUTION")
    print("=" * 60)
    
    for i, platform_df in enumerate(platform_train_dfs):
        data = prepare_platform_data(
            platform_df, feature_cols, test_df, scaler, label_encoder
        )
        platform_data.append(data)
        
        # Print distribution statistics
        mood_counts = platform_df['mood'].value_counts()
        print(f"\nPlatform {i}:")
        print(f"  Training samples: {len(platform_df)}")
        print(f"  Test samples: {len(test_df)} (shared)")
        print(f"  Mood distribution:")
        for mood in all_moods:
            count = mood_counts.get(mood, 0)
            pct = 100 * count / len(platform_df) if len(platform_df) > 0 else 0
            print(f"    {mood}: {count} ({pct:.1f}%)")
    
    # Print test set distribution
    print("\n" + "=" * 60)
    print("GLOBAL TEST SET DISTRIBUTION")
    print("=" * 60)
    test_mood_counts = test_df['mood'].value_counts()
    for mood in all_moods:
        count = test_mood_counts.get(mood, 0)
        pct = 100 * count / len(test_df) if len(test_df) > 0 else 0
        print(f"  {mood}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("DATASET LOADING COMPLETE")
    print("=" * 60)
    print(f"Total platforms: {n_platforms}")
    print(f"Features: {feature_cols}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return platform_data, label_encoder, feature_cols, scaler


def create_dataloaders(platform_data: List[Dict], batch_size: int = 32) -> List[Dict]:
    """
    Create PyTorch DataLoaders for each platform.
    
    Args:
        platform_data: List of dicts with X_train, y_train, X_test, y_test
        batch_size: Batch size for DataLoaders
        
    Returns:
        List of dicts with train_loader and test_loader for each platform
    """
    from torch.utils.data import DataLoader
    
    dataloaders = []
    
    for data in platform_data:
        train_dataset = MoodDataset(data['X_train'], data['y_train'])
        test_dataset = MoodDataset(data['X_test'], data['y_test'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        dataloaders.append({
            'train_loader': train_loader,
            'test_loader': test_loader
        })
    
    return dataloaders


if __name__ == "__main__":
    # Test the data loading
    platform_data, label_encoder, feature_cols, scaler = load_data()
    
    # Create dataloaders
    dataloaders = create_dataloaders(platform_data, batch_size=32)
    
    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)
    
    for i, loaders in enumerate(dataloaders):
        train_loader = loaders['train_loader']
        test_loader = loaders['test_loader']
        
        # Get one batch
        X_batch, y_batch = next(iter(train_loader))
        
        print(f"\nPlatform {i}:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Batch shape: X={X_batch.shape}, y={y_batch.shape}")
        print(f"  Feature dtype: {X_batch.dtype}, Label dtype: {y_batch.dtype}")
