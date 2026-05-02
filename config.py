# config.py
# Configuration for Mobile-Optimized Camouflage Object Detection

class Config:
    # Model - Mobile Optimized
    backbone = 'mobilenet_v3_small'
    input_size = 256  # Mobile-friendly size
    channels = 3

    # Training
    batch_size = 8
    learning_rate = 1e-3
    epochs = 50
    weight_decay = 1e-4

    # Dataset paths - Adjusted for your structure
    dataset_paths = {
        'camo': {
            'train_img': r"E:/object dataset/camo dataset/CAMO-V.1.0-CVIU2019/Test/Train",
            'train_mask': r"E:/object dataset/camo dataset/CAMO-V.1.0-CVIU2019/GT",
            'test_img': r"E:/object dataset/camo dataset/CAMO-V.1.0-CVIU2019/Test/Image",
            'test_mask': r"E:/object dataset/camo dataset/CAMO-V.1.0-CVIU2019/GT"
        },
        'cod10k': {
            'train_img': r"E:\object dataset\cod10k dataset\COD10K-v2\Train\Image",
            'train_mask': r"E:\object dataset\cod10k dataset\COD10K-v2\Train\GT_object",
            'test_img': r"E:\object dataset\cod10k dataset\COD10K-v2\Test\Image",
            'test_mask': r"E:\object dataset\cod10k dataset\COD10K-v2\Test\GT_object"
        },
        'nc4k': {
            'test_img': r"E:/object dataset/Nc4k dataset/Image",
            'test_mask': r"E:/object dataset/Nc4k dataset/GT_object"
        }
    }

    # Which datasets to use for training and testing
    train_datasets = ['camo', 'cod10k']  # Train on CAMO and COD10K
    test_datasets = ['nc4k']  # Test on NC4K

    # Mobile constraints
    target_params = 2e6  # 2M parameters
    target_size = 10  # MB
    target_inference_time = 0.1  # 100ms on mobile

config = Config()
