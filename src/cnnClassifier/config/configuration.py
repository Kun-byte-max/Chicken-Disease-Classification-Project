from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
print("[DEBUG] Root Dir is:", ROOT_DIR)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
                        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        # Go up 3 levels: config/configuration.py → cnnClassifier/config → src → project root
        self.root_dir = Path(__file__).resolve().parents[3]

        self.config.artifacts_root = self.root_dir / self.config.artifacts_root
        create_directories([self.config.artifacts_root])

        print("[DEBUG] Root Dir is:", self.root_dir)


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        root_dir = self.root_dir / config.root_dir
        local_data_file = self.root_dir / config.local_data_file
        unzip_dir = self.root_dir / config.unzip_dir

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        root_dir = self.root_dir / config.root_dir
        base_model_path = self.root_dir / config.base_model_path
        updated_base_model_path = self.root_dir / config.updated_base_model_path

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
        

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        root_dir = self.root_dir / config.root_dir
        tensorboard_root_log_dir = self.root_dir / config.tensorboard_root_log_dir
        checkpoint_model_filepath = self.root_dir / config.checkpoint_model_filepath

        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)

        )

        return prepare_callback_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        root_dir = self.root_dir / training.root_dir
        trained_model_path = self.root_dir / training.trained_model_path
        updated_base_model_path = self.root_dir / prepare_base_model.updated_base_model_path

        # Training data absolute path
        training_data = self.root_dir / self.config.data_ingestion.unzip_dir / "Chicken-fecal-images"

        create_directories([root_dir])

        return TrainingConfig(
            root_dir=root_dir,
            trained_model_path=trained_model_path,
            updated_base_model_path=updated_base_model_path,
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/Chicken-fecal-images",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE        
        )
        return eval_config


    
    
    