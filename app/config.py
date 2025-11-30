from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SAM2_MODEL_TYPE: str = "sam2.1_hiera_large"
    SAM2_CHECKPOINT: str = ""  # path to .pt
    DEVICE: str = "cuda"  # or "cpu"
    DATA_ROOT: str = "./data"
    FRAME_EXT: str = "png"
    MAX_WORKERS: int = 1
    MASK_OUTPUT_MODE: str = "single"  # or "per_label"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()