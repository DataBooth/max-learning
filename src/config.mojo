"""
Configuration management module.

Loads configuration from config.toml and secrets from .env using
mojo-toml and mojo-dotenv libraries.
"""

from pathlib import Path
from logger import Logger, Level
from toml import parse
from dotenv import dotenv_values


struct ModelConfig(Copyable, Movable):
    """Configuration for the ML model."""
    var type: String
    var algorithm: String
    var vocab_size: Int
    var embedding_dim: Int
    
    fn __init__(out self, type: String, algorithm: String, vocab_size: Int, embedding_dim: Int):
        self.type = type
        self.algorithm = algorithm
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    fn copy(self) -> Self:
        return ModelConfig(self.type, self.algorithm, self.vocab_size, self.embedding_dim)


struct InferenceConfig(Copyable, Movable):
    """Configuration for inference settings."""
    var confidence_threshold: Float64
    var max_length: Int
    
    fn __init__(out self, confidence_threshold: Float64, max_length: Int):
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
    
    fn copy(self) -> Self:
        return InferenceConfig(self.confidence_threshold, self.max_length)


struct LoggingConfig(Copyable, Movable):
    """Configuration for logging."""
    var level: String
    
    fn __init__(out self, level: String):
        self.level = level
    
    fn copy(self) -> Self:
        return LoggingConfig(self.level)


struct ServerConfig(Copyable, Movable):
    """Configuration for HTTP server (future)."""
    var host: String
    var port: Int
    var workers: Int
    
    fn __init__(out self, host: String, port: Int, workers: Int):
        self.host = host
        self.port = port
        self.workers = workers
    
    fn copy(self) -> Self:
        return ServerConfig(self.host, self.port, self.workers)


struct AppConfig(Copyable, Movable):
    """Complete application configuration."""
    var model: ModelConfig
    var inference: InferenceConfig
    var logging: LoggingConfig
    var server: ServerConfig
    
    fn __init__(out self, model: ModelConfig, inference: InferenceConfig, logging: LoggingConfig, server: ServerConfig):
        self.model = model.copy()
        self.inference = inference.copy()
        self.logging = logging.copy()
        self.server = server.copy()


struct Secrets(Copyable, Movable):
    """Secrets loaded from .env file."""
    var huggingface_api_key: String
    var auth_token: String
    
    fn __init__(out self, huggingface_api_key: String, auth_token: String):
        self.huggingface_api_key = huggingface_api_key
        self.auth_token = auth_token


fn load_config(config_path: String = "config.toml") raises -> AppConfig:
    """
    Load application configuration from TOML file.
    
    Args:
        config_path: Path to the TOML configuration file.
    
    Returns:
        AppConfig struct with parsed configuration.
    
    Raises:
        Error if configuration file cannot be read or parsed.
    """
    var log = Logger[Level.INFO]()
    log.info("Loading configuration from:", config_path)
    
    # Check if file exists
    var path = Path(config_path)
    if not path.exists():
        raise Error("Configuration file not found: " + config_path)
    
    # Read file content
    var content = path.read_text()
    
    # Parse TOML
    var toml_data = parse(content)
    
    log.info("✅ TOML parsed successfully")
    
    # Extract configuration from parsed TOML
    var model_table = toml_data["model"].as_table()
    var model = ModelConfig(
        type=model_table["type"].as_string(),
        algorithm=model_table["algorithm"].as_string(),
        vocab_size=model_table["vocab_size"].as_int(),
        embedding_dim=model_table["embedding_dim"].as_int()
    )
    
    var inference_table = toml_data["inference"].as_table()
    var inference = InferenceConfig(
        confidence_threshold=inference_table["confidence_threshold"].as_float(),
        max_length=inference_table["max_length"].as_int()
    )
    
    var logging_table = toml_data["logging"].as_table()
    var logging = LoggingConfig(level=logging_table["level"].as_string())
    
    var server_table = toml_data["server"].as_table()
    var server = ServerConfig(
        host=server_table["host"].as_string(),
        port=server_table["port"].as_int(),
        workers=server_table["workers"].as_int()
    )
    
    return AppConfig(model, inference, logging, server)


fn load_secrets(env_path: String = ".env") raises -> Secrets:
    """
    Load secrets from .env file.
    
    Args:
        env_path: Path to the .env file.
    
    Returns:
        Secrets struct with loaded environment variables.
    
    Raises:
        Error if .env file cannot be read or parsed.
    """
    var log = Logger[Level.INFO]()
    log.info("Loading secrets from:", env_path)
    
    # Check if file exists
    var path = Path(env_path)
    if not path.exists():
        log.warning("⚠️  .env file not found, using empty secrets")
        return Secrets(
            huggingface_api_key="",
            auth_token=""
        )
    
    # Load environment variables from .env file
    var env_vars = dotenv_values(env_path)
    
    # Extract secrets with defaults
    var hf_key = env_vars.get("HUGGINGFACE_API_KEY", "")
    var auth = env_vars.get("AUTH_TOKEN", "")
    
    log.info("✅ .env parsed successfully")
    
    return Secrets(
        huggingface_api_key=hf_key,
        auth_token=auth
    )
