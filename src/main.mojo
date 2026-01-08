"""
mojo-inference-service - High-performance AI model inference service

Entry point for the inference service application.
"""

from logger import Logger, Level
from sys import argv
from config import load_config, load_secrets
from classifier import SentimentClassifier


fn parse_level(level_str: String) -> Level:
    """Convert string log level to Level enum."""
    # TODO: Implement proper string comparison
    # For now, return INFO
    return Level.INFO


fn main() raises:
    """Main entry point for the inference service."""
    
    # Initial logger for startup
    var startup_log = Logger[Level.INFO]()
    
    startup_log.info("🔥 mojo-inference-service v0.1.0-dev")
    startup_log.info("Starting up...")
    
    # Load configuration
    var config = load_config("config.toml")
    startup_log.info("✅ Configuration loaded")
    startup_log.info("  Model type:", config.model.type)
    startup_log.info("  Algorithm:", config.model.algorithm)
    startup_log.info("  Vocab size:", config.model.vocab_size)
    startup_log.info("  Confidence threshold:", config.inference.confidence_threshold)
    startup_log.info("  Log level:", config.logging.level)
    
    # Load secrets
    var secrets = load_secrets(".env")
    startup_log.info("✅ Secrets loaded")
    
    # Configure logger based on config
    # TODO: Create logger with config.logging.level
    var log = Logger[Level.INFO]()
    
    # Parse command-line arguments
    var args = argv()
    if len(args) > 1:
        var arg_count = len(args) - 1
        log.info("Arguments received:", arg_count)
        for i in range(1, len(args)):
            log.debug("  Arg", i, ":", args[i])
    else:
        log.warning("No arguments provided.")
        log.info("Usage: pixi run inference --text 'Your text here'")
        return
    
    # Initialize classifier
    log.info("Initializing classifier...")
    var classifier = SentimentClassifier(
        confidence_threshold=config.inference.confidence_threshold,
        max_length=config.inference.max_length
    )
    classifier.load()
    log.info("✅ Classifier ready")
    
    # TODO: Parse --text argument and run inference
    # For now, just indicate we're ready
    log.info("✅ Initialization complete. Ready for inference.")
    log.warning("⚠️  CLI argument parsing not yet implemented")
    log.info("Next: Implement --text argument parsing and run classification")
