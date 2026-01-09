"""
max-learning - Learning and experimenting with MAX framework

Entry point for the lexicon-based sentiment classifier (v0.1.0 baseline).
"""

from logger import Logger, Level
from config import load_config, load_secrets
from classifier import SentimentClassifier, classify_sentiment
from cli import parse_args, print_help


fn parse_level(level_str: String) -> Level:
    """Convert string log level to Level enum."""
    # TODO: Implement proper string comparison
    # For now, return INFO
    return Level.INFO


fn main() raises:
    """Main entry point for the inference service."""
    
    # Initial logger for startup
    var startup_log = Logger[Level.INFO]()
    
    startup_log.info("🔥 max-learning v0.2.0 (Mojo lexicon classifier)")
    startup_log.info("Starting up...")
    
    # Parse command-line arguments
    var cli_args = parse_args()
    
    # Show help if requested
    if cli_args.help_requested:
        print_help()
        return
    
    startup_log.info("Input text:", cli_args.text)
    
    # Load configuration
    var config = load_config(cli_args.config_path)
    startup_log.info("✅ Configuration loaded")
    startup_log.info("  Model type:", config.model.type)
    startup_log.info("  Algorithm:", config.model.algorithm)
    startup_log.info("  Vocab size:", config.model.vocab_size)
    startup_log.info("  Confidence threshold:", config.inference.confidence_threshold)
    startup_log.info("  Log level:", config.logging.level)
    
    # Load secrets
    var secrets = load_secrets(".env")
    startup_log.info("✅ Secrets loaded")
    if len(secrets.huggingface_api_key) > 0:
        startup_log.info("  HuggingFace API key: [REDACTED]")
    if len(secrets.auth_token) > 0:
        startup_log.info("  Auth token: [REDACTED]")
    
    # Configure logger based on config
    var log = Logger[Level.INFO]()
    
    # Initialize classifier
    log.info("Initializing classifier...")
    var classifier = SentimentClassifier(
        confidence_threshold=config.inference.confidence_threshold,
        max_length=config.inference.max_length
    )
    classifier.load()
    log.info("✅ Classifier ready")
    
    # Run inference
    log.info("Running inference...")
    var result = classify_sentiment(classifier, cli_args.text)
    
    # Display results
    print("")
    print("═" * 50)
    print("SENTIMENT ANALYSIS RESULT")
    print("═" * 50)
    print("Input:", cli_args.text)
    print("")
    print("Label:", result.label)
    print("Confidence:", result.confidence)
    print("Score:", result.score)
    print("═" * 50)
    
    log.info("✅ Inference complete")
