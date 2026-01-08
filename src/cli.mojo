"""
CLI argument parsing module.

Simple argument parser for inference service command-line interface.
"""

from sys import argv


struct CliArgs:
    """Parsed command-line arguments."""
    var text: String
    var config_path: String
    var help_requested: Bool
    
    fn __init__(out self, text: String, config_path: String = "config.toml", help_requested: Bool = False):
        self.text = text
        self.config_path = config_path
        self.help_requested = help_requested


fn parse_args() raises -> CliArgs:
    """
    Parse command-line arguments.
    
    Returns:
        CliArgs with parsed arguments.
    
    Raises:
        Error if required arguments are missing or invalid.
    
    Usage:
        pixi run inference --text "Sample text"
        pixi run inference --text "Sample text" --config custom.toml
        pixi run inference --help
    """
    var args = argv()
    var text = String("")
    var config_path = String("config.toml")
    var help_requested = False
    
    # Skip program name (args[0])
    var i = 1
    while i < len(args):
        var arg = args[i]
        
        # Skip -- separator (used by pixi)
        if arg == "--":
            i += 1
            continue
        
        if arg == "--help" or arg == "-h":
            help_requested = True
            i += 1
        elif arg == "--text" or arg == "-t":
            if i + 1 >= len(args):
                raise Error("--text requires a value")
            text = args[i + 1]
            i += 2
        elif arg == "--config" or arg == "-c":
            if i + 1 >= len(args):
                raise Error("--config requires a value")
            config_path = args[i + 1]
            i += 2
        else:
            raise Error("Unknown argument: " + arg)
    
    # Validate required arguments
    if not help_requested and len(text) == 0:
        raise Error("Required argument --text is missing")
    
    return CliArgs(text, config_path, help_requested)


fn print_help():
    """Print usage information."""
    print("mojo-inference-service v0.1.0-dev")
    print("")
    print("USAGE:")
    print("    pixi run inference --text <TEXT> [OPTIONS]")
    print("")
    print("OPTIONS:")
    print("    -t, --text <TEXT>         Text to classify (required)")
    print("    -c, --config <PATH>       Path to config file (default: config.toml)")
    print("    -h, --help                Show this help message")
    print("")
    print("EXAMPLES:")
    print("    pixi run inference --text \"This product is amazing!\"")
    print("    pixi run inference -t \"Terrible experience\" -c custom.toml")
