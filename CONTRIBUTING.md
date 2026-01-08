# Contributing to mojo-inference-service

Thank you for your interest in contributing! This project is part of DataBooth's exploration of high-performance AI infrastructure with Mojo.

## Development Setup

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/databooth/mojo-inference-service.git
   cd mojo-inference-service
   ```

3. **Install dependencies**:
   ```bash
   pixi install
   ```

4. **Run the application**:
   ```bash
   pixi run inference
   ```

## Development Workflow

### Running Tests

```bash
pixi run test              # Run all tests
pixi run test-config       # Run config tests only
pixi run test-inference    # Run inference tests only
```

### Code Formatting

```bash
pixi run format            # Format code
pixi run check             # Check code quality
```

### Building

```bash
pixi run build             # Build standalone binary
```

## Project Structure

- `src/` - Source code
  - `main.mojo` - Entry point
  - `config.mojo` - Configuration management
  - `inference.mojo` - Model inference logic
- `tests/` - Test suite
- `examples/` - Example usage
- `benchmarks/` - Performance benchmarks
- `models/` - Model files (gitignored)

## Coding Guidelines

1. **Follow Mojo best practices** - Use idiomatic Mojo patterns
2. **Document thoroughly** - Include docstrings for all public functions/structs
3. **Write tests** - Add tests for new functionality
4. **Performance matters** - This is a performance-focused project
5. **Use Australian English** - For documentation and comments

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Format

```
<type>: <subject>

<body>

Co-Authored-By: Warp <agent@warp.dev>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Questions?

Open an issue or reach out via [DataBooth](https://www.databooth.com.au).
