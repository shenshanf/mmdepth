# MMDepth: A MMEngine-based Toolbox for Monocular, Stereo, and Multi-view Depth Estimation

MMDepth is an open-source toolbox designed to unify various depth estimation approaches within a single framework. Built on PyTorch and the MMEngine infrastructure, it aims to provide a comprehensive solution for monocular, stereo, and multi-view depth estimation tasks. This project follows the modular design philosophy of the OpenMMLab ecosystem, allowing researchers and developers to easily implement, compare, and build upon state-of-the-art depth estimation algorithms.

**Note**: MMDepth is currently in its initial development phase. The repository structure, core interfaces, and basic functionalities are being established. Most algorithms listed in the documentation are planned for implementation in future updates. Your patience and understanding are appreciated as we continue to develop this project.

We are actively looking for collaborators who are passionate about depth estimation to join this project! Whether you're experienced in depth estimation research, familiar with the MMEngine framework, or simply enthusiastic about computer vision, your contributions would be valuable. Please check our [CONTRIBUTING.md](docs/en/CONTRIBUTING.md) for more details on how to get involved, or open an issue to discuss your ideas and suggestions.

## Features (Planned)

- Support for monocular, stereo, and multi-view depth estimation methods
- Modular design for easy extension and customization
- Standardized training and evaluation pipelines
- Comprehensive benchmarking on standard datasets
- Detailed documentation and tutorials

## Project File Structures
the project structure is designed following established practices from the OpenMMLab ecosystem, particularly referencing:

- [MMEngine Template](https://github.com/open-mmlab/mmengine-template): Provides the foundational modular design pattern
- [MMDetection](https://github.com/open-mmlab/mmdetection): For task-specific organization patterns

This structure follows a modular, registry-based design that enables flexibility and extensibility. 
The organization separates core components (models, datasets, evaluation metrics) while maintaining a consistent interface between modules. 
This approach facilitates easy integration of new algorithms, datasets, and metrics while ensuring backward compatibility.

```text
│   ├── version.py            # Version information
│   ├── datasets/             # Dataset loading and processing utilities
│   │   ├── pipline/          # Data processing pipelines
│   ├── engine/               # Core runtime components
│   │   ├── hook/             # Training/validation process hooks
│   │   ├── logging/          # Logging utilities
│   ├── evaluation/           # Evaluation tools
│   │   ├── metrics/          # Evaluation metrics
│   ├── fileio/               # File I/O handlers for various formats
│   ├── infer/                # Inference tools and utilities
│   ├── models/               # Model implementations
│   │   ├── mono_depth/       # Monocular depth estimation models
│   │   ├── mvstereo/         # Multi-view stereo models
│   │   ├── stereo/           # Stereo matching models
│   ├── registry/             # Component registry system
│   ├── structures/           # Data structures for depth representation
│   ├── tests/                # Unit tests
│   ├── utils/                # Utility functions
│   ├── visualization/        # Visualization tools
```
You can generate the complete project structure tree using:
```bash
python /mmdepth/tools/gen_dirtree.py --project_path /path/to/project --output_file project_tree.txt
```



## License

This project is released under the [Apache 2.0 license](LICENSE).
