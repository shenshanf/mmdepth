import os
import inspect
import importlib
import pkgutil
from functools import wraps
from typing import Type, TypeVar, Any, Optional

# Generic type variable for class type hints
T = TypeVar('T')


class BackendManager:
    """A decorator that enables dynamic backend selection for classes.

    This decorator implements a plugin-like architecture that allows classes to dynamically
    switch their implementation by changing their parent class at runtime based on a
    specified backend. This is particularly useful for providing multiple implementations
    of the same interface.

    Implementation Details:
    - The decorator modifies the class's __init__ method to intercept the backend selection
    - It dynamically loads the implementation class from a corresponding backend package
    - It updates the class's base1 classes (inheritance) at runtime

    Directory Structure Requirements:
    root/
    ├── your_module.py (contains the decorated class)
    ├── backend1/
    │   └── __init__.py (contains implementation for backend1)
    └── backend2/
        └── __init__.py (contains implementation for backend2)

    Example Usage:
        # In root/your_module.py
        @BackendManager(default_backend='backend1')
        class MyClass:
            def __init__(self, x, backend='backend1'):
                super().__init__(x)

        # In root/backend1/__init__.py
        class MyClass:
            def __init__(self, x):
                self.x = x

        # Usage
        obj = MyClass(x=10, backend='backend1')  # Uses backend1 implementation
        obj = MyClass(x=10, backend='backend2')  # Uses backend2 implementation
    """

    def __init__(self, default_backend: Optional[str] = None) -> None:
        """Initialize the backend manager decorator.

        Args:
            default_backend: The default backend package name to use if none specified
                           during class instantiation.
        """
        self.default_backend = default_backend

    def __call__(self, cls: Type[T]) -> Type[T]:
        """Decorate the class to enable dynamic inheritance.

        This method wraps the original class's __init__ method to:
        1. Extract and validate the backend parameter
        2. Load the corresponding backend implementation
        3. Modify the class's inheritance hierarchy
        4. Initialize the instance with the backend implementation

        Args:
            cls: The class to be decorated

        Returns:
            The decorated class with dynamic inheritance capability
        """
        original_init = cls.__init__
        default_backend = self.default_backend

        @wraps(original_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            # Extract backend argument, use default if not provided
            backend = kwargs.pop('backend', default_backend)

            # Get the root directory where backend packages should be located
            try:
                current_file = inspect.getfile(cls)
                directory = os.path.dirname(os.path.abspath(current_file))
            except TypeError:
                raise TypeError(f"Cannot determine file location for class {cls.__name__}")

            # Discover available backend packages in the directory
            backends = [
                name for _, name, is_pkg in pkgutil.iter_modules([directory])
                if is_pkg
            ]

            # Validate backend availability
            if not backends:
                raise RuntimeError(
                    f"No backend packages found in directory: {directory}"
                )

            if backend not in backends:
                raise NotImplementedError(
                    f"Backend '{backend}' not implemented. "
                    f"Available backends: {', '.join(backends)}"
                )

            try:
                # Import the backend package dynamically
                # Note: The decorated class must be in the same root path as the backend packages
                package = inspect.getmodule(cls).__name__.rsplit('.', 1)[0]
                backend_module = importlib.import_module(f'.{backend}', package=package)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import backend '{backend}': {str(e)}"
                )

            # Get the implementation class from the backend module
            impl_cls_name = cls.__name__
            impl_cls = getattr(backend_module, impl_cls_name, None)

            if impl_cls is None:
                raise NotImplementedError(
                    f"Class '{impl_cls_name}' not implemented in backend '{backend}'"
                )

            # Dynamically modify the class's inheritance by replacing its base1 classes
            # This is the key mechanism for dynamic inheritance
            cls.__bases__ = (impl_cls,)

            # Initialize the instance using the backend implementation
            impl_cls.__init__(self, *args, **kwargs)

            # Store the backend name for reference
            cls.backend = backend

        cls.__init__ = __init__
        return cls
